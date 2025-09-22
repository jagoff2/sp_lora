import os
import json
import argparse
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

import yaml
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Optional Llama tokenizer fallbacks for BitConfig-based checkpoints
try:
    from transformers import LlamaTokenizerFast, LlamaTokenizer
    _HAVE_LLAMA_TOK = True
except Exception:
    _HAVE_LLAMA_TOK = False

# -------- Optional PEFT / LoRA --------
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        prepare_model_for_kbit_training,
    )
    _HAVE_PEFT = True
except Exception:
    _HAVE_PEFT = False


# -------- JSONL loader (fast & tolerant) --------
def _loads(line: str) -> Dict[str, Any]:
    line = line.strip()
    if not line:
        return {}
    try:
        return json.loads(line)
    except Exception:
        try:
            import orjson
            return orjson.loads(line)
        except Exception:
            raise


@dataclass
class TokenizedExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


class JsonlCausalDataset(Dataset):
    """
    Accepts JSONL with either:
      - {"text": "<flattened prompt/response text>"}
      - {"messages": [{"role": "...", "content": "..."}, ...]}
    """

    def __init__(self, path: str, tokenizer, max_len: int):
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples: List[TokenizedExample] = []
        self._build()

    def _flatten_messages(self, messages: List[Dict[str, str]]) -> str:
        bufs = []
        for m in messages:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            bufs.append(f"{role}: {content}" if role else content)
        return "\n".join(bufs).strip()

    def _build(self):
        p = pathlib.Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"dataset_path not found: {self.path}")

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                obj = _loads(line)
                if not obj:
                    continue
                if "text" in obj and obj["text"]:
                    text = str(obj["text"])
                elif "messages" in obj and obj["messages"]:
                    text = self._flatten_messages(obj["messages"])
                else:
                    continue

                enc = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_len,
                    padding=False,
                )
                input_ids = enc["input_ids"]
                attn = enc.get("attention_mask", [1] * len(input_ids))

                self.examples.append(
                    TokenizedExample(
                        input_ids=input_ids,
                        attention_mask=attn,
                        labels=list(input_ids),
                    )
                )

        if not self.examples:
            raise ValueError(f"No usable examples parsed from: {self.path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(ex.attention_mask, dtype=torch.long),
            "labels": torch.tensor(ex.labels, dtype=torch.long),
        }


def build_bnb_config(cfg: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    use_4 = bool(cfg.get("load_in_4bit", False))
    use_8 = bool(cfg.get("load_in_8bit", False))
    if not (use_4 or use_8):
        return None
    compute_dtype = getattr(torch, str(cfg.get("bnb_4bit_compute_dtype", "float16")))
    return BitsAndBytesConfig(
        load_in_4bit=use_4,
        load_in_8bit=use_8,
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
    )


def maybe_apply_lora(model, cfg: Dict[str, Any], using_bnb: bool):
    if not bool(cfg.get("use_lora", True)):
        return model
    if not _HAVE_PEFT:
        raise RuntimeError("use_lora=True but 'peft' is not installed. pip install peft")
    if using_bnb:
        model = prepare_model_for_kbit_training(model)
    target_modules = cfg.get(
        "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    lconf = LoraConfig(
        r=int(cfg.get("lora_r", 32)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias=cfg.get("lora_bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    return get_peft_model(model, lconf)


def load_tokenizer(tok_path: str, trust_remote_code: bool, max_len: int):
    """
    Handle BitConfig models by falling back to Llama tokenizers when AutoTokenizer mapping fails.
    """
    tried = []
    try:
        tok = AutoTokenizer.from_pretrained(tok_path, use_fast=True, trust_remote_code=trust_remote_code)
        # Success path
    except KeyError as e:
        tried.append(f"AutoTokenizer(use_fast=True) -> {e}")
        if not _HAVE_LLAMA_TOK:
            raise
        try:
            tok = LlamaTokenizerFast.from_pretrained(tok_path, use_fast=True)
        except Exception as e2:
            tried.append(f"LlamaTokenizerFast -> {e2}")
            tok = LlamaTokenizer.from_pretrained(tok_path, use_fast=False)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if getattr(tok, "model_max_length", None) is None or tok.model_max_length > 1_000_000:
        tok.model_max_length = max_len
    return tok


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_file, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    model_path = cfg["model_name_or_path"]
    tok_path = cfg.get("tokenizer_name_or_path", model_path)
    trust_remote_code = bool(cfg.get("trust_remote_code", True))

    max_len = int(cfg.get("max_seq_length", 2048))
    dataset_path = cfg.get("dataset_path", "training/sft_data.jsonl")
    output_dir = cfg.get("output_dir", "outputs/sft")

    # ---------- Tokenizer (with Bit→Llama fallback) ----------
    tok = load_tokenizer(tok_path, trust_remote_code, max_len)

    # ---------- Dataset ----------
    train_ds = JsonlCausalDataset(dataset_path, tok, max_len=max_len)

    # ---------- Quantization checks & config ----------
    auto_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    preq = getattr(auto_cfg, "quantization_config", None)
    quant_method = (preq or {}).get("quant_method", "").lower()
    is_non_bnb_prequant = bool(preq) and quant_method not in ("bitsandbytes", "bnb", "bnb_4bit")
    if is_non_bnb_prequant:
        raise ValueError(
            f"Checkpoint at {model_path} is pre-quantized with {quant_method}; "
            "use a BnB-4bit base or disable k-bit loading."
        )

    bnb_cfg = build_bnb_config(cfg)
    using_bnb = bnb_cfg is not None

    # ---------- Model ----------
    dtype_str = str(cfg.get("dtype", "float16"))
    torch_dtype = getattr(torch, dtype_str, torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        device_map=cfg.get("device_map", "auto"),
        torch_dtype=torch_dtype,
        quantization_config=bnb_cfg,
        low_cpu_mem_usage=True,
    )

    if bool(cfg.get("gradient_checkpointing", True)):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    # ---------- LoRA ----------
    model = maybe_apply_lora(model, cfg, using_bnb=using_bnb)

    # ---------- Training Arguments ----------
    args_tr = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "cosine")),
        num_train_epochs=float(cfg.get("num_train_epochs", 1.0)),
        max_steps=int(cfg.get("max_steps", -1)),
        logging_steps=int(cfg.get("logging_steps", 50)),
        save_steps=int(cfg.get("save_steps", 500)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        fp16=bool(cfg.get("fp16", True)),
        bf16=bool(cfg.get("bf16", False)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 0)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        optim=str(cfg.get("optim", "paged_adamw_32bit")),
        report_to=cfg.get("report_to", []),
        logging_first_step=True,
        remove_unused_columns=False,
    )

    # ---------- Collator & Trainer ----------
    collator = DataCollatorForLanguageModeling(tok, mlm=False, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Disable Flash-Attention on Windows by default
    os.environ.setdefault("FLASH_ATTENTION_DISABLE", "1")
    # Enable TF32 matmul where available
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
