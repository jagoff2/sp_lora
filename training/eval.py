import json, pathlib, os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts" / "07_eval_questions.jsonl"

def load_qs():
    out = []
    if not EVAL.exists():
        return [{"query":"Where is README.md located?", "want_contains":"README.md"}]
    for line in open(EVAL,"r",encoding="utf-8"):
        out.append(json.loads(line))
    return out

def ask(pipe, q):
    prompt = f"User: {q}\nAssistant:"
    out = pipe(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
    return out.split("Assistant:",1)[-1].strip()

def main():
    model_id = os.environ.get("BASE_MODEL_ID","gpt-oss-20b")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=\"auto\", device_map=\"auto\")
    pipe = pipeline(\"text-generation\", model=model, tokenizer=tok)
    total = 0
    correct = 0
    for itm in load_qs():
        total += 1
        ans = ask(pipe, itm[\"query\"])
        ok = itm[\"want_contains\"].lower() in ans.lower()
        print(\"Q:\", itm[\"query\"])
        print(\"A:\", ans)
        print(\"OK:\", ok)
        correct += int(ok)
    print(f\"Score: {correct}/{total}\")

if __name__ == \"__main__\":
    main()
