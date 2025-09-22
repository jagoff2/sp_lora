# Sunnypilot Expert Starter

Turn `gpt-oss-20b` into a Sunnypilot expert using Retrieval-Augmented Generation plus a light LoRA fine-tune.

## What you get

1. Data pipeline to build a "developer brain" corpus from the Sunnypilot repo, issues, and docs.
2. A FAISS-based RAG service with a strict answer template and inline citations.
3. LoRA fine-tuning scripts to teach style and domain patterns.
4. A tiny eval suite to measure correctness and citation fidelity.

## Quick start

### 0) System requirements
- Python 3.10+
- 32GB RAM recommended for indexing large repos
- GPU recommended for inference/fine-tune (24GB+ VRAM is ideal)
- Git installed

### 1) Install
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit values
```

### 2) Build the corpus and index
```
python scripts/01_clone_repos.py
python scripts/02_collect_docs.py
python scripts/03_parse_and_chunk.py
python scripts/04_build_index.py
```

### 3) Run the RAG API
```
uvicorn rag.server:app --reload --port 8000
```

POST to `/chat` with:
```json
{"query": "How do I enable feature X in Sunnypilot?", "version": "latest"}
```

### 4) Prepare and run LoRA SFT
```
python training/sft_prepare.py
accelerate launch training/sft_train.py --config_file training/config/sft_config.yaml
```

### 5) Evaluate
```
python training/eval.py
```

## Notes
- This project does **not** ship Sunnypilot content. The scripts clone and snapshot sources you point to in `data/sources.yaml`.
- Start with RAG only. Add LoRA after you have a solid corpus and answer style.
