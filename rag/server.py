from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, pathlib, orjson, os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import hnswlib  # <- replaces faiss

load_dotenv()
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "gpt-oss-20b")
EMB_MODEL_ID = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
HNSW_EF = int(os.getenv("HNSW_EF", "256"))  # query-time accuracy/latency knob

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"

app = FastAPI()

# Embeddings
emb = SentenceTransformer(EMB_MODEL_ID)
emb_dim = emb.get_sentence_embedding_dimension()

# HNSW index (built with space="ip" and normalized vectors)
index = hnswlib.Index(space="ip", dim=emb_dim)
index.load_index(str(INDEX_DIR / "hnsw.index"))
index.set_ef(HNSW_EF)

# Sidecar data
with open(INDEX_DIR / "metas.jsonl", "r", encoding="utf-8") as f:
    metas = [orjson.loads(l) for l in f]
with open(INDEX_DIR / "texts.jsonl", "r", encoding="utf-8") as f:
    texts = [orjson.loads(l) for l in f]

_model = None
_tokenizer = None
_pipe = None

class ChatReq(BaseModel):
    query: str
    version: str = "latest"
    k: int = 6
    max_new_tokens: int = 512

def ensure_model():
    global _model, _tokenizer, _pipe
    if _pipe is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype="auto", device_map="auto"
        )
        _pipe = pipeline("text-generation", model=_model, tokenizer=_tokenizer)
    return _pipe

ANSWER_TEMPLATE = """You are Sunnypilot's developer assistant. Answer precisely. If you cite facts, include short inline citations like [repo:path/to/file] or [wiki:Page#Section].
If a claim is not in the provided CONTEXT, say that it was not found. Prefer file names, function names, config paths, defaults, and version tags.

Question: {question}

CONTEXT:
{context}

Answer:
"""

def retrieve(query, k=6):
    q = emb.encode([query], normalize_embeddings=True).astype(np.float32)
    labels, scores = index.knn_query(q, k=k)
    ids = labels[0]
    ctxs = []
    for i in ids:
        meta = metas[i]
        text = texts[i]
        ctxs.append(f"{text}\nMETA={meta}")
    return ctxs

@app.post("/chat")
def chat(req: ChatReq):
    ctxs = retrieve(req.query, k=req.k)
    context_joined = "\n---\n".join(ctxs)
    prompt = ANSWER_TEMPLATE.format(question=req.query, context=context_joined)
    pipe = ensure_model()
    out = pipe(prompt, max_new_tokens=req.max_new_tokens, do_sample=False)[0]["generated_text"]
    if "Answer:" in out:
        out = out.split("Answer:", 1)[1].strip()
    return {"answer": out, "used_context": ctxs}
