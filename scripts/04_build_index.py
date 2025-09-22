import json, pathlib
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import orjson
import hnswlib  # <- replace faiss

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHUNKS = DATA / "processed" / "chunks" / "corpus.jsonl"
INDEX_DIR = DATA / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    model = SentenceTransformer(EMB_MODEL)
    texts, metas = [], []
    with open(CHUNKS, "r", encoding="utf-8") as f:
        for line in f:
            obj = orjson.loads(line)
            texts.append(obj["text"])
            metas.append(obj["meta"])

    print("Embedding", len(texts), "chunks")
    embs = model.encode(
        texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)

    d = embs.shape[1]
    num_elements = embs.shape[0]

    # HNSW index with inner product (cosine for normalized vectors)
    p = hnswlib.Index(space="ip", dim=d)
    # Construction params: M (graph degree) and ef_construction (build-time accuracy)
    p.init_index(max_elements=num_elements, ef_construction=200, M=48)
    # You can set num_threads to control parallelism
    p.add_items(embs, ids=np.arange(num_elements), num_threads=0)

    # Query-time recall/latency tradeoff
    p.set_ef(256)

    # Save the index
    p.save_index(str(INDEX_DIR / "hnsw.index"))

    # Persist sidecar files (same as your code)
    with open(INDEX_DIR / "metas.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(orjson.dumps(m).decode("utf-8") + "\n")
    with open(INDEX_DIR / "texts.jsonl", "w", encoding="utf-8") as f:
        for t in texts:
            f.write(orjson.dumps(t).decode("utf-8") + "\n")

    print("Index built at", INDEX_DIR)

if __name__ == "__main__":
    main()
