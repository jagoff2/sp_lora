import pathlib, os, re, json
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUT = DATA / "processed" / "chunks"
OUT.mkdir(parents=True, exist_ok=True)

def text_chunks(text, meta, max_words=280):
    words = text.split()
    chunk = []
    for w in words:
        chunk.append(w)
        if len(chunk) >= max_words:
            yield " ".join(chunk), meta
            chunk = []
    if chunk:
        yield " ".join(chunk), meta

def html_to_text(p):
    html = p.read_text(errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script","style","nav","footer","header"]):
        s.extract()
    return md(str(soup))

def scan_repo_texts():
    for repo_dir in RAW.glob("*"):
        if not repo_dir.is_dir():
            continue
        for p in repo_dir.rglob("*"):
            if p.is_dir():
                continue
            if any(p.name.endswith(ext) for ext in [".md",".rst",".txt",".py",".cpp",".cc",".c",".h",".hpp",".json",".yaml",".yml",".ini",".toml",".sh"]):
                try:
                    txt = p.read_text(errors="ignore")
                    yield txt, {"source":"repo", "path":str(p.relative_to(RAW)), "repo":repo_dir.name}
                except Exception:
                    pass

def scan_processed_html():
    for p in PROCESSED.rglob("*.html"):
        try:
            txt = html_to_text(p)
            yield txt, {"source":"html", "path":str(p.relative_to(PROCESSED))}
        except Exception:
            pass

def main():
    i = 0
    out_path = OUT / "corpus.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for txt, meta in tqdm(list(scan_repo_texts()) + list(scan_processed_html())):
            for chunk, m in text_chunks(txt, meta, max_words=280):
                rec = {"id": i, "text": chunk, "meta": m}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                i += 1
    print("Wrote", i, "chunks to", out_path)

if __name__ == "__main__":
    main()
