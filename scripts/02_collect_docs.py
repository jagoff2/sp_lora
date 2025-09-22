import pathlib, yaml, requests, time, re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
# ensure UTF-8 when reading YAML (in case this ever contains non-ASCII)
SOURCES = yaml.safe_load((DATA / "sources.yaml").read_text(encoding="utf-8"))

PROCESSED.mkdir(exist_ok=True, parents=True)

def fetch(url):
    print("GET", url)
    r = requests.get(url, timeout=30, headers={"User-Agent":"sunnypilot-expert/0.1"})
    r.raise_for_status()
    return r.text  # keep text; we'll write as UTF-8 below

def save_text(name, path_rel, text):
    out_dir = PROCESSED / name / path_rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # write as UTF-8 to avoid UnicodeEncodeError on Windows
    (out_dir / path_rel.name).write_text(text, encoding="utf-8")

def crawl_wiki_from_repo(name, url):
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")
    save_text(name, pathlib.Path("index.html"), html)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("#"):
            continue
        if href.startswith("http") and "github.com" not in href:
            continue
        if "/wiki/" not in href:
            continue
        page_url = urljoin(url, href)
        try:
            page_html = fetch(page_url)
            slug = re.sub(r"[^a-zA-Z0-9._/-]", "_", href.split("/wiki/")[-1])
            save_text(name, pathlib.Path(slug + ".html"), page_html)
            time.sleep(0.2)
        except Exception as e:
            print("skip", page_url, e)

def crawl_web_pages(pages):
    for url in pages:
        try:
            html = fetch(url)
            host = re.sub(r"[^a-zA-Z0-9._-]", "_", url.replace("https://","").replace("http://",""))
            save_text("web", pathlib.Path(host + ".html"), html)
            time.sleep(0.2)
        except Exception as e:
            print("skip", url, e)

def main():
    for doc in SOURCES.get("docs", []):
        if doc.get("type") == "github_wiki":
            crawl_wiki_from_repo(doc["name"], doc["url"])
    crawl_web_pages(SOURCES.get("web_pages", []))

if __name__ == "__main__":
    main()
