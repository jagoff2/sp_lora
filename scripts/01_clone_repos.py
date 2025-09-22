import os, yaml, subprocess, pathlib
from dotenv import load_dotenv

load_dotenv()
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
SOURCES = yaml.safe_load((DATA / "sources.yaml").read_text())

RAW.mkdir(exist_ok=True, parents=True)

def clone_or_update(name, url, branch):
    repo_dir = RAW / name
    if repo_dir.exists():
        print(f"Updating {name}...")
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "--all", "--tags"], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "checkout", branch], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
    else:
        print(f"Cloning {name}...")
        subprocess.run(["git", "clone", "--branch", branch, "--single-branch", url, str(repo_dir)], check=True)

def main():
    for repo in SOURCES.get("git_repos", []):
        clone_or_update(repo["name"], repo["url"], repo.get("branch", "master"))
    print("Done. Repos ready under data/raw/")

if __name__ == "__main__":
    main()
