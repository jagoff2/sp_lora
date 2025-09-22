import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Iterable, Tuple, Optional
from collections import defaultdict, Counter
# --- add/replace near the top of the file ---

import argparse  # NEW

random.seed(1337)

# Default: current dir, but we’ll override via CLI
DEFAULT_ROOTS = [Path(".").resolve()]

OUT_PATH = Path("training/sft_data.jsonl")
MAX_EXAMPLES = 12000  # was 30000; ~10–12k is a sweet spot for your QA style

# Rebalanced buckets (sum ~= 1.0)
BUCKET_WEIGHTS = {
    "locate_path":       0.22,  # ↓ (you had ~41% realized)
    "list_dir":          0.07,
    "symbol_def":        0.20,  # ↑
    "symbol_refs":       0.16,  # ↑
    "string_search":     0.08,
    "repo_facts":        0.06,  # ↑
    "not_found":         0.08,
    "function_summary":  0.07,  # keep decent, summaries are short
    "class_summary":     0.04,
    "build_targets":     0.02,  # expect few, keep small
    "dependency_facts":  0.02,
    "language_detect":   0.02,
    "hdr_impl_pairs":    0.06,  # NEW bucket
}

# Avoid floods from a single dir/file.
PER_DIR_CAP = 45
PER_FILE_CAP = 3

# Exclude heavy vendor dirs (edit to taste)
EXCLUDE_DIRS = {
    "third_party", "external", "vendor", ".git", ".github", ".venv",
    "build", "dist", "__pycache__", ".idea", ".vscode"
}

# --- add this helper ---
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", default=None,
                    help="Path to a repo root. Repeat for multiple repos.")
    ap.add_argument("--out", default=str(OUT_PATH),
                    help="Output JSONL file (default training/sft_data.jsonl)")
    ap.add_argument("--max-examples", type=int, default=MAX_EXAMPLES,
                    help="Cap on total examples")
    ap.add_argument("--exclude", action="append", default=[],
                    help="Extra directory names to exclude (basename match)")
    return ap.parse_args()


# File extensions we consider.
INCLUDE_EXTS = {
    ".h", ".hpp", ".hh", ".hxx",
    ".c", ".cc", ".cpp", ".cxx",
    ".py", ".cmake", ".txt", ".md",
    ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".mk", ".sh", ".bat",
    ".js", ".ts", ".tsx", ".jsx",
}

# -------------------- REGEXES --------------------
SYMBOL_PATTERNS = {
    "cpp_class": re.compile(r"^\s*(?:class|struct)\s+([A-Za-z_]\w*)", re.M),
    "cpp_enum":  re.compile(r"^\s*enum(?:\s+class)?\s+([A-Za-z_]\w*)", re.M),
    "cpp_fn":    re.compile(r"^\s*[A-Za-z_][\w:<>,\s\*&]+?\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:\{|;)", re.M),
    "py_def":    re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.M),
    "py_class":  re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*[:\(]", re.M),
}

INCLUDE_REF_HINTS = [
    (re.compile(r'#include\s*["<]([^">]+)[">]'), "include"),
    (re.compile(r'^\s*import\s+([A-Za-z_][\w\.]*)', re.M), "import"),
    (re.compile(r'^\s*from\s+([A-Za-z_][\w\.]*)\s+import', re.M), "import"),
    (re.compile(r'add_executable\s*\(([^)]+)\)', re.M), "cmake_exe"),
    (re.compile(r'add_library\s*\(([^)]+)\)', re.M),    "cmake_lib"),
]

C_COMMENT_LINE = re.compile(r"^\s*//\s?(.*)$")
C_COMMENT_BLOCK = re.compile(r"/\*+(.*?)\*+/", re.S)
PY_DOCSTRING = re.compile(r"^[ \t]*def[ \t]+\w+[^\n]*:\s*\n[ \t]*([\"']{3})(.*?)(\1)", re.S)
PY_CLASS_DOCSTRING = re.compile(r"^[ \t]*class[ \t]+\w+[^\n]*:\s*\n[ \t]*([\"']{3})(.*?)(\1)", re.S)

# Strings we like to search for (flags/macros/tokens).
INTERESTING_TOKEN_RX = re.compile(
    r"(SNPE|ZDL|UDL|RAYLIB|RUNTIME|ENABLE_[A-Z_]+|DISABLE_[A-Z_]+|TODO|FIXME|"
    r"ERROR|WARNING|DEBUG|TRACE|PLATFORM|DLSystem|SNPE|PSNPE|Tensor|Buffer|Runtime)"
)

LANG_EXT_MAP = {
    "C/C++": {".c",".cc",".cpp",".cxx",".h",".hpp",".hh",".hxx"},
    "Python": {".py"},
    "CMake": {".cmake",},
    "JS/TS": {".js",".jsx",".ts",".tsx"},
    "Config/Docs": {".json",".yaml",".yml",".toml",".ini",".cfg",".md",".txt",".mk",".sh",".bat"},
}

# -------------------- HELPERS --------------------
def walk_files(roots: List[Path]) -> List[Path]:
    files = []
    for r in roots:
        for p in r.rglob("*"):
            if p.is_file():
                # Skip excluded dirs early
                parts = {q for q in p.parts}
                if parts & EXCLUDE_DIRS:
                    continue
                if p.suffix.lower() in INCLUDE_EXTS:
                    files.append(p)
    return files


def repo_rel(p: Path, roots: List[Path]) -> str:
    p = p.resolve()
    for r in roots:
        try:
            return str(p.relative_to(r))
        except Exception:
            continue
    return str(p.name)

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def norm_path(rel: str) -> str:
    return rel.replace("\\", "/")

def uniq_norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

# -------------------- DATA STRUCTS --------------------
DUPE_WINDOW_NORM = set()

def add_example(q: str, a: str, out: List[Dict]) -> bool:
    key = uniq_norm(q) + "||" + uniq_norm(a)
    if key in DUPE_WINDOW_NORM:
        return False
    DUPE_WINDOW_NORM.add(key)
    out.append({"messages":[
        {"role":"user","content": q},
        {"role":"assistant","content": a}
    ]})
    return True

# -------------------- EXTRACTION --------------------
def harvest_symbols(text: str) -> Dict[str, List[str]]:
    syms = defaultdict(list)
    for tag, rx in SYMBOL_PATTERNS.items():
        for m in rx.findall(text):
            if len(m) >= 3:
                syms[tag].append(m)
    return syms

def extract_c_like_summary(text: str, name: str, context_lines: int = 6) -> Optional[str]:
    """
    Pull short header-style summary immediately preceding a declaration for C/C++.
    """
    # Find the line with the symbol
    lines = text.splitlines()
    name_rx = re.compile(rf"\b{name}\b")
    for i, line in enumerate(lines):
        if name_rx.search(line):
            # Search upward for // comments or block comment ending near declaration
            start = max(0, i - context_lines)
            chunk = "\n".join(lines[start:i])
            # Priority: block comment closest to decl
            mblock = list(C_COMMENT_BLOCK.finditer(chunk))
            if mblock:
                summary = mblock[-1].group(1).strip()
                summary = re.sub(r"\s+", " ", summary)
                if summary:
                    return summary[:240]
            # Else collect // lines
            comments = [m.group(1).strip() for m in map(C_COMMENT_LINE.match, chunk.splitlines()) if m]
            comments = [c for c in comments if c]
            if comments:
                summary = " ".join(comments)
                return summary[:240]
            break
    return None

def extract_py_docstrings(text: str) -> Tuple[Dict[str,str], Dict[str,str]]:
    fns = {}
    classes = {}
    for m in PY_DOCSTRING.finditer(text):
        body = m.group(2).strip()
        if body:
            # extract def name
            header = text.rfind("\n", 0, m.start())
            # find the "def name(" on previous lines
            name_m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", text[max(0, header-200):m.start()])
            if name_m:
                fns[name_m.group(1)] = re.sub(r"\s+", " ", body)[:240]
    for m in PY_CLASS_DOCSTRING.finditer(text):
        body = m.group(2).strip()
        if body:
            header = text.rfind("\n", 0, m.start())
            name_m = re.search(r"class\s+([A-Za-z_]\w*)\s*[:\(]", text[max(0, header-200):m.start()])
            if name_m:
                classes[name_m.group(1)] = re.sub(r"\s+", " ", body)[:240]
    return fns, classes

def grep_refs(files: List[Path], token: str) -> List[str]:
    hits = []
    rx = re.compile(re.escape(token))
    for p in files:
        txt = read_text(p)
        if rx.search(txt):
            hits.append(p)
    return hits

def detect_languages(paths: List[str]) -> List[str]:
    langs = set()
    for rel in paths:
        ext = Path(rel).suffix.lower()
        for lang, exts in LANG_EXT_MAP.items():
            if ext in exts:
                langs.add(lang)
    return sorted(langs)

# -------------------- QA BUILDERS --------------------
def q_locate_path(path_rel: str) -> Tuple[str,str]:
    q = f"Where is {path_rel} located in the repository?"
    a = f"It is in [repo:{path_rel}]."
    return q, a

def q_list_dir(dir_rel: str, items: List[str]) -> Tuple[str,str]:
    shown = ", ".join(sorted(items)[:8])
    q = f"What files are in {dir_rel}?"
    a = f"{dir_rel} contains: {shown}."
    return q, a

def q_symbol_def(sym: str, path_rel: str) -> Tuple[str,str]:
    q = f"Where is {sym} defined?"
    a = f"{sym} is defined in [repo:{path_rel}]."
    return q, a

def q_symbol_refs(token: str, paths_rel: List[str]) -> Tuple[str,str]:
    refs = sorted(set(paths_rel))[:8]
    q = f"Which files reference {token}?"
    a = "References found in: " + ", ".join(f"[repo:{p}]" for p in refs) + "."
    return q, a

def q_string_search(s: str, path_rel: str) -> Tuple[str,str]:
    q = f"Which file contains the string '{s}'?"
    a = f"[repo:{path_rel}] contains '{s}'."
    return q, a

def q_repo_fact(q: str, path_rel: str) -> Tuple[str,str]:
    return q, f"See [repo:{path_rel}]."

def q_not_found(path_rel: str) -> Tuple[str,str]:
    q = f"Where is {path_rel} located in the repository?"
    a = "I can’t find that in the repository."
    return q, a

def q_function_summary(name: str, path_rel: str, summary: str) -> Tuple[str,str]:
    q = f"What does the function {name} do?"
    a = f"In short: {summary} (see [repo:{path_rel}])."
    return q, a

def q_class_summary(name: str, path_rel: str, summary: str) -> Tuple[str,str]:
    q = f"What is the purpose of the {name} class?"
    a = f"Summary: {summary} (see [repo:{path_rel}])."
    return q, a

def q_build_target(name: str, path_rel: str) -> Tuple[str,str]:
    q = f"Is there a build target named {name}?"
    a = f"Yes, declared in [repo:{path_rel}]."
    return q, a

def q_dependency_fact(tool: str, path_rel: str) -> Tuple[str,str]:
    q = f"Where can I see {tool} dependencies?"
    a = f"See [repo:{path_rel}]."
    return q, a

def q_language_detect(path_rel_list: List[str], dir_rel: str) -> Tuple[str,str]:
    langs = detect_languages(path_rel_list)
    q = f"What languages are used in {dir_rel}?"
    a = f"{dir_rel} uses: {', '.join(langs) or 'unknown'}."
    return q, a

# -------------------- MAIN GENERATOR --------------------
def limit_per_dir(path_rels: List[str], cap: int) -> List[str]:
    counts = Counter()
    kept = []
    for rel in path_rels:
        d = os.path.dirname(rel)
        if counts[d] < cap:
            kept.append(rel)
            counts[d] += 1
    return kept

def main():
    args = parse_args()
    if args.repo:
        repo_roots = [Path(p).resolve() for p in args.repo]
    else:
        repo_roots = DEFAULT_ROOTS

    # Let user add more excludes
    for name in args.exclude:
        EXCLUDE_DIRS.add(name)

    files = walk_files(repo_roots)
    if not files:
        raise SystemExit("No files found. Use --repo <path> or run from repo root.")

    # Relative normalized paths
    rels_all = [norm_path(repo_rel(p, repo_roots)) for p in files]
    rels_all = limit_per_dir(rels_all, PER_DIR_CAP)

    # dir -> entries
    dir_map = defaultdict(list)
    for r in rels_all:
        d = os.path.dirname(r)
        b = os.path.basename(r)
        dir_map[d].append(b)

    # Pre-scan content
    path_to_text = {}
    for p in files:
        r = norm_path(repo_rel(p, repo_roots))
        if r in rels_all:
            path_to_text[r] = read_text(p)

    # Harvest symbols + summaries
    file_syms: Dict[str, Dict[str,List[str]]] = {}
    py_fn_docs: Dict[str, Dict[str,str]] = {}
    py_cls_docs: Dict[str, Dict[str,str]] = {}
    c_like_summaries: Dict[str, Dict[str,str]] = {}

    for r, txt in path_to_text.items():
        ext = Path(r).suffix.lower()
        if ext in {".h",".hpp",".hh",".hxx",".c",".cc",".cpp",".cxx",".py"}:
            syms = harvest_symbols(txt)
            if syms:
                # small per-file cap to avoid flooding
                for k in syms:
                    syms[k] = syms[k][:20]
                file_syms[r] = syms
        # Python docstrings
        if ext == ".py":
            fdocs, cdocs = extract_py_docstrings(txt)
            if fdocs:
                py_fn_docs[r] = fdocs
            if cdocs:
                py_cls_docs[r] = cdocs
        # C-like summaries for extracted symbols
        if ext in {".h",".hpp",".hh",".hxx",".c",".cc",".cpp",".cxx"} and r in file_syms:
            clike = {}
            all_syms = set(sum(file_syms[r].values(), []))
            for name in list(all_syms)[:12]:
                s = extract_c_like_summary(txt, name)
                if s:
                    clike[name] = s
            if clike:
                c_like_summaries[r] = clike

    # Token candidates for grep (includes includes/imports hints + CMake targets)
    token_refs = defaultdict(set)
    build_targets = []  # (target_name, path_rel)
    for r, txt in path_to_text.items():
        for rx, kind in INCLUDE_REF_HINTS:
            for m in rx.findall(txt):
                if isinstance(m, tuple):
                    m = m[0]
                token = str(m).strip()
                if 2 <= len(token) <= 128:
                    if kind.startswith("cmake_"):
                        # target names are first token before whitespace
                        tgt = token.split()[0]
                        if len(tgt) >= 2 and len(tgt) <= 80:
                            build_targets.append((tgt, r))
                    else:
                        token_refs[token].add(r)

    # Strings for string-search tasks
    string_to_file = {}
    for r, txt in list(path_to_text.items())[:3000]:  # sample to keep it fast
        for m in INTERESTING_TOKEN_RX.findall(txt):
            s = str(m).strip()
            if 3 <= len(s) <= 40 and s not in string_to_file:
                string_to_file[s] = r

    # Repo facts
    license_paths = [r for r in rels_all if Path(r).name.upper().startswith("LICENSE")]
    readmes = [r for r in rels_all if Path(r).name.lower().startswith("readme")]
    cmakelists = [r for r in rels_all if Path(r).name == "CMakeLists.txt"]
    makefiles = [r for r in rels_all if Path(r).name == "Makefile"]

    # Dependency facts
    dep_files = []
    for name in ("requirements.txt","pyproject.toml","environment.yml","Pipfile",
                 "package.json","conanfile.txt","conanfile.py","vcpkg.json"):
        dep_files += [r for r in rels_all if Path(r).name == name]

    # Assemble buckets
    n_total = min(MAX_EXAMPLES, max(2000, int(1.5 * len(rels_all))))
    target = {k: max(1, int(v * n_total)) for k, v in BUCKET_WEIGHTS.items()}

    out: List[Dict] = []
    used_file = Counter()

    # locate_path
    loc_paths = rels_all[:]
    random.shuffle(loc_paths)
    for r in loc_paths:
        if target["locate_path"] <= 0: break
        if used_file[r] >= PER_FILE_CAP: continue
        q, a = q_locate_path(r)
        if add_example(q, a, out):
            used_file[r] += 1
            target["locate_path"] -= 1

    # list_dir
    dirs = [d for d, items in dir_map.items() if d and len(items) >= 2]
    random.shuffle(dirs)
    for d in dirs:
        if target["list_dir"] <= 0: break
        q, a = q_list_dir(d, dir_map[d])
        if add_example(q, a, out):
            target["list_dir"] -= 1

    # symbol_def
    items = list(file_syms.items())
    random.shuffle(items)
    for r, syms in items:
        if target["symbol_def"] <= 0: break
        # choose up to 3 symbols per file
        candidates = []
        for arr in syms.values():
            candidates.extend(arr)
        random.shuffle(candidates)
        for s in candidates[:3]:
            if target["symbol_def"] <= 0: break
            q, a = q_symbol_def(s, r)
            if add_example(q, a, out):
                target["symbol_def"] -= 1

    # symbol_refs (grep by token)
    tok_items = [(t, list(paths)) for t, paths in token_refs.items() if len(paths) >= 2]
    random.shuffle(tok_items)
    for t, paths in tok_items:
        if target["symbol_refs"] <= 0: break
        q, a = q_symbol_refs(t, [norm_path(p) for p in paths])
        if add_example(q, a, out):
            target["symbol_refs"] -= 1

    # string_search
    s_items = list(string_to_file.items())
    random.shuffle(s_items)
    for s, r in s_items:
        if target["string_search"] <= 0: break
        q, a = q_string_search(s, r)
        if add_example(q, a, out):
            target["string_search"] -= 1

    # repo_facts
    if license_paths and target["repo_facts"] > 0:
        q, a = q_repo_fact("What license file does the repository provide?", license_paths[0])
        if add_example(q, a, out):
            target["repo_facts"] -= 1
    if cmakelists and target["repo_facts"] > 0:
        q, a = q_repo_fact("Does the repository use CMake?", cmakelists[0])
        if add_example(q, a, out):
            target["repo_facts"] -= 1
    if makefiles and target["repo_facts"] > 0:
        q, a = q_repo_fact("Is there a Makefile?", makefiles[0])
        if add_example(q, a, out):
            target["repo_facts"] -= 1
    if readmes and target["repo_facts"] > 0:
        q, a = q_repo_fact("Where can I find the README?", readmes[0])
        if add_example(q, a, out):
            target["repo_facts"] -= 1

    # function_summary (Python docstrings + C/C++ comment summaries)
    # Python
    py_fn_items = []
    for r, d in py_fn_docs.items():
        for name, summ in d.items():
            py_fn_items.append((name, r, summ))
    random.shuffle(py_fn_items)
    for name, r, summ in py_fn_items:
        if target["function_summary"] <= 0: break
        q, a = q_function_summary(name, r, summ)
        if add_example(q, a, out):
            target["function_summary"] -= 1
    # C/C++
    clike_items = []
    for r, d in c_like_summaries.items():
        for name, summ in d.items():
            clike_items.append((name, r, summ))
    random.shuffle(clike_items)
    for name, r, summ in clike_items:
        if target["function_summary"] <= 0: break
        q, a = q_function_summary(name, r, summ)
        if add_example(q, a, out):
            target["function_summary"] -= 1

    # class_summary (Python + C/C++)
    py_cls_items = []
    for r, d in py_cls_docs.items():
        for name, summ in d.items():
            py_cls_items.append((name, r, summ))
    random.shuffle(py_cls_items)
    for name, r, summ in py_cls_items:
        if target["class_summary"] <= 0: break
        q, a = q_class_summary(name, r, summ)
        if add_example(q, a, out):
            target["class_summary"] -= 1
    # For C/C++, reuse class/struct names with summaries if we have them
    cc_classes = []
    for r, syms in file_syms.items():
        names = syms.get("cpp_class", [])
        for n in names[:6]:
            s = c_like_summaries.get(r, {}).get(n)
            if s:
                cc_classes.append((n, r, s))
    random.shuffle(cc_classes)
    for name, r, summ in cc_classes:
        if target["class_summary"] <= 0: break
        q, a = q_class_summary(name, r, summ)
        if add_example(q, a, out):
            target["class_summary"] -= 1

    # build_targets
    random.shuffle(build_targets)
    seen_tgts = set()
    for tgt, r in build_targets:
        if target["build_targets"] <= 0: break
        if tgt in seen_tgts:
            continue
        seen_tgts.add(tgt)
        q, a = q_build_target(tgt, r)
        if add_example(q, a, out):
            target["build_targets"] -= 1

    # dependency_facts
    random.shuffle(dep_files)
    for r in dep_files:
        if target["dependency_facts"] <= 0: break
        q, a = q_dependency_fact(Path(r).name, r)
        if add_example(q, a, out):
            target["dependency_facts"] -= 1

    # language_detect (per directory)
    lang_dirs = [d for d in dir_map if d and len(dir_map[d]) >= 2]
    random.shuffle(lang_dirs)
    for d in lang_dirs:
        if target["language_detect"] <= 0: break
        # build file list for dir
        files_in_dir = [norm_path(os.path.join(d, b)) for b in dir_map[d]]
        q, a = q_language_detect(files_in_dir, d)
        if add_example(q, a, out):
            target["language_detect"] -= 1

    # not_found (negatives)
    fake_paths = [f"nonexistent/dir/{i}/ghost_file_{i}.hpp" for i in range(1000)]
    random.shuffle(fake_paths)
    for fp in fake_paths:
        if target["not_found"] <= 0: break
        q, a = q_not_found(fp)
        if add_example(q, a, out):
            target["not_found"] -= 1

    # -------------------- WRITE --------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # -------------------- STATS --------------------
    print(f"Wrote {len(out)} examples to {OUT_PATH}")
    buckets = Counter()
    for ex in out:
        u = ex["messages"][0]["content"].lower()
        a = ex["messages"][1]["content"].lower()
        if u.startswith("where is ") and "[repo:" in a and "can’t find" not in a:
            buckets["locate_path"] += 1
        elif u.startswith("what files are in"):
            buckets["list_dir"] += 1
        elif u.startswith("where is ") and " defined?" in u:
            buckets["symbol_def"] += 1
        elif u.startswith("which files reference"):
            buckets["symbol_refs"] += 1
        elif u.startswith("which file contains the string"):
            buckets["string_search"] += 1
        elif "license" in u or "readme" in u or "cmake" in u or "makefile" in u:
            buckets["repo_facts"] += 1
        elif u.startswith("what does the function"):
            buckets["function_summary"] += 1
        elif u.startswith("what is the purpose of the"):
            buckets["class_summary"] += 1
        elif u.startswith("is there a build target named"):
            buckets["build_targets"] += 1
        elif u.startswith("where can i see") and "dependencies" in u:
            buckets["dependency_facts"] += 1
        elif u.startswith("what languages are used in"):
            buckets["language_detect"] += 1
        elif "can’t find" in a:
            buckets["not_found"] += 1
        else:
            buckets["other"] += 1
    print("Bucket counts:", dict(buckets))


if __name__ == "__main__":
    main()
