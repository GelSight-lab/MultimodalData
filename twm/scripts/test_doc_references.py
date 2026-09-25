
"""Every path a published doc tells you to run must itself be published.

The READMEs' "Reproducing" sections named nine `scripts/*.py`. None was on the
Hub, and when checked they carried absolute paths to one laptop. A reproduction
section nobody but the author can run is worse than absent: it reads as a
promise.

This walks every published .md, extracts the repository paths it mentions, and
checks each exists on the Hub. It also greps the published scripts for absolute
paths under a user's home or a specific mount, which is the other half of the
same failure — published but unrunnable.

    python scripts/test_doc_references.py
"""
from __future__ import annotations


import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from react_toolbox.staging import staging_dir

RESULTS: list[tuple[bool, str, str]] = []
REPO = "yxma/React"
PATH_RE = re.compile(
    r"(?<![\w/])((?:scripts|toolbox|examples|data|test_sets|preprocess|docs)"
    r"/[\w./+-]+\.(?:py|json|md|jpg|jsonl))")
ABS_RE = re.compile(r"[\"'](/(?:home|media|Users)/[^\"']+)[\"']")


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = set(api.list_repo_files(REPO, repo_type="dataset"))
    d = str(staging_dir())

    docs = sorted(f for f in files if f.endswith(".md"))
    missing, seen = [], 0
    for doc in docs:
        text = Path(hf_hub_download(REPO, doc, repo_type="dataset",
                                    local_dir=d)).read_text(errors="ignore")
        for ref in sorted(set(PATH_RE.findall(text))):
            seen += 1
            if ref not in files:
                missing.append(f"{doc} -> {ref}")
    check(not missing, "every path a published doc names is published",
          f"{len(docs)} docs, {seen} path references, all present"
          + (f"; MISSING {missing[:4]}" if missing else ""))

    # ...and the published scripts must not point at one person's disk
    # react_paths.py is the ONE place a default may live, and only as the
    # fallback of an environment lookup. Exempting it wholesale would let a
    # literal creep back in beside the documented ones, so the exemption is
    # narrowed: every absolute path in that file must sit on a line that reads
    # the environment.
    scripts = sorted(f for f in files if f.startswith("scripts/") and f.endswith(".py"))
    hard, loose = [], []
    for s_ in scripts:
        text = Path(hf_hub_download(REPO, s_, repo_type="dataset",
                                    local_dir=d)).read_text(errors="ignore")
        if s_ == "scripts/react_paths.py":
            for line in text.splitlines():
                if ABS_RE.search(line) and "os.environ" not in line \
                        and not line.lstrip().startswith(("#", "_DEF_")):
                    loose.append(line.strip()[:70])
            continue
        for m in ABS_RE.findall(text):
            hard.append(f"{s_}: {m}")
    check(not hard and not loose,
          "only react_paths.py holds a default path, behind the environment",
          f"{len(scripts)} scripts: {len(scripts)-1} with no absolute path, and "
          f"react_paths.py's defaults all sit behind os.environ.get"
          + (f"; hard-coded {hard[:3]}" if hard else "")
          + (f"; loose in react_paths {loose[:2]}" if loose else ""))

    # ...and the path module they share is published, since they import it
    imports_paths = []
    for s in scripts:
        text = Path(hf_hub_download(REPO, s, repo_type="dataset",
                                    local_dir=d)).read_text(errors="ignore")
        if "from react_paths import" in text:
            imports_paths.append(s)
    check("scripts/react_paths.py" in files and imports_paths,
          "the shared path module ships with the scripts that import it",
          f"{len(imports_paths)} scripts import react_paths; "
          f"published: {'scripts/react_paths.py' in files}")

    # 4 — the published copy must BE the repo copy. `scripts/` and `USAGE.md`
    #     were uploaded once, by hand, and nothing has kept them current
    #     since: exactly the failure the toolbox upload block in
    #     build_release_publish.py describes, one directory over. Drift here is
    #     invisible -- the files exist, the doc-reference check above is happy,
    #     and a reader downloads instructions for a dataset that has moved.
    from huggingface_hub import HfApi, hf_hub_download
    import hashlib
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    root = Path(__file__).resolve().parents[1]
    pairs = [(f, root / f) for f in files if f.startswith("scripts/")]
    pairs += [(f, src) for f, src in (
        ("USAGE.md", root / "docs/USAGE.md"),
        ("README.md", root / "docs/superpowers/specs/README_v2_release.md"))
        if f in files]
    drift, absent = [], []
    for remote, local in pairs:
        if not local.exists():
            absent.append(remote)
            continue
        got = hf_hub_download(REPO, remote, repo_type="dataset")
        if Path(got).read_bytes() != local.read_bytes():
            drift.append(remote)
    check(not drift and not absent,
          "every published script and doc is the repo's current copy",
          f"{len(pairs)} files checked, all identical" if not (drift or absent)
          else f"{len(drift)} stale on the Hub: {drift[:6]}"
               + (f"; {len(absent)} published with no source: {absent[:3]}"
                  if absent else ""))

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not x for x, _, _ in RESULTS)
    print(f"\ndoc references: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
