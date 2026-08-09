"""Push the results site to a Hugging Face Space (static SDK)."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

SITE = Path("/media/yxma/Disk1/twm/force_recovery/site")
SPACE_ID = "yxma/react-force-recovery"


def check_fresh_and_labelled() -> list[str]:
    """The site must be REGENERATED, not just re-uploaded, and correctly named.

    Both failures happened on the same deploy. The dataset behind the LUT was
    renamed `cnc_mini_26` in the generators, which broke `site._force_row`'s
    prefix lookup — so a regeneration would have crashed. Nothing regenerated,
    because `publish` only uploads a folder; the stale HTML shipped, and the
    Space went on claiming "validated on GlowTact rho=0.986" for a row that is
    GelSight Mini data. The rename was the very thing being fixed.

    So: refuse if a page is older than the generator that writes it, and
    refuse if a page still labels cnc_mini_26 as GlowTact. The citation of the
    GlowTact *release* is legitimate and is allowed by requiring the dataset
    handle beside it.
    """
    import re
    src = Path(__file__).parent
    gens = {"index.html": "site.py", "results.html": "results_page.py",
            "results_zh.html": "results_page.py",
            "method.html": "method_page.py", "method_zh.html": "method_page.py",
            "reconstruction.html": "calibfree_page.py"}
    bad = []
    for page, gen in gens.items():
        pg, gn = SITE / page, src / gen
        if not pg.exists():
            bad.append(f"{page}: missing — regenerate before publishing")
            continue
        if gn.exists() and pg.stat().st_mtime < gn.stat().st_mtime:
            bad.append(f"{page}: older than {gen}; `publish` only uploads, it "
                       f"does not build — regenerate it")
        txt = pg.read_text()
        for m in re.finditer(r"GlowTact", txt):
            around = txt[max(0, m.start() - 60):m.start() + 60]
            if "GlowTact_Datasets" in around or "dacongming666" in around:
                continue                      # citing the source release: fine
            bad.append(f"{page}: calls cnc_mini_26 'GlowTact' — that dataset is "
                       f"the GelSight Mini arm of the GlowTact release")
            break
    return bad


def publish() -> str:
    problems = check_fresh_and_labelled()
    if problems:
        raise SystemExit("refusing to publish:\n  " + "\n  ".join(problems))
    api = HfApi()
    api.create_repo(SPACE_ID, repo_type="space", space_sdk="static",
                    exist_ok=True)
    api.upload_folder(folder_path=str(SITE), repo_id=SPACE_ID,
                      repo_type="space",
                      commit_message="force recovery: methods, evaluation, debug log")
    return f"https://huggingface.co/spaces/{SPACE_ID}"


if __name__ == "__main__":
    print(publish())
