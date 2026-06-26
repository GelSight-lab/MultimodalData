"""Re-encode every shipped GIF as a 5×-speed MP4 (sharper text via yuv444p,
8-bit, CRF 20), then drop every GIF from HF in the same commit. Local GIFs
are also removed after a verified push.
"""
import os
import subprocess
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

FIG_ROOT = Path("/media/yxma/Disk1/twm/figures")
SEARCH_DIRS = [
    FIG_ROOT / "episode_previews",
    FIG_ROOT / "dataloader_examples",
    FIG_ROOT / "dataset_figures" / "freeze_diagnose" / "ot_loss",
]


def reencode(gif: Path, mp4: Path) -> bool:
    """5x speed, yuv444p, 8-bit, CRF 20. Returns True on success."""
    mp4.unlink(missing_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(gif),
        # setpts/5 → 5× speed (no frame drop, time compression)
        # pad to even dims for h264; yuv444p preserves chroma for crisp text
        "-vf", "setpts=PTS/5,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv444p",
        "-c:v", "libx264",
        "-profile:v", "high444",
        "-preset", "medium",
        "-crf", "20",
        "-movflags", "+faststart",
        "-an",
        str(mp4),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL {gif.name}: {r.stderr[:200]}")
        return False
    return True


def main():
    api = HfApi()
    remote = api.list_repo_files("yxma/React", repo_type="dataset")
    gifs_local = []
    for root in SEARCH_DIRS:
        if root.is_dir():
            gifs_local.extend(sorted(root.rglob("*.gif")))
    print(f"Found {len(gifs_local)} GIFs locally")

    ops = []
    n_ok = 0
    total_in = 0
    total_out = 0
    for gif in gifs_local:
        mp4 = gif.with_suffix(".mp4")
        if not reencode(gif, mp4):
            continue
        n_ok += 1
        total_in += gif.stat().st_size
        total_out += mp4.stat().st_size
        rel = mp4.relative_to(Path("/media/yxma/Disk1/twm"))
        ops.append(CommitOperationAdd(path_in_repo=str(rel), path_or_fileobj=str(mp4)))

    # Delete every .gif on HF (under figures/)
    gifs_remote = [f for f in remote if f.startswith("figures/") and f.endswith(".gif")]
    for f in gifs_remote:
        ops.append(CommitOperationDelete(path_in_repo=f))

    print(f"  re-encoded {n_ok}/{len(gifs_local)} MP4s")
    print(f"  GIF total: {total_in/1024/1024:.0f} MB  →  MP4 total: {total_out/1024/1024:.0f} MB")
    print(f"  HF GIF deletions queued: {len(gifs_remote)}")
    print(f"Total ops: {len(ops)}")

    api.create_commit(
        repo_id="yxma/React", repo_type="dataset", operations=ops,
        commit_message=(
            "MP4s re-encoded at 5× speed (yuv444p / 8-bit / CRF 20 — sharper "
            f"text overlays); all {len(gifs_remote)} GIFs removed from HF "
            f"(MP4-only going forward). "
            f"Source 8-bit RGB → 10-bit would be wasted bits; 4:4:4 chroma "
            f"is the upgrade that actually matters for the text panels."
        ),
    )
    print("Pushed.")

    # Delete local GIFs after successful push
    print("\nDeleting local GIFs:")
    for gif in gifs_local:
        gif.unlink()
    print(f"  removed {len(gifs_local)} local GIFs")


if __name__ == "__main__":
    main()
