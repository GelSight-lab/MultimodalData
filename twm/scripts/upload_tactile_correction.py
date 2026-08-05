"""Upload the tactile-latency-corrected data, overwriting both HF repos.

React (yxma/React):
  - tactile_{left,right}.mp4 per episode (shifted +15, single-gen from H5)
  - meta/*.parquet per episode (tactile scalar columns shifted)
  - per-task segments.json + bad_frames.json (intensity_spikes follow tactile)
  - tasks.json + README: mark tactile latency CORRECTED in-data
React-lerobot (yxma/React-lerobot):
  - full re-upload (tactile videos + parquet + meta carry the correction)
"""
from __future__ import annotations
import os, json
from pathlib import Path
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

REL = Path("/media/yxma/Disk1/twm/release")
LR = Path("/media/yxma/Disk1/twm/lerobot")
api = HfApi()
SHIFT = 15


def upload_react():
    for task in ("motherboard", "pushT"):
        ops = []
        base = REL / task
        # tactile videos only (RGB/depth unchanged)
        for mp4 in base.rglob("videos/**/tactile_*.mp4"):
            rel = mp4.relative_to(base)
            ops.append(CommitOperationAdd(f"data/{task}/{rel}", str(mp4)))
        # parquet (tactile scalar cols changed)
        for pq in base.rglob("meta/**/*.parquet"):
            rel = pq.relative_to(base)
            ops.append(CommitOperationAdd(f"data/{task}/{rel}", str(pq)))
        # curation
        for j in ("segments.json", "bad_frames.json", "episodes.jsonl"):
            p = base / j
            if p.exists():
                ops.append(CommitOperationAdd(f"data/{task}/{j}", str(p)))
        print(f"[react] {task}: {len(ops)} ops", flush=True)
        api.create_commit(repo_id="yxma/React", repo_type="dataset", operations=ops,
            commit_message=(f"Correct tactile latency (+{SHIFT}f) for {task}: tactile videos "
                            f"shifted to align with cameras/poses, rebuilt from raw H5 "
                            f"(single-gen, raw-quality contact scalars). RGB/depth/pose unchanged."))


def update_react_docs():
    # tasks.json: flip tactile_latency to corrected-in-data
    tj = json.loads(Path(hf_hub_download("yxma/React", "tasks.json", repo_type="dataset")).read_text())
    tl = tj.get("tactile_latency", {})
    tl.update({"status": "CORRECTED_IN_DATA", "frames_shifted": SHIFT,
               "method": "tactile streams shifted +15 frames (rebuilt from raw H5) so tactile[i] aligns with view[i]/pose[i]",
               "note": "Recordings up to 2026-06-18 had a ~15-frame GelSight buffer lag; it has been baked out of the published tactile videos + contact scalars. No loader compensation needed. Rig fixed 2026-06-27."})
    tj["tactile_latency"] = tl
    p = "/tmp/tasks_corrected.json"; Path(p).write_text(json.dumps(tj, indent=2))
    ops = [CommitOperationAdd("tasks.json", p)]
    # README: replace the "known issue" section header note if present
    rp = hf_hub_download("yxma/React", "README.md", repo_type="dataset")
    rd = Path(rp).read_text()
    rd = rd.replace("## ⚠️ Known issue: tactile acquisition latency (~15 frames)",
                    "## ✅ Tactile latency corrected (was ~15 frames)")
    rd = rd.replace("have a GelSight-vs-camera capture\nlag of **≈15 frames",
                    "HAD a GelSight-vs-camera capture\nlag of **≈15 frames")
    rd = rd.replace("**correctable**. The reference loader compensates at load time:",
                    "**now corrected in the published data** (tactile shifted +15f, rebuilt from raw H5). No loader flag needed. The loader still accepts `tactile_latency=` for raw data:")
    p2 = "/tmp/README_corrected.md"; Path(p2).write_text(rd)
    ops.append(CommitOperationAdd("README.md", p2))
    api.create_commit(repo_id="yxma/React", repo_type="dataset", operations=ops,
        commit_message="Docs: mark tactile latency CORRECTED in-data (no loader compensation needed)")
    print("[react] docs updated", flush=True)


def upload_lerobot():
    print("[lerobot] full re-upload ...", flush=True)
    api.upload_folder(repo_id="yxma/React-lerobot", repo_type="dataset",
        folder_path=str(LR),
        commit_message=f"Tactile latency corrected (+{SHIFT}f): tactile videos + scalars aligned to cameras/poses")
    print("[lerobot] done", flush=True)


if __name__ == "__main__":
    upload_react()
    update_react_docs()
    upload_lerobot()
    print("[upload] ALL DONE", flush=True)
