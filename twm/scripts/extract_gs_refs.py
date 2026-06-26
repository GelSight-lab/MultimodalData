"""For each episode .pt, pull the p01 reference *index* from
`_contact_meta.ref_p01_idx_{left,right}`, shift back to H5 coords by adding
`trim_offset`, read the native 640×480 GelSight frame from the H5, save as
PNG next to the .pt. Push to HF.

After this lands, render_clip's diff thumbnails will be against the
quietest frame of the episode rather than the (possibly-already-contacted)
first frame of whatever clip is being rendered.
"""
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/yxma/MultimodalData")

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from PIL import Image

PT_ROOT = Path("/media/yxma/Disk1/twm/processed/mode1_v1/motherboard")
H5_ROOT = Path("/media/yxma/Disk1/twm/data/motherboard")

def main():
    out_files = []
    skipped = []
    for pt in sorted(PT_ROOT.rglob("episode_*.pt")):
        date = pt.parent.name
        if date == "2026-03-23":
            continue
        ep_stem = pt.stem
        ep_key = f"{date}/{ep_stem}"
        h5_path = H5_ROOT / date / f"{ep_stem}.h5"
        if not h5_path.exists():
            skipped.append(ep_key + " (no H5)")
            continue
        print(f"loading {ep_key}...")
        d = torch.load(pt, weights_only=False, map_location="cpu")
        meta = d.get("_contact_meta", {})
        trim_off = int(meta.get("trim_offset", 0))
        # `ref_p01_idx_*` are in trimmed-pt frame coordinates (the .pt's own
        # arrays), so add trim_off to reindex into the H5.
        ref_L_idx = int(meta.get("ref_p01_idx_left",  0)) + trim_off
        ref_R_idx = int(meta.get("ref_p01_idx_right", 0)) + trim_off
        del d
        gc.collect()

        with h5py.File(h5_path, "r") as h5:
            n_gs_L = h5["gelsight/left/frames"].shape[0]
            n_gs_R = h5["gelsight/right/frames"].shape[0]
            ref_L_idx = min(ref_L_idx, n_gs_L - 1)
            ref_R_idx = min(ref_R_idx, n_gs_R - 1)
            # GelSight in H5 is RGB (post-BGR2RGB in the recorder)
            gs_L = h5["gelsight/left/frames"][ref_L_idx]
            gs_R = h5["gelsight/right/frames"][ref_R_idx]

        out_L = pt.parent / f"{ep_stem}.gs_ref_left.png"
        out_R = pt.parent / f"{ep_stem}.gs_ref_right.png"
        Image.fromarray(gs_L).save(out_L)
        Image.fromarray(gs_R).save(out_R)
        out_files.append((out_L, ref_L_idx, "left"))
        out_files.append((out_R, ref_R_idx, "right"))
        print(f"  L p01_idx_h5={ref_L_idx:5d}  R p01_idx_h5={ref_R_idx:5d}  "
              f"sizes: L={out_L.stat().st_size//1024}KB R={out_R.stat().st_size//1024}KB")

    if skipped:
        print(f"\nSkipped {len(skipped)}: {skipped}")
    print(f"\nWrote {len(out_files)} reference PNGs.")

    # Push to HF
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    ops = []
    for p, _, _ in out_files:
        rel = p.relative_to(Path("/media/yxma/Disk1/twm"))
        ops.append(CommitOperationAdd(path_in_repo=str(rel), path_or_fileobj=str(p)))
    api.create_commit(
        repo_id="yxma/React", repo_type="dataset", operations=ops,
        commit_message=(
            f"Add per-episode GelSight reference images: episode_NNN.gs_ref_{{left,right}}.png "
            f"= the 1st-percentile-intensity frame at native 640×480 (the 'quietest' / "
            f"least-contacted moment of the episode). Use these as the diff baseline "
            f"in any visualization — gives a stable 'this is what's currently pressing "
            f"on the gel' diff rather than 'change since the start of this clip'. "
            f"{len(out_files)//2} episodes × 2 sides = {len(out_files)} files."
        ),
    )
    print("Pushed.")


if __name__ == "__main__":
    main()
