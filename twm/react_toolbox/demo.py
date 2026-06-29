"""End-to-end demo: download one episode, run every toolbox utility, and save
a montage (raw | diff heatmap | contact overlay | depth) plus a quick console
report. Doubles as the end-to-end verification.

    python -m react_toolbox.demo                 # default motherboard episode
    python -m react_toolbox.demo --task pushT --episode 2026-06-18/episode_000
"""
from __future__ import annotations

import argparse

import numpy as np

import react_toolbox as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="yxma/React")
    ap.add_argument("--task", default="motherboard")
    ap.add_argument("--episode", default="2026-05-10/episode_000")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="toolbox_demo.png")
    ap.add_argument("--with_depth", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    date, ep = args.episode.split("/")
    vid = hf_hub_download(args.repo, f"data/{args.task}/videos/{date}/{ep}/tactile_{args.side}.mp4", repo_type="dataset")
    pqf = hf_hub_download(args.repo, f"data/{args.task}/meta/{date}/{ep}.parquet", repo_type="dataset")

    meta = T.load_meta(pqf)
    inten = meta[f"tactile_{args.side}_intensity"][:args.n]
    frames = T.load_video(vid, range(args.n))
    ref = T.get_reference(frames, mode="p01", intensity=inten)
    i = int(np.argmax(inten))           # peak-contact frame
    frame = frames[i]

    # signals
    metrics = T.contact_metrics(frame, ref)
    mask = T.contact_mask(frame, ref)
    centroid = T.contact_centroid(mask)
    print(f"[demo] {args.task} {args.episode} side={args.side} frame={i}")
    print(f"  contact metrics: {metrics}")
    print(f"  contact mask area={mask.mean()*100:.1f}%  centroid={centroid}")

    panels = [frame,
              T.diff_heatmap(frame, ref),
              T.contact_overlay(frame, ref),
              T.reference_compare(frame, ref)[:, :640]]   # ref panel for size match
    labels = ["raw tactile", "diff heatmap", "contact overlay", "reference"]

    if args.with_depth:
        from react_toolbox import depth, viz
        h = depth.height_map(frame, ref)
        panels.append(viz.depth_view(h)); labels.append("depth (approx)")
        print(f"  depth: range=[{h.min():.2f},{h.max():.2f}] in-contact={h[mask].mean():.2f} out={h[~mask].mean():.2f}")

    # actions
    pose = meta[f"sensor_{args.side}_pose"][:args.n]
    act = T.next_state_action(pose); dlt = T.delta_pose_action(pose)
    print(f"  actions: next_state {act.shape}, delta {dlt.shape}")

    try:
        import cv2
        montage = np.concatenate([cv2.resize(p, (320, 240)) for p in panels], axis=1)
        cv2.imwrite(args.out, montage[..., ::-1])   # RGB->BGR for cv2
        print(f"  saved montage -> {args.out}  ({', '.join(labels)})")
    except Exception as e:
        print(f"  (montage skipped: {e})")
    print("[demo] OK")


if __name__ == "__main__":
    main()
