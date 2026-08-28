
"""Download the PUBLISHED toolbox and run the README's own example with it.

Every other toolbox check imports the working copy on this disk. That is not
what a user gets. Measured: the published `calibration.py` was missing
`project_gel_frame` and `project_rigid_origin`, so the published `probe_eval`
and `viz` could not import, and the first usage example in the test set README
raised ImportError for anyone who downloaded it. Local tests were all green.

This installs the toolbox from the Hub into a scratch directory, fetches a
probe package from the Hub, and runs the documented workflow against both.

    python scripts/test_published_toolbox.py
"""
from __future__ import annotations

from react_toolbox.staging import staging_dir

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def fetch(root: Path):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    pkg = root / "react_toolbox"
    pkg.mkdir(parents=True, exist_ok=True)
    files = [f for f in api.list_repo_files("yxma/React", repo_type="dataset")
             if f.startswith("toolbox/")]
    for f in files:
        p = hf_hub_download("yxma/React", f, repo_type="dataset",
                            local_dir=str(staging_dir()))
        shutil.copy(p, pkg / Path(f).name)
    data = root / "data"
    want = ["test_sets/probes_v1/manifest.json",
            "test_sets/probes_v1/probes/run0/meta.json",
            "test_sets/probes_v1/probes/run0/trans+x.npz",
            "test_sets/probes_v1/probes/run0/rot+y.npz",
            "test_sets/probes_v1/probes/run0/context/ctx3_view_middle.jpg"]
    want += [f"test_sets/probes_v1/calibration/T_mocap_to_cam_{v}.json"
             for v in ("left", "middle", "right")]
    want += [f"test_sets/probes_v1/calibration/T_gel_to_rigid_{s}.json"
             for s in ("left", "right")]
    for f in want:
        hf_hub_download("yxma/React", f, repo_type="dataset", local_dir=str(data))
    return len(files), data / "test_sets/probes_v1"


# Run in a SUBPROCESS with only the downloaded toolbox importable, so this
# repo's working copy cannot satisfy an import the published one is missing.
SCRIPT = textwrap.dedent('''
    import json, sys, numpy as np, cv2
    root, pkg = sys.argv[1], sys.argv[2]
    sys.path.insert(0, root)
    from react_toolbox.calibration import load_calibration, project_gel_frame
    from react_toolbox.probe_eval import overlay_gt, rollout_error, project_gt
    from react_toolbox.synth_actions import make_translation_set, make_rotation_set
    from react_toolbox.splits import build_splits, assert_window_fits
    from react_toolbox.actions import delta_pose_action, integrate_delta

    cal = load_calibration(pkg)
    meta = json.load(open(pkg + "/probes/run0/meta.json"))
    d = np.load(pkg + "/probes/run0/trans+x.npz")
    gel = cal["gel_" + meta["moving_side"]]
    cam = cal["cams"]["middle"]

    # the README's example
    img = cv2.imread(pkg + "/probes/run0/context/ctx3_view_middle.jpg")[:, :, ::-1]
    vis = overlay_gt(img, d["poses"], gel, cam, held_pose7=d["held_pose"],
                     held_gel_mm=cal["gel_" + meta["held_side"]])
    err = rollout_error(d["poses"], d["poses"], gel, cam)

    # the stored ground truth must recompute from the PUBLISHED code
    got = project_gt(d["poses"], gel, cam)
    worst = float(np.nanmax(np.linalg.norm(got - d["gt_px_middle"], axis=1)))

    # the generator and the split module must work too
    tr = make_translation_set(d["poses"][0], seed=0)
    ro = make_rotation_set(d["poses"][0], gel, seed=0)

    print(json.dumps({
        "overlay_changed_px": int((np.abs(vis.astype(int) - img.astype(int)).sum(2) > 25).sum()),
        "err_zero": float(err["pos_mm_final"]),
        "gt_recompute_px": worst,
        "n_trans": len(tr), "n_rot": len(ro),
        "axes": sorted({t["axis"] for t in tr}),
    }))
''')


def main() -> int:
    root = staging_dir()
    n_files, pkg = fetch(root)
    check(n_files >= 15, "the published toolbox downloads",
          f"{n_files} files under toolbox/")

    p = subprocess.run([sys.executable, "-c", SCRIPT, str(root), str(pkg)],
                       capture_output=True, text=True, timeout=600)
    ok = p.returncode == 0
    out = {}
    if ok:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    check(ok, "the README's example runs against the PUBLISHED code",
          f"overlay changed {out.get('overlay_changed_px')} px, "
          f"rollout_error on truth {out.get('err_zero')} mm"
          if ok else (p.stderr.strip().splitlines() or ["?"])[-1][:150])

    if ok:
        check(out["gt_recompute_px"] < 1e-6,
              "published code reproduces the published ground truth",
              f"worst disagreement {out['gt_recompute_px']:.2e} px")
        check(out["n_trans"] == 6 and out["n_rot"] == 6 and len(out["axes"]) == 6,
              "the published generator still makes 12 single-axis probes",
              f"{out['n_trans']} translations, {out['n_rot']} rotations, "
              f"axes {out['axes']}")

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for okk, name, ev in RESULTS:
        print(f"  [{'ok' if okk else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not x for x, _, _ in RESULTS)
    print(f"\npublished toolbox: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
