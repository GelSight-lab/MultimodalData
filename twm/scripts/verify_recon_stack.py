"""Every reconstruction improvement, checked ON THE PRODUCTION PATH.

Each fix in this stack was measured, gated by its own test, and committed. That
is not the same as being LIVE: a test can pass against a helper while the
shipping entry point quietly takes another route. This asks the production
functions themselves — `debug_gallery.stages`, `calib_free.reconstruct`,
`react_calib.force_stages`, `eval_panel.mesh` — and fails if any improvement is
not actually in the path a user's frame travels.

Ten checks, one line of evidence each:

  1  boundary condition is data-driven          poisson.integrate
  2  marker gel keeps the clamped solver        poisson.free_boundary_ok
  3  calibration-free normalises by the ref     calib_free.gradients
  4  LUT depth cannot exceed the gel            debug_gallery.stages
  5  the ceiling does not bind on good data     ditto
  6  truncated frames are flagged               calib_free.reconstruct
  7  force comes from the declared recon        react_calib.force_stages
  8  force refuses the wrong reconstruction     react_calib.fit().predict
  9  mesh shading scales on the imprint         o3d_view.gradient_shade
 10  mesh camera does not crop the pad          o3d_view.render_mesh

    xvfb-run -a -s "-screen 0 1400x1000x24" python -m scripts.verify_recon_stack

Checks 9-10 need a display; without one they are REPORTED AS UNVERIFIED rather
than skipped quietly.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Run me directly: `python scripts/verify_recon_stack.py`. Without this the
# `force_recovery` import fails and the run reads as a broken checker rather
# than as ten checks that never ran.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    from force_recovery import calib_free as CF
    from force_recovery import debug_gallery as dg
    from force_recovery import react_calib as RC
    from force_recovery.debug_gallery import stages
    from force_recovery.lut_calibration import GEL_THICKNESS_MM
    from force_recovery.o3d_view import gradient_shade, has_display
    from force_recovery.poisson import free_boundary_ok
    from force_recovery.visible_eval import in_fov, visible

    # `debug_gallery.RNG` is module-level and every loader draws from it, so
    # two calls in one process return DIFFERENT frames. Found by sabotaging a
    # production setting and watching two unrelated checks flip: the second
    # run had simply sampled other presses. A verifier whose verdict depends
    # on call order verifies nothing.
    dg.RNG = np.random.default_rng(0)
    rows, get = dg.load_glowtact()
    rng = np.random.default_rng(0)
    sel = [rows[i] for i in rng.permutation(len(rows))[:25]]
    frames = [(*get(fr), fr) for fr in sel]

    # 1 / 6 — the solver actually chosen, and the truncation flag
    solvers, trunc = set(), 0
    for img, ref, fr in frames:
        r = CF.reconstruct(img, ref)
        solvers.add(r["solver"])
        trunc += bool(r["truncated"])
    check(solvers == {"neumann-detrended"},
          "boundary condition is data-driven",
          f"markerless frames all solved as {sorted(solvers)}")
    check(0 < trunc < len(frames), "truncated frames are flagged",
          f"{trunc}/{len(frames)} flagged truncated")

    # 2 — marker gel must NOT get the free boundary
    try:
        dg.RNG = np.random.default_rng(0)
        frows, fget = dg.load_feats()
        _, fref = fget(frows[0])
        check(not free_boundary_ok(fref), "marker gel keeps the clamped solver",
              "FEATS reference -> free_boundary_ok False")
    except Exception as exc:                                   # noqa: BLE001
        check(False, "marker gel keeps the clamped solver",
              f"UNVERIFIED: FEATS unavailable ({exc})")

    # 3 — reference normalisation, measured as illumination invariance
    img, ref, _ = frames[0]
    peaks = []
    for g in (0.6, 1.0, 1.4):
        h, w = img.shape[:2]
        _, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        vg = (g * (0.7 + 0.3 * xx / w))[..., None]
        peaks.append(float(np.percentile(
            CF.reconstruct(img * vg, ref * vg)["depth"], 99.8)))
    spread = max(peaks) / max(min(peaks), 1e-9)
    check(spread < 1.15, "calibration-free normalises by the reference",
          f"peak spread x{spread:.3f} over a 2.3x illumination change")

    # 4 / 5 — the physical ceiling, and that it spares good data
    peak_all, peak_whole = [], []
    for img, ref, fr in frames:
        pk = float(stages(img, ref)["depth"].max())
        peak_all.append(pk)
        if in_fov(fr) and visible(img, ref):
            peak_whole.append(pk)
    check(max(peak_all) <= GEL_THICKNESS_MM + 1e-9,
          "LUT depth cannot exceed the gel",
          f"max {max(peak_all):.3f} mm, ceiling {GEL_THICKNESS_MM} mm")
    check(bool(peak_whole) and max(peak_whole) < GEL_THICKNESS_MM - 0.5,
          "the ceiling does not bind on fully imaged contacts",
          f"{len(peak_whole)} such frames, max {max(peak_whole or [0]):.3f} mm")

    # 7 / 8 — the force channel's reconstruction, and its refusal
    st = RC.force_stages(img, ref)
    check(st.get("recon") == RC.FORCE_RECONSTRUCTION == "calibfree",
          "force comes from the declared reconstruction",
          f"force_stages -> {st.get('recon')!r}, "
          f"FORCE_RECONSTRUCTION={RC.FORCE_RECONSTRUCTION!r}")
    predict = RC.fit(report=False)
    try:
        predict(stages(img, ref))
        check(False, "force refuses the wrong reconstruction",
              "a plain stages() dict was ACCEPTED")
    except TypeError as exc:
        check(True, "force refuses the wrong reconstruction",
              f"TypeError: {str(exc)[:60]}...")

    # 9 — shading scaled on the imprint, not the flat gel
    z = np.zeros((240, 320), np.float32)
    z[8:64, 150:206] = 1.4
    import cv2
    z = cv2.GaussianBlur(z, (9, 9), 3)
    on = z > 0.05 * z.max()
    sh = gradient_shade(z)[on]
    sat = float(((np.abs(sh - sh.min()) < 1e-9)
                 | (np.abs(sh - sh.max()) < 1e-9)).mean())
    check(sat < 0.10, "mesh shading scales on the imprint",
          f"{sat*100:.1f}% of a 5%-of-frame imprint saturated")

    # 10 — the camera frames the pad without cropping it
    if not has_display():
        check(False, "mesh camera does not crop the pad",
              "UNVERIFIED: no DISPLAY, run under xvfb-run")
    else:
        from force_recovery.o3d_view import (MESH_KW, MM_PER_PIXEL,
                                             content_box, render_depth_mesh)
        rgb = render_depth_mesh(np.zeros((240, 320), np.float32), MM_PER_PIXEL,
                                stride=2, bg=1.0, width=432, height=324,
                                **MESH_KW)
        y0, y1, x0, x1, border = content_box(rgb, pad=0)
        check(border == 0.0 and (x1 - x0) / rgb.shape[1] > 0.8,
              "mesh camera does not crop the pad",
              f"flat pad fills {(x1-x0)/rgb.shape[1]:.3f} of the width, "
              f"border occupancy {border:.3f}")

    width = max(len(n) for _, n, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{width}}  {ev}")
    bad = sum(not ok for ok, _, _ in RESULTS)
    print(f"\nrecon stack: {len(RESULTS)} checks, {bad} not verified")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
