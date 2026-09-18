"""Recover V8 force from released tactile MP4 when the raw H5 is gone.

The released tactile video has exactly one frame per parquet row, so video
frame ``i`` maps directly to row ``i``.  H.264 is lossy; outputs from this
module are therefore explicitly marked as MP4 recovery and must not be
represented as bit-equivalent to the raw-H5 V8 channel.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Callable, Iterable, Iterator

import av
import numpy as np
import pyarrow.parquet as pq

from . import calib_free as CF
from .run_episode import (
    FIELDS,
    PIPELINE_VERSION,
    _reference_rows,
    reference_noise_area,
)


STAGE_ROOT = Path(os.environ.get("REACT_STAGE_ROOT", "/media/yxma/Disk1/twm/release"))
OUT_ROOT = Path(os.environ.get(
    "REACT_FORCE_RECOVERY_ROOT",
    "/media/yxma/Disk1/twm/force_v8_mp4_20260918/force"))

OLD_SCOPES = {
    "motherboard": ("2026-05-10", "2026-05-11", "2026-05-19"),
    "pushT": ("2026-06-18",),
}


def decode_rgb(path: Path) -> Iterator[np.ndarray]:
    """Yield every decoded video frame as RGB uint8, in presentation order."""
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def evaluate_fresh_rows(
    frames: Iterable[np.ndarray],
    is_new: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[float, float, float, float]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Evaluate fresh rows and forward-fill duplicate rows."""
    fresh = np.asarray(is_new, bool)
    total = len(fresh)
    out = {key: np.zeros(total, np.float32) for key in FIELDS}
    source = np.zeros(total, np.int32)
    iterator = iter(frames)
    last = None
    last_row = -1
    for row in range(total):
        try:
            frame = next(iterator)
        except StopIteration as exc:
            raise ValueError(
                f"video decoded {row} frames but parquet has {total} parquet rows") from exc
        if fresh[row] or last is None:
            last = evaluate(frame)
            last_row = row
        source[row] = last_row
        for key, value in zip(FIELDS, last):
            out[key][row] = value
    try:
        next(iterator)
    except StopIteration:
        return out, source
    raise ValueError(f"video has more than {total} frames for {total} parquet rows")


def _reference_images(video: Path, rows: np.ndarray, crop) -> np.ndarray:
    wanted = {int(row): slot for slot, row in enumerate(rows)}
    found: list[np.ndarray | None] = [None] * len(rows)
    for index, frame in enumerate(decode_rgb(video)):
        slot = wanted.get(index)
        if slot is not None:
            found[slot] = crop(frame).astype(np.float32)
        if index >= int(rows[-1]) and all(image is not None for image in found):
            break
    if any(image is None for image in found):
        missing = [int(rows[i]) for i, image in enumerate(found) if image is None]
        raise ValueError(f"{video}: missing reference video rows {missing}")
    return np.stack(found)


def process_side(task: str, date: str, episode: str, side: str, *,
                 keep_top_depths: int = 3) -> dict:
    parquet = STAGE_ROOT / task / "meta" / date / f"{episode}.parquet"
    video = STAGE_ROOT / task / "videos" / date / episode / f"tactile_{side}.mp4"
    table = pq.read_table(parquet)
    intensity = np.asarray(table[f"tactile_{side}_intensity"].to_numpy())
    is_new = np.asarray(table[f"tactile_{side}_is_new"].to_numpy(), bool)
    reference_rows = _reference_rows(intensity, is_new)[:12]
    if not len(reference_rows):
        raise ValueError(f"{task}/{date}/{episode}_{side}: no fresh reference rows")

    from .debug_gallery import stages
    from .lut_calibration import crop
    from .react_calib import (
        CALIBRATION_NAME,
        F_MAX_N,
        FORCE_RECONSTRUCTION,
        fit,
        force_stages,
    )

    predict_force = fit(report=False)
    refs = _reference_images(video, reference_rows, crop)
    reference = np.median(refs, axis=0)
    noise_area = reference_noise_area(refs)

    def evaluate(frame: np.ndarray) -> tuple[float, float, float, float]:
        image = crop(frame).astype(np.float32)
        geometry = stages(image, reference)["feats"]
        force = predict_force(
            force_stages(image, reference), noise_area_mm2=noise_area)
        return force, geometry["vol"], geometry["area"], geometry["maxd"]

    out, source = evaluate_fresh_rows(decode_rgb(video), is_new, evaluate)
    out["source_frame"] = source
    kept_depths: list[tuple[int, np.ndarray]] = []
    if keep_top_depths:
        wanted = {int(row): slot for slot, row in enumerate(
            np.argsort(out["force_normal_n"])[::-1][:keep_top_depths])}
        images: dict[int, np.ndarray] = {}
        for row, frame in enumerate(decode_rgb(video)):
            if row in wanted:
                images[row] = frame
            if len(images) == len(wanted):
                break
        for row in wanted:
            depth = stages(crop(images[row]).astype(np.float32), reference)["depth"]
            kept_depths.append((row, depth.astype(np.float32)))

    destination = OUT_ROOT / task / date
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": task,
        "date": date,
        "episode": episode,
        "side": side,
        "source_format": "release_tactile_mp4_h264",
        "source_video": str(video),
        "lossy_input": True,
        "alignment": "video_frame_i_equals_parquet_row_i",
        "contact_threshold_mm": 0.05,
        "valid_mask_dI": float(CF.VALID_DI),
        "reference_noise_area_mm2": noise_area,
        "force_calibration_max_n": F_MAX_N,
        "force_calibration_ceiling_n": predict_force.force_ceiling_n,
        "absolute_force_validated_on_react": False,
        "force_reconstruction": FORCE_RECONSTRUCTION,
        "geometry_reconstruction": "stages (LUT, millimetres)",
        "reference_rows": reference_rows,
        "force_calibration": CALIBRATION_NAME,
        "scale_source": CALIBRATION_NAME,
        "pipeline_version": np.int64(PIPELINE_VERSION),
    }
    output = destination / f"{episode}_{side}.npz"
    np.savez_compressed(
        output,
        **out,
        **{f"depth_row_{row}": depth for row, depth in kept_depths},
        **{key: (np.str_(value) if isinstance(value, str) else value)
           for key, value in metadata.items()},
    )
    metadata["out"] = str(output)
    metadata["force_max_n"] = float(out["force_normal_n"].max())
    metadata["rows"] = len(table)
    metadata["fresh_rows"] = int(is_new.sum())
    return metadata


def jobs(include_push_t: bool = True) -> list[tuple[str, str, str, str]]:
    scopes = OLD_SCOPES if include_push_t else {"motherboard": OLD_SCOPES["motherboard"]}
    result = []
    for task, dates in scopes.items():
        for date in dates:
            for parquet in sorted((STAGE_ROOT / task / "meta" / date).glob("episode_*.parquet")):
                for side in ("left", "right"):
                    result.append((task, date, parquet.stem, side))
    return result


def is_done(job: tuple[str, str, str, str]) -> bool:
    task, date, episode, side = job
    path = OUT_ROOT / task / date / f"{episode}_{side}.npz"
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (int(data.get("pipeline_version", 0)) >= PIPELINE_VERSION
                    and str(data.get("source_format", "")) == "release_tactile_mp4_h264")
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker", type=int)
    parser.add_argument("workers", type=int)
    parser.add_argument("--motherboard-only", action="store_true")
    args = parser.parse_args(argv)
    selected = jobs(not args.motherboard_only)[args.worker::args.workers]
    print(f"[mp4 worker {args.worker}] {len(selected)} sides", flush=True)
    failures = 0
    for job in selected:
        label = "/".join(job)
        if is_done(job):
            print(f"[mp4 worker {args.worker}] skip {label}", flush=True)
            continue
        started = time.time()
        try:
            result = process_side(*job)
            print(json.dumps({
                "worker": args.worker,
                "job": label,
                "seconds": round(time.time() - started, 1),
                "fresh_rows": result["fresh_rows"],
                "max_force_n": round(result["force_max_n"], 3),
            }), flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[mp4 worker {args.worker}] FAIL {label}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
    print(f"[mp4 worker {args.worker}] DONE ({failures} failures)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
