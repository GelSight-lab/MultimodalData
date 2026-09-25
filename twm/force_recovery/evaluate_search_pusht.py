"""Unlabeled PushT deployment check for the experimental force regressors."""
from __future__ import annotations

from collections import defaultdict
import json

import h5py
import hdf5plugin  # noqa: F401
import joblib
import numpy as np
import pyarrow.parquet as pq

from . import react_calib as RC
from .lut_calibration import crop
from .model_search import ROOT, observation_features
from .run_episode import DATA_ROOT, STAGE_ROOT, _reference_rows, open_episode


def main():
    source = RC.CACHE.parent / "pusht_dI4_contact_comparison.json"
    rows = json.loads(source.read_text())["rows"]
    models = {name: joblib.load(ROOT / filename) for name, filename in (
        ("round", "candidate.joblib"), ("multishape", "multishape_candidate.joblib"))}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["date"], row["episode"], row["side"]].append(dict(row))
    results = []
    for (date, episode, side), selected in grouped.items():
        path = DATA_ROOT / "pushT" / date / f"{episode}.h5"
        table = pq.read_table(STAGE_ROOT / "pushT" / "meta" / date / f"{episode}.parquet")
        alignment = open_episode(path, "pushT").align[side].index_map
        if len(alignment) != len(table):
            raise ValueError("Force alignment and parquet lengths differ")
        intensity = table[f"tactile_{side}_intensity"].to_numpy()
        fresh = table[f"tactile_{side}_is_new"].to_numpy()
        refs = _reference_rows(intensity, fresh)
        with h5py.File(path, "r") as recording:
            frames = recording[f"gelsight/{side}/frames"]
            reference = np.median([crop(frames[min(int(alignment[r]), len(frames) - 1)]).astype(np.float32)
                                   for r in refs[:12]], axis=0)
            for row in selected:
                index = min(int(alignment[row["row"]]), len(frames) - 1)
                img = crop(frames[index]).astype(np.float32)
                stage = RC.force_stages(img, reference)
                features = observation_features(img, reference, stage)
                contact = stage["feats"]["area"] >= 1 and stage["contact"].sum() >= 30
                for name, bundle in models.items():
                    value = bundle["model"].predict(features[bundle["feature"]][None])[0] if contact else 0
                    row[f"{name}_candidate_n"] = float(max(value, 0))
                row["source_frame"] = index
                results.append(row)
        print(f"PushT {date}/{episode}/{side}: {len(selected)} checked", flush=True)
    summary = {}
    light = np.array([3 <= r["tactile_intensity"] < 6 for r in results])
    for name in ("old_force", "new_force", "round_candidate_n", "multishape_candidate_n"):
        values = np.array([r[name] for r in results])
        summary[name] = {"contact_fraction": float(np.mean(values > 0)),
                         "light_contact_fraction": float(np.mean(values[light] > 0)),
                         "range_n": [float(values.min()), float(values.max())],
                         "above_8n_fraction": float(np.mean(values > 8)),
                         "quantiles_n": np.percentile(values, [25, 50, 75, 95]).tolist()}
    report = {"scope": "835 previously sampled PushT rows, four episode/sensor pairs. No force ground truth; these are NOT accuracy metrics.",
              "n": len(results), "summary": summary, "rows": results}
    output = ROOT / "pusht_candidates.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
