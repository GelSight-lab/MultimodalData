"""Sensor-transfer diagnostics; support coverage is not a proof of accuracy."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json

import h5py
import hdf5plugin  # noqa: F401
import joblib
import numpy as np
import pyarrow.parquet as pq

from . import react_calib as RC
from .bounded_force_eval import ROOT, DOMAIN, evaluate
from .force_recon_matrix import _rows
from .lut_calibration import crop
from .model_search import CF, FEATURE_VERSION, observation_features
from .run_episode import DATA_ROOT, STAGE_ROOT, _reference_rows, open_episode


def pack(records, columns):
    data = {k: np.asarray(v) for k, v in columns.items()}
    data["physical"] = data["basic"][:, :5]
    return records, data


def collect_other_sensor(name, cap=192):
    path = ROOT / f"transfer_{name}_{cap}_v{FEATURE_VERSION}_di{CF.VALID_DI:g}.joblib"
    if path.exists():
        return joblib.load(path)
    records, columns = [], defaultdict(list)
    rows, get = _rows(name)
    order = np.random.default_rng(17).permutation(len(rows))
    for index in order:
        row = rows[index]
        if not np.isfinite(row["f"]) or not 0 <= row["f"] <= 15:
            continue
        img, ref = get(row)
        features = observation_features(img, ref, RC.force_stages(img, ref))
        for key, value in features.items():
            columns[key].append(value)
        records.append({"f": float(row["f"]), "group": str(row["group"]),
                        "source": str(row.get("path", row.get("key", index)))})
        if len(records) >= cap:
            break
    result = pack(records, columns)
    joblib.dump(result, path)
    print(f"cross-sensor cache {name}: {len(records)}", flush=True)
    return result


def collect_pusht():
    path = ROOT / f"transfer_pusht_v{FEATURE_VERSION}_di{CF.VALID_DI:g}.joblib"
    if path.exists():
        return joblib.load(path)
    source = RC.CACHE.parent / "pusht_dI4_contact_comparison.json"
    selected = json.loads(source.read_text())["rows"]
    batches = defaultdict(list)
    for row in selected:
        batches[row["date"], row["episode"], row["side"]].append(row)
    records, columns = [], defaultdict(list)
    for (date, episode, side), items in batches.items():
        h5path = DATA_ROOT / "pushT" / date / f"{episode}.h5"
        table = pq.read_table(STAGE_ROOT / "pushT" / "meta" / date / f"{episode}.parquet")
        index_map = open_episode(h5path, "pushT").align[side].index_map
        if len(index_map) != len(table):
            raise ValueError("Episode alignment and parquet lengths differ")
        refs = _reference_rows(table[f"tactile_{side}_intensity"].to_numpy(),
                               table[f"tactile_{side}_is_new"].to_numpy())
        with h5py.File(h5path, "r") as recording:
            frames = recording[f"gelsight/{side}/frames"]
            reference = np.median([crop(frames[min(int(index_map[r]), len(frames) - 1)]).astype(np.float32)
                                   for r in refs[:12]], axis=0)
            for row in items:
                frame = min(int(index_map[row["row"]]), len(frames) - 1)
                img = crop(frames[frame]).astype(np.float32)
                stage = RC.force_stages(img, reference)
                features = observation_features(img, reference, stage)
                for key, value in features.items():
                    columns[key].append(value)
                records.append({**row, "source_frame": frame})
        print(f"PushT cache {date}/{episode}/{side}: {len(items)}", flush=True)
    result = pack(records, columns)
    joblib.dump(result, path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    collected = {name: collect_other_sensor(name) for name in ("cnc", "faf")}
    collected["pusht"] = collect_pusht()
    if args.prepare:
        return
    bundle = joblib.load(ROOT / "guarded_model.joblib")
    model, feature = bundle["model"], bundle["feature"]
    report = {"note": "Diagnostic support-only scores deliberately bypass the domain lock. Actual unregistered domains are rejected. Neither result establishes OOD immunity.", "datasets": {}}
    for name, (rows, data) in collected.items():
        x = data[feature]
        actual = model.predict_result(x, domain=name)
        assert not actual["supported"].any() and np.isnan(actual["force_n"]).all()
        pred = model.predict_candidates(x)
        support_only = model.predict_result(x, domain=DOMAIN)
        scores = {"n": len(rows), "registered_domain_coverage": float(actual["supported"].mean()),
                  "diagnostic_support_only_coverage": float(support_only["supported"].mean()),
                  "bounded_prediction_range_n": [float(pred.min()), float(pred.max())],
                  "prediction_quantiles_n": np.percentile(pred, [5, 50, 95]).tolist(),
                  "above_15_n": int(np.sum(pred > 15))}
        if name != "pusht":
            y = np.array([r["f"] for r in rows])
            scores["diagnostic_errors"] = evaluate(model, x, y)[0]
        else:
            scores["absolute_error"] = "Unknown: no PushT force ground truth"
        report["datasets"][name] = scores
        print(name, json.dumps(scores), flush=True)
    (ROOT / "transfer_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
