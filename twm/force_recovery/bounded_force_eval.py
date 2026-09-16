"""0-15 N calibration experiment, shape transfer and empirical support checks."""
from __future__ import annotations

import argparse
from collections import Counter
import json

import joblib
import numpy as np
from PIL import Image
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import react_calib as RC
from .bounded_force import (FORCE_LIMIT_N, GuardedForceModel, PositiveGeometryRegressor,
                            RidgeIsotonicRegressor)
from .lut_calibration import CNC_MINI_26, PAT, crop
from .model_search import (ROOT as OLD_ROOT, CF, FEATURE_VERSION, load_features,
                           observation_features, position_groups, split_positions)

ROOT = OLD_ROOT.parent / "bounded_force_0_15_2026-09-16"
FAMILIES = ("round", "quad", "star", "triangle", "B", "quad_small")
DOMAIN = "cnc_mini_26_calibration"


def load_full_range(family):
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{family}_v{FEATURE_VERSION}_di{CF.VALID_DI:g}.joblib"
    if path.exists():
        return joblib.load(path)
    old_rows, old_data = load_features(family)
    rows = list(old_rows)
    features = {k: list(v) for k, v in old_data.items()}
    seen = {r["file"] for r in rows}
    directory = CNC_MINI_26 / family
    ref = crop(np.asarray(Image.open(directory / "initial.jpg").convert("RGB"))).astype(np.float32)
    for image_path in sorted(directory.glob("*.jpg")):
        match = PAT.search(image_path.name)
        if not match or image_path.name in seen or not 0 <= float(match["f"]) <= FORCE_LIMIT_N:
            continue
        img = crop(np.asarray(Image.open(image_path).convert("RGB"))).astype(np.float32)
        stage = RC.force_stages(img, ref)
        values = observation_features(img, ref, stage)
        # Include low-force and weak-mask examples; selection must not discard misses.
        rows.append({"file": image_path.name, "family": family,
                     **{k: float(match[k]) for k in ("x", "y", "z", "f")},
                     "area": stage["feats"]["area"]})
        for key, value in values.items():
            features[key].append(value)
        if (len(rows) - len(old_rows)) % 100 == 0:
            print(f"extend {family}: +{len(rows) - len(old_rows)}", flush=True)
    data = {k: np.asarray(v) for k, v in features.items()}
    data["physical"] = data["basic"][:, :5]
    joblib.dump((rows, data), path)
    print(f"0-15 N {family}: {len(rows)}", flush=True)
    return rows, data


def configs():
    result = [("nonnegative_physics", "physical", PositiveGeometryRegressor())]
    for feature in ("basic", "geometry", "combined"):
        for k in (5, 25):
            result.append((f"knn{k}", feature, make_pipeline(StandardScaler(), KNeighborsRegressor(k, weights="distance"))))
    for feature in ("basic", "geometry"):
        result.append(("extra_trees", feature,
                       ExtraTreesRegressor(n_estimators=80, min_samples_leaf=3,
                                           max_features=0.7, random_state=42, n_jobs=1)))
    for alpha in (1, 10, 100):
        result.append((f"ridge_isotonic{alpha}", "combined", RidgeIsotonicRegressor(alpha)))
    return result


def select_by_shape(data, y, shapes):
    splits = list(LeaveOneGroupOut().split(y, y, shapes))
    ranking = []
    choices = configs()
    for name, feature, estimator in choices:
        errors, fold_errors = [], []
        for train, valid in splits:
            model = clone(estimator).fit(data[feature][train], y[train])
            pred = np.clip(model.predict(data[feature][valid]), 0, FORCE_LIMIT_N)
            residuals = np.abs(pred - y[valid])
            errors.extend(residuals)
            fold_errors.append(float(residuals.mean()))
        ranking.append({"name": name, "feature": feature, "mae_n": float(np.mean(errors)),
                        "worst_shape_mae_n": max(fold_errors)})
    # Mean error selects the model; worst-shape error remains visible in the report.
    winner = int(np.argmin([r["mae_n"] for r in ranking]))
    return choices[winner], sorted(ranking, key=lambda r: r["mae_n"])


def evaluate(model, x, y):
    pred = model.predict_candidates(x)
    result = model.predict_result(x, domain=DOMAIN)
    supported = result["supported"]
    summary = {"n": len(y), "mae_all_n": float(np.abs(pred - y).mean()),
               "coverage": float(supported.mean()),
               "mae_supported_n": float(np.abs(result["force_n"][supported] - y[supported]).mean()) if supported.any() else None,
               "prediction_range_n": [float(pred.min()), float(pred.max())],
               "empirical_error_radius_n": model.error_radius_n_,
               "reasons": dict(Counter(result["reason"].tolist())), "bands": []}
    for low, high in ((0, 1), (1, 4), (4, 8), (8, 12), (12, 15.001)):
        mask = (y >= low) & (y < high)
        if mask.any():
            summary["bands"].append({"range_n": [low, min(high, 15)], "n": int(mask.sum()),
                                     "mae_all_n": float(np.abs(pred[mask] - y[mask]).mean()),
                                     "coverage": float(supported[mask].mean())})
    return summary, pred, supported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--nested-shapes", action="store_true")
    parser.add_argument("--position-diagnostics", action="store_true",
                        help="Score bounded ridge variants descriptively; do not select on this test")
    args = parser.parse_args()
    rows, parts = [], []
    for family in FAMILIES:
        r, d = load_full_range(family)
        rows.extend(r)
        parts.append(d)
    if args.build_only:
        return
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    y = np.array([r["f"] for r in rows])
    shapes = np.array([r["family"] for r in rows])
    groups = position_groups(rows)
    development, test = split_positions(groups, seed=0)
    dev_indices = np.flatnonzero(development)
    fit_local, cal_local = split_positions(groups[development], seed=42)
    fit_indices, cal_indices = dev_indices[fit_local], dev_indices[cal_local]
    if args.position_diagnostics:
        diagnostics = []
        for name, feature, estimator in configs():
            if not name.startswith("ridge_isotonic"):
                continue
            model = GuardedForceModel(estimator).fit(data[feature][fit_indices], y[fit_indices],
                                                    data[feature][cal_indices], y[cal_indices], domain=DOMAIN)
            summary = evaluate(model, data[feature][test], y[test])[0]
            diagnostics.append({"name": name, "feature": feature, **summary})
            print(f"POSITION DIAGNOSTIC {name}: {summary}", flush=True)
        (ROOT / "position_diagnostics.json").write_text(json.dumps(
            {"not_used_for_selection": True, "results": diagnostics}, indent=2))
        return
    winner, ranking = select_by_shape({k: v[fit_indices] for k, v in data.items()}, y[fit_indices], shapes[fit_indices])
    name, feature, estimator = winner
    model = GuardedForceModel(estimator).fit(data[feature][fit_indices], y[fit_indices],
                                            data[feature][cal_indices], y[cal_indices], domain=DOMAIN)
    summary, pred, supported = evaluate(model, data[feature][test], y[test])
    report = {"scope": "0-15 N calibration with bounded output and empirical support rejection; NOT an OOD-free model",
              "n_total": len(y), "n_fit": len(fit_indices), "n_calibration": len(cal_indices),
              "n_test": int(test.sum()), "force_range_n": [float(y.min()), float(y.max())],
              "count_8_15_n": int(np.sum(y > 8)), "count_0_1_n": int(np.sum(y < 1)),
              "chosen": {"name": name, "feature": feature}, "selection_by_shape": ranking,
              "heldout_positions": summary,
              "test_predictions": [{**rows[i], "bounded_candidate_n": float(p), "supported": bool(s)}
                                   for i, p, s in zip(np.flatnonzero(test), pred, supported)]}
    path = ROOT / "report.json"
    path.write_text(json.dumps(report, indent=2))
    # Keep the calibration set separate instead of refitting on it after threshold selection.
    joblib.dump({"model": model, "feature": feature, "feature_version": FEATURE_VERSION,
                 "valid_di": CF.VALID_DI, "domain": DOMAIN}, ROOT / "guarded_model.joblib")
    print(f"CHOSEN {name}/{feature}\nPOSITIONS {summary}", flush=True)
    if args.nested_shapes:
        folds = []
        for train, valid in LeaveOneGroupOut().split(y, y, shapes):
            tr_local, ca_local = split_positions(groups[train], seed=42)
            tr, ca = train[tr_local], train[ca_local]
            chosen, table = select_by_shape({k: v[tr] for k, v in data.items()}, y[tr], shapes[tr])
            nm, ft, est = chosen
            guard = GuardedForceModel(est).fit(data[ft][tr], y[tr], data[ft][ca], y[ca], domain=DOMAIN)
            scores, _, _ = evaluate(guard, data[ft][valid], y[valid])
            folds.append({"shape": str(shapes[valid[0]]), "model": nm, "feature": ft, **scores})
            report["unseen_shapes"] = folds
            path.write_text(json.dumps(report, indent=2))
            print(f"UNSEEN {folds[-1]}", flush=True)
    print(path, flush=True)


if __name__ == "__main__":
    main()
