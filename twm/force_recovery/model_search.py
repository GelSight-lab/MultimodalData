"""Reproducible, position-grouped force model search; never changes deployment.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m
twm.force_recovery.model_search. Artifacts live alongside calibration caches.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import cv2
import joblib
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from . import react_calib as RC
from . import calib_free as CF
from .lut_calibration import CNC_MINI_26, PAT, crop

ROOT = RC.CACHE.parent / "model_search_2026-09-16"
FEATURE_VERSION = 1


def position_groups(rows):
    # Preserve the historical partition, but verify that its numeric key is unique.
    xy = np.round([[r["x"], r["y"]] for r in rows], 1)
    key = xy[:, 0] * 1000 + xy[:, 1]
    if len(np.unique(key)) != len(np.unique(xy, axis=0)):
        raise ValueError("Position encoding collision")
    return key


def split_positions(groups, seed=0):
    uniq = np.unique(groups)
    hold = uniq[np.random.default_rng(seed).permutation(len(uniq))[:max(len(uniq) // 3, 1)]]
    test = np.isin(groups, hold)
    return ~test, test


def observation_features(img, ref, stage):
    """Only image-derived quantities; no force, commanded depth or stage pose."""
    d = np.maximum(stage["depth"], 0).astype(np.float32)
    m = np.asarray(stage["contact"], bool)
    yy, xx = np.indices(d.shape, dtype=np.float32)
    weight = d * m
    total = float(weight.sum())
    cx = float((xx * weight).sum() / total) if total > 0 else d.shape[1] / 2
    cy = float((yy * weight).sum() / total) if total > 0 else d.shape[0] / 2
    basic = np.array([stage["feats"][k] for k in RC.FEATURES] + [cx, cy], float)
    qs = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    geom = list(basic)
    geom.extend(np.percentile(d[m], qs) if m.any() else np.zeros(len(qs)))
    geom.extend([m.mean(), np.mean(d), np.std(d)])
    for ax in (xx - cx, yy - cy):
        geom.extend([float((weight * ax ** p).sum() / max(total, 1e-6)) for p in (2, 4)])
    for region in (m[:12], m[-12:], m[:, :12], m[:, -12:]):
        geom.append(float(region.mean()))
    geom.extend(cv2.resize(d, (8, 6), interpolation=cv2.INTER_AREA).ravel())
    diff = np.asarray(img, np.float32) - np.asarray(ref, np.float32)
    relative = diff / np.maximum(ref, 20)
    image = []
    for arr in (diff / 255, relative):
        image.extend(cv2.resize(arr, (8, 6), interpolation=cv2.INTER_AREA).ravel())
        image.extend(cv2.resize(np.abs(arr), (8, 6), interpolation=cv2.INTER_AREA).ravel())
        for ch in range(3):
            image.extend(np.percentile(arr[..., ch], qs))
            image.extend(np.percentile(np.abs(arr[..., ch]), qs))
    geometry = np.asarray(geom, float)
    image = np.asarray(image, float)
    return {"basic": basic, "geometry": geometry, "image": image,
            "combined": np.r_[geometry, image]}


def load_features(family="round"):
    ROOT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / f"{family}_v{FEATURE_VERSION}_di{CF.VALID_DI:g}.joblib"
    if cache.exists():
        return joblib.load(cache)
    directory = CNC_MINI_26 / family
    ref = crop(np.asarray(Image.open(directory / "initial.jpg").convert("RGB"))).astype(np.float32)
    rows, data = [], {k: [] for k in ("basic", "geometry", "image", "combined")}
    baseline = RC.fit(report=False) if family != "round" else None
    for path in sorted(directory.glob("*.jpg")):
        match = PAT.search(path.name)
        if not match or not 0.15 < float(match["f"]) <= RC.F_MAX_N:
            continue
        img = crop(np.asarray(Image.open(path).convert("RGB"))).astype(np.float32)
        stage = RC.force_stages(img, ref)
        if stage["contact"].sum() < 30:
            continue
        features = observation_features(img, ref, stage)
        rows.append({"file": path.name, "family": family,
                     **{k: float(match[k]) for k in ("x", "y", "z", "f")},
                     "area": stage["feats"]["area"]})
        if baseline is not None:
            rows[-1]["baseline_n"] = baseline(stage)
        for key, value in features.items():
            data[key].append(value)
        if len(rows) % 100 == 0:
            print(f"features {family}: {len(rows)}", flush=True)
    result = (rows, {k: np.asarray(v) for k, v in data.items()})
    joblib.dump(result, cache)
    print(f"cached {family}: {len(rows)}", flush=True)
    return result


def candidates(data):
    out = []
    for feature, x in data.items():
        for c in (1, 10, 100):
            for gamma in (0.01, 0.1, 1):
                out.append((f"svr_C{c}_g{gamma}", feature,
                            make_pipeline(StandardScaler(), SVR(C=c, gamma=gamma / x.shape[1], epsilon=0.05))))
        for alpha in (0.001, 0.01, 0.1, 1):
            for gamma in (0.01, 0.1, 1):
                out.append((f"krr_a{alpha}_g{gamma}", feature,
                            make_pipeline(StandardScaler(), GaussianProcessRegressor(
                                kernel=RBF(np.sqrt(x.shape[1] / (2 * gamma)), length_scale_bounds="fixed"),
                                alpha=alpha, optimizer=None))))
        for alpha in (0.1, 10, 1000):
            out.append((f"ridge_a{alpha}", feature, make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))))
        for leaf in (1, 3, 8):
            out.append((f"extra_leaf{leaf}", feature,
                        ExtraTreesRegressor(n_estimators=160, min_samples_leaf=leaf, max_features=0.8, random_state=42, n_jobs=1)))
        for leaf in (10, 25):
            out.append((f"hist_leaf{leaf}", feature,
                        HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15,
                                                      min_samples_leaf=leaf, l2_regularization=1,
                                                      early_stopping=False, random_state=42)))
        if feature == "basic":
            for degree in (2, 3):
                for alpha in (0.1, 10, 1000):
                    out.append((f"poly{degree}_a{alpha}", feature,
                                make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False),
                                              StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))))
        if feature in ("image", "combined"):
            for components in (16, 48):
                for alpha in (0.1, 10):
                    out.append((f"pca{components}_ridge{alpha}", feature,
                                make_pipeline(StandardScaler(), PCA(n_components=components, random_state=42), Ridge(alpha=alpha, solver="lsqr"))))
    return out


def select_model(data, target, groups, configs, folds=4):
    """This function receives only training data; outer labels are inaccessible."""
    splits = list(GroupKFold(n_splits=folds).split(target, target, groups))
    scores = []
    for index, (name, feature, estimator) in enumerate(configs):
        x, errors = data[feature], []
        for train, valid in splits:
            model = clone(estimator).fit(x[train], target[train])
            prediction = np.maximum(model.predict(x[valid]), 0)
            errors.extend(np.abs(prediction - target[valid]))
        scores.append({"name": name, "feature": feature, "cv_mae_n": float(np.mean(errors))})
        if (index + 1) % 20 == 0:
            best = min(scores, key=lambda s: s["cv_mae_n"])
            print(f"search {index + 1}/{len(configs)} best={best}", flush=True)
    winner = int(np.argmin([s["cv_mae_n"] for s in scores]))
    return configs[winner], sorted(scores, key=lambda s: s["cv_mae_n"])


def metrics(target, prediction, groups):
    errors = np.abs(target - prediction)
    rng = np.random.default_rng(42)
    unique = np.unique(groups)
    sums = np.array([errors[groups == g].sum() for g in unique])
    counts = np.array([(groups == g).sum() for g in unique])
    draws = rng.integers(0, len(unique), (2000, len(unique)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    result = {"n": len(target), "positions": len(unique), "mae_n": float(errors.mean()),
              "mae_position_bootstrap_ci95_n": np.percentile(boot, [2.5, 97.5]).tolist(),
              "rmse_n": float(np.sqrt(np.mean(errors ** 2))),
              "rho": float(spearmanr(target, prediction).statistic),
              "median_relative_error": float(np.median(errors / np.maximum(target, 0.15))),
              "bands": []}
    for low, high in ((0, 1), (1, 2), (2, 4), (4, 6), (6, 8.01)):
        mask = (target >= low) & (target < high)
        if mask.any():
            result["bands"].append({"range_n": [low, high], "n": int(mask.sum()), "mae_n": float(errors[mask].mean())})
    return result


def predict_images(bundle, img, ref):
    """Experimental inference; calibration accuracy does not transfer to PushT."""
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Model feature version differs from the active extractor")
    if bundle.get("valid_di") != CF.VALID_DI:
        raise ValueError("Model contact threshold differs from the active reconstruction")
    stage = RC.force_stages(img, ref)
    if stage["feats"]["area"] < 1 or stage["contact"].sum() < 30:
        return 0.0
    features = observation_features(img, ref, stage)[bundle["feature"]]
    return float(max(bundle["model"].predict(features[None])[0], 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested", action="store_true", help="Run four outer folds, each with full inner selection")
    parser.add_argument("--transfer", action="store_true", help="Evaluate the saved round model on five other shapes")
    args = parser.parse_args()
    rows, data = load_features()
    target = np.array([r["f"] for r in rows])
    groups = position_groups(rows)
    train, test = split_positions(groups)
    configs = candidates(data)
    report_path = ROOT / "report.json"
    if args.transfer:
        report = json.loads(report_path.read_text())
        bundle = joblib.load(ROOT / "candidate.joblib")
        report["transfer"] = {}
        for family in ("quad", "star", "triangle", "B", "quad_small"):
            other_rows, other_data = load_features(family)
            y = np.array([r["f"] for r in other_rows])
            pred = np.maximum(bundle["model"].predict(other_data[bundle["feature"]]), 0)
            pred[np.array([r["area"] < 1 for r in other_rows])] = 0
            report["transfer"][family] = metrics(y, pred, position_groups(other_rows))
            report["transfer"][family]["baseline"] = metrics(
                y, np.array([r["baseline_n"] for r in other_rows]), position_groups(other_rows))
            print(f"transfer {family}: {report['transfer'][family]}", flush=True)
            report_path.write_text(json.dumps(report, indent=2))
        return
    chosen, scores = select_model({k: v[train] for k, v in data.items()}, target[train], groups[train], configs)
    name, feature, estimator = chosen
    model = clone(estimator).fit(data[feature][train], target[train])
    pred = np.maximum(model.predict(data[feature][test]), 0)
    # Preserve the production no-contact gate in the headline evaluation.
    pred[np.array([r["area"] < 1 for r in rows])[test]] = 0
    _, historical = RC.fit(report=False, holdout=True)
    np.testing.assert_allclose(target[test], historical["f"])
    report = {"feature_version": FEATURE_VERSION, "valid_di": CF.VALID_DI,
              "selection": "4-fold GroupKFold on historical training positions only",
              "n_candidates": len(configs), "n_train": int(train.sum()),
              "chosen": {"name": name, "feature": feature}, "search": scores,
              "historical_baseline": metrics(target[test], historical["pred"], groups[test]),
              "historical_baseline_caveat": "Spatial gain uses all samples, including heldout commanded depth.",
              "heldout": metrics(target[test], pred, groups[test]),
              "data_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
              "scope": "Single-sensor round indenter calibration, 0.15-8 N. PushT absolute MAE is unknown.",
              "test_predictions": [{**rows[i], "prediction_n": float(p)} for i, p in zip(np.flatnonzero(test), pred)]}
    report_path.write_text(json.dumps(report, indent=2))
    print(f"CHOSEN {name} {feature}\nHELDOUT {report['heldout']}", flush=True)
    # Save the evaluated train-only model separately from the all-round research candidate.
    joblib.dump({"model": model, "feature": feature, "name": name,
                 "feature_version": FEATURE_VERSION, "valid_di": CF.VALID_DI}, ROOT / "holdout_model.joblib")
    final = clone(estimator).fit(data[feature], target)
    joblib.dump({"model": final, "feature": feature, "name": name,
                 "feature_version": FEATURE_VERSION, "valid_di": CF.VALID_DI}, ROOT / "candidate.joblib")
    if args.nested:
        outer = GroupKFold(n_splits=4)
        predictions = np.empty(len(target))
        folds = []
        for fold, (tr, te) in enumerate(outer.split(target, target, groups)):
            winner, ranking = select_model({k: v[tr] for k, v in data.items()}, target[tr], groups[tr], configs)
            nm, ft, est = winner
            fitted = clone(est).fit(data[ft][tr], target[tr])
            predictions[te] = np.maximum(fitted.predict(data[ft][te]), 0)
            predictions[te[np.array([rows[i]["area"] < 1 for i in te])]] = 0
            folds.append({"fold": fold, "name": nm, "feature": ft,
                          "inner_cv_mae_n": ranking[0]["cv_mae_n"], **metrics(target[te], predictions[te], groups[te])})
            print(f"OUTER {fold}: {folds[-1]}", flush=True)
        report["nested_cv"] = {**metrics(target, predictions, groups), "folds": folds,
                               "predictions_n": predictions.tolist()}
        report_path.write_text(json.dumps(report, indent=2))
        print(f"NESTED MAE {report['nested_cv']['mae_n']:.4f} N", flush=True)
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
