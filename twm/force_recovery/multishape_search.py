"""Test whether the image-based force improvement survives shape changes.

Uses position-grouped selection for a mixed-shape holdout, then nested
leave-one-shape-out evaluation. No held-out shape chooses its own model.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .model_search import (ROOT, load_features, metrics, position_groups,
                           select_model, split_positions, FEATURE_VERSION, CF)


def main():
    families = ("round", "quad", "star", "triangle", "B", "quad_small")
    rows, parts = [], []
    for family in families:
        family_rows, data = load_features(family)
        rows.extend(family_rows)
        parts.append(data)
    data = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    target = np.array([r["f"] for r in rows])
    groups = position_groups(rows)
    shapes = np.array([r["family"] for r in rows])
    gate = np.array([r["area"] < 1 for r in rows])
    configs = [(f"ridge_a{alpha}", feature,
                make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr")))
               for feature in ("geometry", "image", "combined")
               for alpha in (0.1, 1, 10, 100, 1000)]
    train, test = split_positions(groups)
    winner, ranking = select_model({k: v[train] for k, v in data.items()},
                                   target[train], groups[train], configs)
    name, feature, estimator = winner
    fitted = clone(estimator).fit(data[feature][train], target[train])
    pred = np.maximum(fitted.predict(data[feature][test]), 0)
    pred[gate[test]] = 0
    report = {"scope": "Six shapes, same calibration sensor. PushT MAE remains unknown.",
              "selection": "4-fold position-grouped CV on training partition",
              "chosen": {"name": name, "feature": feature}, "ranking": ranking,
              "n_train": int(train.sum()), "n_total": len(target),
              "heldout": metrics(target[test], pred, groups[test]),
              "per_shape_heldout": {family: metrics(target[test][shapes[test] == family],
                  pred[shapes[test] == family], groups[test][shapes[test] == family]) for family in families}}
    print(f"MULTISHAPE {name} {feature}: {report['heldout']}", flush=True)
    path = ROOT / "multishape_report.json"
    path.write_text(json.dumps(report, indent=2))
    joblib.dump({"model": fitted, "name": name, "feature": feature,
                 "feature_version": FEATURE_VERSION, "valid_di": CF.VALID_DI},
                ROOT / "multishape_holdout_model.joblib")
    predictions = np.empty(len(target))
    folds = []
    for tr, te in LeaveOneGroupOut().split(target, target, shapes):
        winner, ranking = select_model({k: v[tr] for k, v in data.items()},
                                       target[tr], shapes[tr], configs, folds=5)
        nm, ft, est = winner
        fitted = clone(est).fit(data[ft][tr], target[tr])
        predictions[te] = np.maximum(fitted.predict(data[ft][te]), 0)
        predictions[te[gate[te]]] = 0
        folds.append({"heldout_shape": str(shapes[te[0]]), "name": nm, "feature": ft,
                      "inner_shape_cv_mae_n": ranking[0]["cv_mae_n"],
                      **metrics(target[te], predictions[te], groups[te])})
        print(f"UNSEEN SHAPE {folds[-1]}", flush=True)
    report["unseen_shapes"] = {**metrics(target, predictions, groups), "folds": folds}
    path.write_text(json.dumps(report, indent=2))
    final = clone(estimator).fit(data[feature], target)
    joblib.dump({"model": final, "name": name, "feature": feature,
                 "feature_version": FEATURE_VERSION, "valid_di": CF.VALID_DI},
                ROOT / "multishape_candidate.joblib")
    print(f"UNSEEN SHAPES MAE {report['unseen_shapes']['mae_n']:.4f} N\n{path}", flush=True)


if __name__ == "__main__":
    main()
