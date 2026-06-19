"""
labflow.stages.qc_audit

Flag low-quality localizations / tracks over their numeric quality metrics.
`model` (bind) selects the scorer; `randomforest`:

  - if `params['model_path']` points to a trained scikit-learn model -> load it and
    predict (a real supervised audit when a lab has a labelled model);
  - otherwise -> an unsupervised IsolationForest over the numeric features, which
    needs no labels and flags the `contamination` fraction as outliers. This makes
    QC runnable out of the box; supply a trained model to graduate to supervised.

Input : any localization / analysis CSV with numeric quality columns
Output: qc_audit.csv  (input rows + qc_score in [0,1] + qc_flag pass/fail)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..io import read_table, write_table

# Quality-bearing columns we score when present; fall back to all numeric columns.
_FEATURES = ["photons", "background", "confidence", "sigma", "sigma_x", "sigma_y",
             "length_nm", "duration_frames", "diffusion_coefficient", "alpha"]


def run(*, input_csv: str, output_csv: str, params: dict | None = None) -> str:
    params = dict(params or {})
    df = read_table(input_csv)

    feats = [c for c in _FEATURES if c in df.columns] or \
            list(df.select_dtypes("number").columns)
    if not feats:
        raise ValueError("qc_audit found no numeric columns to score.")
    X = df[feats].to_numpy(dtype=float)
    med = np.nanmedian(X) if np.isfinite(X).any() else 0.0
    X = np.nan_to_num(X, nan=med, posinf=med, neginf=med)

    model = str(params.get("model", "randomforest")).lower()
    model_path = str(params.get("model_path") or "")
    if model in ("xgboost", "lightgbm") and not model_path:
        raise NotImplementedError(
            f"{model} QC is supervised - pass model_path=<trained model> "
            f"(or use -b randomforest for unsupervised IsolationForest QC).")
    if model_path:
        import joblib
        clf = joblib.load(model_path)
        if hasattr(clf, "predict_proba"):
            score = clf.predict_proba(X)[:, -1]
        else:
            score = clf.predict(X).astype(float)
        flag = np.where(score >= float(params.get("threshold", 0.5)), "fail", "pass")
    else:
        from sklearn.ensemble import IsolationForest
        contamination = float(params.get("contamination", 0.05))
        iso = IsolationForest(contamination=contamination, random_state=0).fit(X)
        raw = iso.score_samples(X)                      # higher = more normal
        rng = raw.max() - raw.min()
        score = (raw - raw.min()) / (rng + 1e-12)        # 1 = clean, 0 = suspect
        flag = np.where(iso.predict(X) == -1, "fail", "pass")

    out = df.copy()
    out["qc_score"] = np.round(score, 4)
    out["qc_flag"] = flag
    return write_table(out, output_csv)
