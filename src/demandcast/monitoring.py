"""Univariate feature-drift diagnostics; drift is not model-performance degradation."""

import numpy as np

from .features import HISTORY_FEATURES


def _distribution(values, cuts):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    bins = np.concatenate(([-np.inf], cuts, [np.inf]))
    counts = np.histogram(finite, bins=bins)[0].astype(float)
    counts = np.append(counts, len(values) - len(finite))
    # Additive smoothing prevents infinite PSI when a bin is empty.
    return (counts + 0.5) / (counts.sum() + 0.5 * len(counts))


def build_reference(frame):
    reference = {}
    for feature in HISTORY_FEATURES:
        values = np.asarray(frame[feature], float)
        finite = values[np.isfinite(values)]
        cuts = np.unique(np.quantile(finite, np.linspace(0.1, 0.9, 9))).tolist() if len(finite) else []
        reference[feature] = {"cuts": cuts, "probabilities": _distribution(values, cuts).tolist(),
                              "rows": len(values), "missing_rate": float(np.mean(~np.isfinite(values)))}
    return reference


def monitor_batch(frame, reference, min_rows=100):
    rows = []
    for feature, ref in reference.items():
        current = _distribution(frame[feature], ref["cuts"])
        expected = np.asarray(ref["probabilities"])
        psi = float(np.sum((current - expected) * np.log(current / expected)))
        values = np.asarray(frame[feature], float)
        rows.append({"feature": feature, "psi": psi,
                     "missing_rate": float(np.mean(~np.isfinite(values))) if len(values) else 0.0,
                     "signal": "review" if psi >= 0.20 else "stable"})
    sufficient = len(frame) >= min_rows
    return {"rows": len(frame), "minimum_rows": min_rows, "sufficient_sample": sufficient,
            "status": "insufficient_sample" if not sufficient else (
                "review" if any(r["signal"] == "review" for r in rows) else "stable"),
            "features": rows,
            "caveat": "PSI is a heuristic marginal-distribution check. Seasonality can trigger it; it does not prove concept drift or justify automatic retraining."}
