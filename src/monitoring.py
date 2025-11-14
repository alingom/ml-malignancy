import os
import numpy as np
from scipy import stats


def psi(expected, actual, buckets=10):
    """Population Stability Index between two distributions (arrays).

    expected, actual: 1D arrays
    returns: PSI value (float)
    """
    expected = np.array(expected).astype(float)
    actual = np.array(actual).astype(float)
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # compute breakpoints
    breaks = np.linspace(0, 100, buckets + 1)
    exp_percents = np.percentile(expected, breaks)
    # avoid duplicates
    exp_percents[0] = -np.inf
    exp_percents[-1] = np.inf

    def _get_counts(arr, bins):
        counts = np.histogram(arr, bins=bins)[0].astype(float)
        # replace zeros with a small value to avoid divide-by-zero
        counts[counts == 0] = 1e-6
        return counts / counts.sum()

    exp_dist = _get_counts(expected, exp_percents)
    act_dist = _get_counts(actual, exp_percents)

    psi_val = np.sum((exp_dist - act_dist) * np.log(exp_dist / act_dist))
    return float(psi_val)


def ks_test_featurewise(baseline_df, current_df, feature_names=None, alpha=0.05):
    """Run KS test per numeric feature; returns dict of {feature: {pvalue, statistic, drift_bool}}"""
    results = {}
    if feature_names is None:
        feature_names = baseline_df.columns.tolist()
    for f in feature_names:
        try:
            a = baseline_df[f].dropna().values
            b = current_df[f].dropna().values
            if len(a) < 2 or len(b) < 2:
                results[f] = {"stat": None, "pvalue": None, "drift": False}
                continue
            stat, p = stats.ks_2samp(a, b)
            results[f] = {"stat": float(stat), "pvalue": float(p), "drift": p < alpha}
        except Exception as e:
            results[f] = {"stat": None, "pvalue": None, "drift": False, "error": str(e)}
    return results


def detect_drift(baseline_df, current_df, features=None, psi_threshold=0.1, alpha=0.05):
    """Run a set of drift checks and return a structured report.

    - PSI per feature
    - KS test per feature
    """
    report = {"psi": {}, "ks": {}, "drift_features": []}
    if features is None:
        features = baseline_df.columns.tolist()

    for f in features:
        try:
            exp = baseline_df[f].dropna().values
            act = current_df[f].dropna().values
            # Only numeric features are checked for PSI/KS
            psi_val = psi(exp, act) if exp.size and act.size else None
            report["psi"][f] = psi_val
        except Exception:
            report["psi"][f] = None

    ks = ks_test_featurewise(baseline_df, current_df, feature_names=features, alpha=alpha)
    report["ks"] = ks

    # collect features with drift by psi or ks
    for f in features:
        psi_flag = False
        try:
            if report["psi"][f] is not None and report["psi"][f] > psi_threshold:
                psi_flag = True
        except Exception:
            psi_flag = False

        ks_flag = ks.get(f, {}).get("drift", False)
        if psi_flag or ks_flag:
            report["drift_features"].append({"feature": f, "psi": report["psi"].get(f), "ks": ks.get(f)})

    return report
