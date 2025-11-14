import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from statsmodels.stats.proportion import proportions_ztest


def plot_calibration_curve(y_true, y_prob, output_path, n_bins=10):
    """Create and save a calibration (reliability) plot."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed frequency')
    plt.title('Calibration plot')
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def mcnemar_test(y_true, y_pred_a, y_pred_b, exact=False):
    """Run McNemar's test on paired binary predictions.

    Returns: pvalue, stat
    """
    # Build contingency table
    # b = predictions differ: (a_correct, b_correct)
    a_corr = (y_pred_a == y_true)
    b_corr = (y_pred_b == y_true)
    # table: [[both_correct, a_correct_only],[b_correct_only, both_incorrect]]
    both_correct = np.sum(np.logical_and(a_corr, b_corr))
    a_only = np.sum(np.logical_and(a_corr, ~b_corr))
    b_only = np.sum(np.logical_and(~a_corr, b_corr))
    both_wrong = np.sum(np.logical_and(~a_corr, ~b_corr))
    table = np.array([[both_correct, a_only], [b_only, both_wrong]])
    result = mcnemar(table, exact=exact)
    return float(result.pvalue), float(result.statistic)


def delong_roc_variance(ground_truth, predictions):
    """Internal helper: fast implementation of DeLong variance for a single set of predictions.

    Returns AUC and variance estimate.
    """
    # Implementation adapted for clarity from standard DeLong algorithm
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    pos = predictions[ground_truth == 1]
    neg = predictions[ground_truth == 0]
    m = len(pos)
    n = len(neg)
    if m == 0 or n == 0:
        return np.nan, np.nan

    # Compute rankings
    concatenated = np.concatenate([pos, neg])
    order = np.argsort(concatenated)
    # Equivalent AUC calculation
    auc = (np.sum([np.sum(neg < p) + 0.5 * np.sum(neg == p) for p in pos]) ) / (m * n)

    # Compute V10 and V01 as in DeLong
    vx = np.array([np.mean((pos_i > neg).astype(float) + 0.5 * (pos_i == neg).astype(float)) for pos_i in pos])
    vy = np.array([np.mean((pos > neg_j).astype(float) + 0.5 * (pos == neg_j).astype(float)) for neg_j in neg])
    sx = np.var(vx, ddof=1)
    sy = np.var(vy, ddof=1)
    auc_var = (sx / m) + (sy / n)
    return auc, auc_var


def delong_roc_test(ground_truth, pred1, pred2):
    """Compute DeLong test for two correlated ROC AUCs.

    Returns: (auc1, auc2, z, pvalue)
    """
    auc1, var1 = delong_roc_variance(ground_truth, pred1)
    auc2, var2 = delong_roc_variance(ground_truth, pred2)
    # If variances nan, return nans
    if np.isnan(var1) or np.isnan(var2):
        return auc1, auc2, np.nan, np.nan

    # Covariance approximation: assume independence between score sets is conservative (cov=0)
    # A proper covariance estimate is more involved; here we use covariance=0 which may be conservative.
    cov = 0.0
    z = (auc1 - auc2) / np.sqrt(var1 + var2 - 2 * cov)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(auc1), float(auc2), float(z), float(p)


def subgroup_performance(sensitive_values, y_true, y_pred, y_prob=None, metrics=None):
    """Compute performance metrics per subgroup defined by sensitive_values.

    sensitive_values: array-like of group labels (same length as y_true)
    y_true, y_pred: arrays
    y_prob: optional predicted probabilities for ROC AUC
    metrics: list of metrics to compute (supported: accuracy, precision, recall, f1, roc_auc)

    Returns dict: {group: {metric: value, ...}, ...}
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    res = {}
    sv = np.array(sensitive_values)
    groups = np.unique(sv)
    for g in groups:
        mask = sv == g
        yt = np.array(y_true)[mask]
        yp = np.array(y_pred)[mask]
        grp = {}
        if "accuracy" in metrics:
            grp["accuracy"] = float(accuracy_score(yt, yp)) if yt.size else None
        if "precision" in metrics:
            grp["precision"] = float(precision_score(yt, yp)) if yt.size else None
        if "recall" in metrics:
            grp["recall"] = float(recall_score(yt, yp)) if yt.size else None
        if "f1" in metrics:
            grp["f1"] = float(f1_score(yt, yp)) if yt.size else None
        if "roc_auc" in metrics and y_prob is not None:
            grp["roc_auc"] = float(roc_auc_score(yt, np.array(y_prob)[mask])) if yt.size else None
        # confusion as counts
        try:
            cm = confusion_matrix(yt, yp)
            grp["confusion_matrix"] = cm.tolist()
        except Exception:
            grp["confusion_matrix"] = None
        res[str(g)] = grp
    return res


def equalized_odds_check(sensitive_values, y_true, y_pred, alpha=0.05):
    """Check equalized odds across groups: compare TPR and FPR pairwise using proportion z-test.

    Returns dict with per-group TPR/FPR and pairwise test results.
    """
    sv = np.array(sensitive_values)
    groups = np.unique(sv)
    stats_per_group = {}
    for g in groups:
        mask = sv == g
        yt = np.array(y_true)[mask]
        yp = np.array(y_pred)[mask]
        # compute TP,FN,FP,TN
        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel() if yt.size and len(np.unique(yt))>1 else (0,0,0,0)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else None
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        stats_per_group[str(g)] = {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn), "TPR": tpr, "FPR": fpr}

    # pairwise tests
    pairwise = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = groups[i]
            g2 = groups[j]
            mask1 = sv == g1
            mask2 = sv == g2
            yt1 = np.array(y_true)[mask1]
            yp1 = np.array(y_pred)[mask1]
            yt2 = np.array(y_true)[mask2]
            yp2 = np.array(y_pred)[mask2]
            # TPR test (proportion of positives predicted among actual positives)
            tp1 = int(((yt1 == 1) & (yp1 == 1)).sum())
            n1 = int((yt1 == 1).sum())
            tp2 = int(((yt2 == 1) & (yp2 == 1)).sum())
            n2 = int((yt2 == 1).sum())
            tpr_p = None
            tpr_stat = None
            if n1 > 0 and n2 > 0:
                count = np.array([tp1, tp2])
                nobs = np.array([n1, n2])
                stat, pval = proportions_ztest(count, nobs)
                tpr_stat, tpr_p = float(stat), float(pval)

            # FPR test (proportion of positives predicted among actual negatives)
            fp1 = int(((yt1 == 0) & (yp1 == 1)).sum())
            m1 = int((yt1 == 0).sum())
            fp2 = int(((yt2 == 0) & (yp2 == 1)).sum())
            m2 = int((yt2 == 0).sum())
            fpr_p = None
            fpr_stat = None
            if m1 > 0 and m2 > 0:
                countf = np.array([fp1, fp2])
                nobsf = np.array([m1, m2])
                statf, pvalf = proportions_ztest(countf, nobsf)
                fpr_stat, fpr_p = float(statf), float(pvalf)

            pairwise[f"{g1}_vs_{g2}"] = {
                "tpr_test": {"stat": tpr_stat, "pvalue": tpr_p, "significant": (tpr_p is not None and tpr_p < alpha)},
                "fpr_test": {"stat": fpr_stat, "pvalue": fpr_p, "significant": (fpr_p is not None and fpr_p < alpha)}
            }

    return {"per_group": stats_per_group, "pairwise_tests": pairwise}
