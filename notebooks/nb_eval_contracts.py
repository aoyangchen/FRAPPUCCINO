"""
nb_eval_contracts.py
====================

Phase 2 extracted evaluation/bundle-completeness helpers for the enzyme reactivity notebook.

These helpers are intentionally side-effect free at import time. Runtime-specific
notebook globals remain notebook-local; the Phase 2 notebook rebinds imported helper
globals in-place so behavior matches the original notebook as closely as can be
established statically.
"""

from __future__ import annotations

import contextlib
import io as _io
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from nb_run_contracts import _ensure_dir, _read_manifest


def weighted_ece(y_true, p, w=None, n_bins=10):
    """Compute the weighted expected calibration error (ECE).

    This helper implements a classic reliability metric by binning the
    predicted probabilities and comparing the weighted average predicted
    confidence to the weighted empirical positive rate within each bin.

    Parameters
    ----------
    y_true : array‑like
        Binary ground‑truth labels (0/1) for each sample.
    p : array‑like
        Predicted probabilities for the positive class.
    w : array‑like, optional
        Non‑negative sample weights.  When ``None`` (default), equal
        weights are assumed.
    n_bins : int, optional
        Number of bins used to partition the [0, 1] probability
        interval.  More bins provide a finer calibration curve at the
        expense of higher variance.

    Returns
    -------
    float
        The weighted ECE, computed as the sum of the absolute
        differences between the weighted bin accuracy and the weighted
        mean predicted probability, scaled by the bin weights.
    """
    y_true = np.asarray(y_true, float)
    p      = np.asarray(p, float)
    w      = np.ones_like(y_true, float) if w is None else np.asarray(w, float)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    totw = float(np.sum(w)) + 1e-12
    ece  = 0.0

    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (p >= b0) & (p < b1) if b1 < 1.0 else (p >= b0) & (p <= b1)
        if not np.any(m):
            continue
        wb, pb, yb = w[m], p[m], y_true[m]
        wsum = float(np.sum(wb))
        conf = float(np.sum(wb * pb) / (wsum + 1e-12))
        acc  = float(np.sum(wb * yb) / (wsum + 1e-12))
        ece += (wsum / totw) * abs(acc - conf)

    return float(ece)

def _cm_rates_from_weighted_counts(tn, fp, fn, tp):
    """Return common performance rates from weighted confusion matrix counts.

    Given weighted true/false positives/negatives, this helper derives
    precision, recall, F1, balanced accuracy, Matthews correlation
    coefficient (MCC) and true negative rate (TNR).  The implementation
    guards against divide‑by‑zero by returning zero when a term's
    denominator is zero.

    Parameters
    ----------
    tn, fp, fn, tp : float
        Weighted counts of true negatives, false positives, false
        negatives and true positives, respectively.

    Returns
    -------
    dict
        A dictionary containing the derived rates.
    """
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1  = (2*ppv*tpr/(ppv+tpr)) if (ppv+tpr) else 0.0
    bal = 0.5*(tpr+tnr)
    denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    mcc = ((tp*tn - fp*fn)/math.sqrt(denom)) if denom > 0 else 0.0
    return dict(precision=ppv, recall=tpr, f1=f1, balanced_accuracy=bal, mcc=mcc, tnr=tnr)

def _as_threshold_dict(thresholds):
    """Normalize various threshold specifications to a uniform dict form.

    Accepts a variety of threshold descriptions and returns a mapping from
    human‑readable names to threshold values.  Thresholds may be given
    as a scalar, an iterable of scalars, a dictionary mapping names to
    scalars, or ``None``.  When ``None`` is passed, the function
    returns ``None`` to indicate that no thresholding should occur.

    Parameters
    ----------
    thresholds : float, int, sequence of float, dict or None
        Threshold specification.  See above for supported types.

    Returns
    -------
    dict or None
        Normalized mapping from names to threshold values, or ``None``.
    """
    if thresholds is None:
        return None
    if isinstance(thresholds, (float, int, np.floating)):
        t = float(thresholds)
        return {f"t_{t:.3f}": t}
    if isinstance(thresholds, (list, tuple)):
        return {f"t_{float(t):.3f}": float(t) for t in thresholds}
    if isinstance(thresholds, dict):
        return {str(k): float(v) for k, v in thresholds.items()}
    raise TypeError("thresholds must be None, float, list/tuple, or dict[str,float]")

def _threshold_report(y, p, w, t):
    """Return a detailed report for a single threshold.

    Generates predictions by thresholding probabilities, computes the
    weighted confusion matrix, and derives rate metrics.  The report
    mirrors the structure expected by downstream helpers in this module.

    Parameters
    ----------
    y : array‑like
        Ground‑truth binary labels.
    p : array‑like
        Predicted probabilities.
    w : array‑like
        Sample weights.
    t : float
        Threshold applied to the probabilities.

    Returns
    -------
    dict
        A dictionary containing the threshold, weighted counts,
        derived rates, the confusion matrix and the predicted labels.
    """
    yhat = (p >= t).astype(int)
    cm_w = confusion_matrix(y, yhat, labels=[0, 1], sample_weight=w)
    tn, fp, fn, tp = map(float, cm_w.ravel())
    rates = _cm_rates_from_weighted_counts(tn, fp, fn, tp)
    return dict(
        threshold=float(t),
        counts_w=dict(tn=tn, fp=fp, fn=fn, tp=tp),
        rates=rates,
        cm_w=cm_w,
        yhat=yhat,
    )

def postprocess_eval_dir(
    eval_dir: Path, *,
    t_report=0.50,
    do_calib=True,
    do_thr_sweep=True,
    do_cm=True,
    do_per_enzyme=True,
    ece_bins=10,
    sweep_n_t=401,
):
    """
    Expects eval_dir contains preds.csv with at least:
      - prob_raw
      - y_true (for labeled)
      - weight (optional)

    Writes canonical files into eval_dir:
      - roc_curve.png, pr_curve.png, score_hist.png
      - reliability_curve.png, pr_calibration_summary.json
      - confusion_matrix_report.json + cm plots
      - threshold_sweep_weighted.csv
      - per_enzyme_confusion_at_t_report.csv, per_enzyme_metrics_at_t_report.csv
      - per_enzyme_plots/*
    """
    eval_dir = Path(eval_dir)
    preds_fp = eval_dir / "preds.csv"
    if not preds_fp.exists():
        return

    df = pd.read_csv(preds_fp)
    if "prob_raw" not in df.columns:
        return

    p = df["prob_raw"].to_numpy(dtype=float)
    w = df["weight"].to_numpy(dtype=float) if "weight" in df.columns else np.ones(len(df), float)

    if "y_true" not in df.columns:
        # ranking-only
        return

    y = df["y_true"].to_numpy(dtype=int)

    # --- threshold-free metrics ---
    eps = 1e-15
    p_clip = np.clip(p, eps, 1 - eps)

    auroc = float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else float("nan")
    ap    = float(average_precision_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else float("nan")
    ll    = float(log_loss(y, p_clip, sample_weight=w, labels=[0, 1]))
    brier = float(brier_score_loss(y, p, sample_weight=w))
    pos_rate = float(np.average(y, weights=w))
    ece = float(weighted_ece(y, p, w, n_bins=ece_bins)) if do_calib else None

    # --- ROC/PR plots ---
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC (weighted)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(eval_dir / "roc_curve.png", dpi=180)
        plt.close()

        prec, rec, _ = precision_recall_curve(y, p, sample_weight=w)
        plt.figure()
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.hlines(pos_rate, 0, 1, linestyles="--", linewidth=1, label=f"Chance={pos_rate:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision–Recall (weighted)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(eval_dir / "pr_curve.png", dpi=180)
        plt.close()

    # --- histogram ---
    plt.figure()
    plt.hist(p[y == 0], bins=50, alpha=0.7, density=True, label="neg")
    plt.hist(p[y == 1], bins=50, alpha=0.7, density=True, label="pos")
    plt.xlabel("Predicted probability"); plt.ylabel("Density")
    plt.title("Score histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_dir / "score_hist.png", dpi=180)
    plt.close()

    # --- calibration / reliability ---
    if do_calib:
        bins = np.linspace(0, 1, int(ece_bins) + 1)
        xs, ys = [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = (p >= b0) & (p < b1) if b1 < 1.0 else (p >= b0) & (p <= b1)
            if not np.any(m):
                continue
            wb, pb, yb = w[m], p[m], y[m]
            wsum = float(np.sum(wb))
            xs.append(float(np.sum(wb * pb) / (wsum + 1e-12)))
            ys.append(float(np.sum(wb * yb) / (wsum + 1e-12)))

        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="ideal")
        if xs:
            plt.plot(xs, ys, marker="o")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Empirical positive rate")
        plt.title("Reliability curve (weighted bins)")
        plt.tight_layout()
        plt.savefig(eval_dir / "reliability_curve.png", dpi=180)
        plt.close()

        (eval_dir / "pr_calibration_summary.json").write_text(json.dumps({
            "n": int(len(y)),
            "weight_sum": float(np.sum(w)),
            "pos_rate_weighted": float(pos_rate),
            "auroc_weighted": float(auroc),
            "average_precision_weighted": float(ap),
            "log_loss_weighted": float(ll),
            "brier_weighted": float(brier),
            "ece_weighted_L1": float(ece),
            "ece_bins": int(ece_bins),
        }, indent=2))

    # --- canonical CM report @ t_report ---
    if do_cm:
        rep = _threshold_report(y, p, w, float(t_report))
        (eval_dir / "confusion_matrix_report.json").write_text(json.dumps({
            "t_report": float(t_report),
            "counts_w": rep["counts_w"],
            "rates": rep["rates"],
        }, indent=2))

        disp = ConfusionMatrixDisplay(rep["cm_w"], display_labels=["neg", "pos"])
        disp.plot(values_format=".1f")
        plt.title(f"Confusion matrix (weighted) @ t={t_report:.2f}")
        plt.tight_layout()
        plt.savefig(eval_dir / "confusion_matrix_report_raw.png", dpi=180)
        plt.close()

        cm_norm_true = confusion_matrix(y, rep["yhat"], labels=[0, 1], sample_weight=w, normalize="true")
        disp = ConfusionMatrixDisplay(cm_norm_true, display_labels=["neg", "pos"])
        disp.plot(values_format=".2f")
        plt.title(f"CM norm(true) @ t={t_report:.2f}")
        plt.tight_layout()
        plt.savefig(eval_dir / "confusion_matrix_report_raw_normalized_true.png", dpi=180)
        plt.close()

    # --- threshold sweep (inspection) ---
    if do_thr_sweep:
        thr_grid = np.linspace(0.0, 1.0, int(sweep_n_t))
        rows = []
        for t in thr_grid:
            rep = _threshold_report(y, p, w, float(t))
            r = rep["rates"]
            rows.append({
                "threshold": float(t),
                "tn_w": rep["counts_w"]["tn"],
                "fp_w": rep["counts_w"]["fp"],
                "fn_w": rep["counts_w"]["fn"],
                "tp_w": rep["counts_w"]["tp"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "mcc": r["mcc"],
                "balanced_accuracy": r["balanced_accuracy"],
            })
        pd.DataFrame(rows).to_csv(eval_dir / "threshold_sweep_weighted.csv", index=False)

    # --- per-enzyme reports ---
    if do_per_enzyme and ("enzyme" in df.columns):
        g = df["enzyme"].astype(str).fillna("NA")
        rep_rows_c, rep_rows_m = [], []

        for enz, idx in g.groupby(g).groups.items():
            idx = np.array(list(idx), dtype=int)
            if len(idx) == 0:
                continue
            ye, pe, we = y[idx], p[idx], w[idx]
            rep = _threshold_report(ye, pe, we, float(t_report))
            c, r = rep["counts_w"], rep["rates"]

            rep_rows_c.append({
                "enzyme": enz,
                "n_rows": int(len(idx)),
                "weight_sum": float(np.sum(we)),
                "tn_w": c["tn"], "fp_w": c["fp"], "fn_w": c["fn"], "tp_w": c["tp"],
            })
            rep_rows_m.append({
                "enzyme": enz,
                "n_rows": int(len(idx)),
                "weight_sum": float(np.sum(we)),
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "mcc": r["mcc"],
                "balanced_accuracy": r["balanced_accuracy"],
                "tnr": r["tnr"],
            })

        df_c = pd.DataFrame(rep_rows_c).sort_values(["weight_sum", "n_rows"], ascending=False)
        df_m = pd.DataFrame(rep_rows_m).sort_values(["weight_sum", "n_rows"], ascending=False)

        df_c.to_csv(eval_dir / "per_enzyme_confusion_at_t_report.csv", index=False)
        df_m.to_csv(eval_dir / "per_enzyme_metrics_at_t_report.csv", index=False)

        pdir = eval_dir / "per_enzyme_plots"
        pdir.mkdir(parents=True, exist_ok=True)

        x = np.log10(np.maximum(df_m["weight_sum"].to_numpy(float), 1e-6))
        yerr = 1.0 - df_m["balanced_accuracy"].to_numpy(float)
        plt.figure()
        plt.scatter(x, yerr)
        plt.xlabel("log10(weight_sum)")
        plt.ylabel("1 - balanced_accuracy @ t_report")
        plt.title("Per-enzyme error profile")
        plt.tight_layout()
        plt.savefig(pdir / "error_profile_scatter.png", dpi=180)
        plt.close()

        top = df_m.head(50).copy()
        plt.figure()
        plt.plot(top["mcc"].to_numpy(float), label="MCC")
        plt.plot(top["recall"].to_numpy(float), label="Recall")
        plt.xlabel("Top enzymes (by weight_sum)")
        plt.ylabel("Metric value")
        plt.title("Ranked MCC + Recall (top 50)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(pdir / "ranked_mcc_with_recall_ci.png", dpi=180)
        plt.close()

        M = top[["precision", "recall", "f1", "mcc", "balanced_accuracy"]].to_numpy(float).T
        plt.figure()
        plt.imshow(M, aspect="auto")
        plt.yticks(range(M.shape[0]), ["precision", "recall", "f1", "mcc", "bal_acc"])
        plt.xlabel("Top enzymes (by weight_sum)")
        plt.title("Per-enzyme metrics heatmap (top 50)")
        plt.tight_layout()
        plt.savefig(pdir / "metrics_heatmap.png", dpi=180)
        plt.close()

def _eval_headline(split_name: str, y, p, w, thresholds=None):
    """Summarize threshold‑free and thresholded performance metrics.

    Builds a canonical "headline" dictionary used throughout the GT1
    workflow.  It aggregates the number of samples, weighted sums,
    positive rate, AUROC, average precision, log loss, and Brier score.
    Optionally, it can compute per‑threshold confusion matrix counts and
    derived rates using ``_threshold_report``.

    Parameters
    ----------
    split_name : str
        Name of the evaluation split (e.g. ``"A0"`` or ``"test"``).
    y : array‑like
        Ground‑truth binary labels.
    p : array‑like
        Predicted probabilities for the positive class.
    w : array‑like
        Sample weights.
    thresholds : scalar, sequence or dict, optional
        Optional threshold specification.  See ``_as_threshold_dict`` for
        accepted formats.

    Returns
    -------
    dict
        A dictionary with top‑level summary metrics and, if
        ``thresholds`` is provided, a nested ``thresholded`` mapping.
    """
    eps = 1e-15
    p_clip = np.clip(p, eps, 1 - eps)

    auroc = float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else float("nan")
    ap    = float(average_precision_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else float("nan")
    ll    = float(log_loss(y, p_clip, sample_weight=w))
    brier = float(brier_score_loss(y, p, sample_weight=w))
    pos_rate = float(np.average(y, weights=w))

    headline = dict(
        split=split_name,
        n=int(len(y)),
        weight_sum=float(np.sum(w)),
        pos_rate_weighted=pos_rate,
        auroc_weighted=auroc,
        ap_weighted=ap,
        log_loss_weighted=ll,
        brier_weighted=brier,
        thresholded=None,
    )

    thr_dict = _as_threshold_dict(thresholds)
    if thr_dict is not None:
        headline["thresholded"] = {}
        for name, t in thr_dict.items():
            rep = _threshold_report(y, p, w, t)
            headline["thresholded"][name] = {
                "threshold": rep["threshold"],
                "counts_w": rep["counts_w"],
                "rates": rep["rates"],
            }

    return headline

def _eval_and_write(run_dir: Path, split_name: str, df_part: pd.DataFrame, y, w, p,
                   thresholds, prefix: str):
    """Write canonical evaluation artifacts for a single split.

    This helper persists ground‑truth labels, predicted scores, sample
    weights, prediction tables, the headline JSON and various plots
    under a run directory.  When canonical evaluation output is
    requested (via global flags), it also writes standardized arrays,
    the unfiltered predictions, and invokes ``postprocess_eval_dir`` to
    generate reliability curves and per‑enzyme summaries.  Confusion
    matrices are produced both at default thresholds and at any
    explicitly provided thresholds.

    Parameters
    ----------
    run_dir : Path
        Base directory for the current run.  Artifacts are written
        underneath ``run_dir / prefix``.
    split_name : str
        Name of the current split (e.g. ``"train"``, ``"val"``, ``"test"``).
    df_part : pd.DataFrame
        Slice of the full dataset corresponding to the split; used to
        extract metadata columns for the predictions table.
    y : array‑like
        Ground‑truth binary labels.
    w : array‑like
        Sample weights.
    p : array‑like
        Predicted probabilities.
    thresholds : scalar, sequence or dict, optional
        Threshold specification for additional confusion matrices.  See
        ``_as_threshold_dict`` for details.
    prefix : str
        Subdirectory name used to separate multiple evaluation runs
        under ``run_dir``.

    Returns
    -------
    dict
        The headline summary dictionary produced by ``_eval_headline``.
    """
    out   = _ensure_dir(run_dir / prefix)
    plots = _ensure_dir(out / "plots")

    # Save arrays
    np.save(out / f"{split_name}_y_true.npy",  y.astype(int))
    np.save(out / f"{split_name}_y_score.npy", p.astype(float))
    np.save(out / f"{split_name}_w.npy",       w.astype(float))

    # preds table
    cols_meta = [c for c in ["pair_id", "enzyme", "enz_idx", "sub_idx", "source", "organism"] if c in df_part.columns]
    preds = df_part[cols_meta].copy() if cols_meta else pd.DataFrame(index=np.arange(len(y)))
    preds["y_true"]   = y.astype(int)
    preds["prob_raw"] = p.astype(float)
    preds["weight"]   = w.astype(float)
    preds.to_csv(out / f"{split_name}_preds.csv", index=False)

    # Canonical eval dict (shared)
    headline = _eval_headline(split_name=split_name, y=y, p=p, w=w, thresholds=thresholds)
    (out / f"{split_name}_headline.json").write_text(json.dumps(headline, indent=2))

    # -----------------------------
    # NEW: canonical filenames for audits + postprocess
    # -----------------------------
    if EVAL_WRITE_CANONICAL:
        # canonical arrays
        np.save(out / "y_true.npy",  y.astype(int))
        np.save(out / "y_score.npy", p.astype(float))
        np.save(out / "w_test.npy",  w.astype(float))

        # canonical preds + headline
        preds.to_csv(out / "preds.csv", index=False)
        (out / "headline.json").write_text(json.dumps(headline, indent=2))

        # postprocess: generate canonical plots/tables into THIS eval dir
        postprocess_eval_dir(
            out,
            t_report=EVAL_T_REPORT,
            do_calib=EVAL_DO_CALIB_DIAG,
            do_thr_sweep=EVAL_DO_THR_SWEEP,
            do_cm=EVAL_DO_CM_REPORT,
            do_per_enzyme=EVAL_DO_PER_ENZYME,
            ece_bins=EVAL_ECE_BINS,
            sweep_n_t=EVAL_SWEEP_N_T,
        )

    # ROC / PR plots (threshold-free)
    if len(np.unique(y)) > 1:
        auroc = headline["auroc_weighted"]
        ap    = headline["ap_weighted"]
        pos_rate = headline["pos_rate_weighted"]

        fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
        plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"{split_name} ROC (weighted)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / f"{split_name}_roc.png", dpi=160)
        plt.close()

        prec, rec, _ = precision_recall_curve(y, p, sample_weight=w)
        plt.figure()
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.hlines(pos_rate, 0, 1, linestyles="--", linewidth=1, label=f"Chance={pos_rate:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"{split_name} PR (weighted)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / f"{split_name}_pr.png", dpi=160)
        plt.close()

    # Optional: thresholded confusion matrices for every reported threshold
    thr_dict = _as_threshold_dict(thresholds)
    if thr_dict is not None:
        for name, t in thr_dict.items():
            rep = _threshold_report(y, p, w, t)

            disp = ConfusionMatrixDisplay(rep["cm_w"], display_labels=["neg","pos"])
            disp.plot(values_format=".1f")
            plt.title(f"{split_name} CM (weighted) @ {name}={t:.3f}")
            plt.tight_layout()
            plt.savefig(plots / f"{split_name}_cm_counts__{name}.png", dpi=160)
            plt.close()

            cm_norm_true = confusion_matrix(y, rep["yhat"], labels=[0,1], sample_weight=w, normalize="true")
            disp = ConfusionMatrixDisplay(cm_norm_true, display_labels=["neg","pos"])
            disp.plot(values_format=".2f")
            plt.title(f"{split_name} CM norm(true) @ {name}={t:.3f}")
            plt.tight_layout()
            plt.savefig(plots / f"{split_name}_cm_norm_true__{name}.png", dpi=160)
            plt.close()

    return headline

def _bundle_smoke_check(eval_dir: Path):
    """Verify that expected evaluation artifacts exist in ``eval_dir``.

    A lightweight sanity check used by the notebook to confirm that
    previous evaluation runs emitted all required files.  The list of
    required files depends on whether both classes are present in the
    labels and on various global flags controlling calibration diagnostics
    and threshold sweeps.

    Parameters
    ----------
    eval_dir : Path
        Directory that should contain canonical evaluation artifacts.

    Returns
    -------
    tuple[bool, list[str]]
        ``(ok, missing)`` where ``ok`` is ``True`` if no files are
        missing and ``missing`` is a list of filenames that were not
        found.
    """
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        print(f"[bundle check] {eval_dir} | ok=False | missing=DIR_NOT_FOUND")
        return False, ["DIR_NOT_FOUND"]

    # always expected if EVAL_WRITE_CANONICAL=True
    need = ["preds.csv", "y_true.npy", "y_score.npy", "w_test.npy", "headline.json", "score_hist.png"]

    # Determine if ROC/PR should exist (only if both classes present)
    y_fp = eval_dir / "y_true.npy"
    two_class = None
    if y_fp.exists():
        try:
            y = np.load(y_fp)
            two_class = (len(np.unique(y)) > 1)
        except Exception:
            two_class = None

    if two_class is True:
        need += ["roc_curve.png", "pr_curve.png"]

    if globals().get("EVAL_DO_CALIB_DIAG", False):
        need += ["reliability_curve.png", "pr_calibration_summary.json"]

    if globals().get("EVAL_DO_CM_REPORT", False):
        need += ["confusion_matrix_report.json"]

    if globals().get("EVAL_DO_THR_SWEEP", False):
        need += ["threshold_sweep_weighted.csv"]

    missing = [f for f in need if not (eval_dir / f).exists()]
    ok = (len(missing) == 0)
    print(f"[bundle check] {eval_dir} | ok={ok} | missing={missing}")
    return ok, missing

def _quiet_bundle_ok(eval_dir: Path):
    import contextlib, io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        ok, missing = _bundle_smoke_check(eval_dir)
    return ok, missing

def _read_headline_json(eval_dir: Path) -> dict | None:
    fp = Path(eval_dir) / "headline.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text())
        except Exception:
            return None
    return None

def _fmt_track_line(headline: dict, prefer: str = "t_oof_f1"):
    """
    Formats: AUROC=... | AP=... | t=... | F1=...
    Assumes headline["thresholded"] exists.
    """
    auroc = headline.get("auroc_weighted", float("nan"))
    ap    = headline.get("ap_weighted", float("nan"))

    t = float("nan")
    f1 = float("nan")

    thr = headline.get("thresholded")
    if isinstance(thr, dict) and len(thr) > 0:
        if prefer in thr:
            entry = thr[prefer]
        elif "t0p5" in thr:
            entry = thr["t0p5"]
        else:
            entry = next(iter(thr.values()))
        t  = entry.get("threshold", float("nan"))
        f1 = entry.get("rates", {}).get("f1", float("nan"))

    return f"AUROC={auroc:.3f} | AP={ap:.3f} | t={t:.3f} | F1={f1:.3f}"

def _cached_eval_dir(run_dir: Path, split_name: str) -> Path:
    return Path(run_dir) / f"trackB/{split_name}/test"

def _load_cached_headline(run_dir: Path, split_name: str) -> Optional[dict]:
    """
    Preference order:
      1) canonical headline.json
      2) legacy <split>__test_headline.json
      3) run-root manifest.json with inline headline (if present)
    """
    ed = _cached_eval_dir(run_dir, split_name)

    fp = ed / "headline.json"
    if fp.exists():
        return json.loads(fp.read_text())

    fp = ed / f"{split_name}__test_headline.json"
    if fp.exists():
        return json.loads(fp.read_text())

    man_fp = Path(run_dir) / "manifest.json"
    if man_fp.exists():
        man = json.loads(man_fp.read_text())
        if isinstance(man, dict) and isinstance(man.get("headline"), dict):
            return man["headline"]

    return None

def _print_perf_line(split_name: str, headline: Optional[dict]):
    if not isinstance(headline, dict):
        print(f"[{split_name}] headline not found — expected: "
              f"{_cached_eval_dir(Path('.'), split_name)}/headline.json or {split_name}__test_headline.json or manifest.json")
        return

    print(f"[{split_name}] AUROC={headline.get('auroc_weighted', float('nan')):.3f} | "
          f"AP={headline.get('ap_weighted', float('nan')):.3f} | "
          f"Brier={headline.get('brier_weighted', float('nan')):.3f} | "
          f"LogLoss={headline.get('log_loss_weighted', float('nan')):.3f}")

def _trackB_eval_complete(run_dir: Path, split_name: str) -> tuple[bool, list[str]]:
    ed = _cached_eval_dir(run_dir, split_name)
    with contextlib.redirect_stdout(_io.StringIO()):
        ok, missing = _bundle_smoke_check(ed)
    return bool(ok), list(missing)

def _repair_eval_bundle_from_preds(eval_dir: Path) -> bool:
    """
    Best-effort regeneration of missing canonical/postprocess artifacts from preds.csv.
    Does NOT retrain.
    """
    eval_dir = Path(eval_dir)
    preds_fp = eval_dir / "preds.csv"
    if not preds_fp.exists():
        return False

    try:
        df = pd.read_csv(preds_fp)
    except Exception:
        return False

    if "prob_raw" not in df.columns:
        return False

    # ensure arrays
    y_true_fp = eval_dir / "y_true.npy"
    y_score_fp = eval_dir / "y_score.npy"
    w_fp = eval_dir / "w_test.npy"
    headline_fp = eval_dir / "headline.json"

    try:
        p = df["prob_raw"].to_numpy(dtype=float)
        np.save(y_score_fp, p.astype(float))
    except Exception:
        pass

    if "y_true" in df.columns:
        try:
            y = df["y_true"].to_numpy(dtype=int)
            np.save(y_true_fp, y.astype(int))
        except Exception:
            pass
    if "weight" in df.columns:
        try:
            w = df["weight"].to_numpy(dtype=float)
            np.save(w_fp, w.astype(float))
        except Exception:
            pass
    else:
        try:
            np.save(w_fp, np.ones(len(df), dtype=float))
        except Exception:
            pass

    # headline.json if missing
    if (not headline_fp.exists()) and ("y_true" in df.columns):
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
            y = df["y_true"].to_numpy(dtype=int)
            w = df["weight"].to_numpy(dtype=float) if "weight" in df.columns else np.ones(len(df), dtype=float)
            eps = 1e-15
            p_clip = np.clip(p, eps, 1 - eps)
            auroc = float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else float("nan")
            ap = float(average_precision_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else float("nan")
            ll = float(log_loss(y, p_clip, sample_weight=w, labels=[0, 1]))
            brier = float(brier_score_loss(y, p, sample_weight=w))
            pos_rate = float(np.average(y, weights=w))
            headline = dict(
                split=str(eval_dir),
                n=int(len(y)),
                weight_sum=float(np.sum(w)),
                pos_rate_weighted=float(pos_rate),
                auroc_weighted=float(auroc),
                ap_weighted=float(ap),
                log_loss_weighted=float(ll),
                brier_weighted=float(brier),
                thresholded=None,
            )
            headline_fp.write_text(json.dumps(headline, indent=2))
        except Exception:
            pass

    # Try to regenerate postprocess artifacts if available
    if "postprocess_eval_dir" in globals():
        try:
            postprocess_eval_dir = globals()["postprocess_eval_dir"]
            postprocess_eval_dir(
                eval_dir,
                t_report=globals().get("EVAL_T_REPORT", 0.5),
                do_calib=globals().get("EVAL_DO_CALIB_DIAG", True),
                do_thr_sweep=globals().get("EVAL_DO_THR_SWEEP", True),
                do_cm=globals().get("EVAL_DO_CM_REPORT", True),
                do_per_enzyme=globals().get("EVAL_DO_PER_ENZYME", True),
                ece_bins=globals().get("EVAL_ECE_BINS", 10),
                sweep_n_t=globals().get("EVAL_SWEEP_N_T", 401),
            )
        except Exception:
            pass

    # Return True; caller re-checks completeness via _trackB_eval_complete
    return True

def _print_header_from_manifest_or_fallback(*, split_name: str, run_dir: Path, split_json_fp: Path):
    man = _read_manifest(run_dir)
    pairs_fp_name = None
    if man.get("pairs_fp_name"):
        pairs_fp_name = str(man["pairs_fp_name"])
    elif man.get("pairs_fp"):
        pairs_fp_name = Path(man["pairs_fp"]).name

    train_n = man.get("train_n", None)
    test_n = man.get("test_n", None)

    # fallback counts
    if test_n is None:
        preds_fp = _cached_eval_dir(run_dir, split_name) / "preds.csv"
        if preds_fp.exists():
            try:
                test_n = int(len(pd.read_csv(preds_fp)))
            except Exception:
                test_n = None

    if train_n is None or test_n is None:
        trc, tec = _infer_train_test_counts_from_split_json(split_json_fp)
        if train_n is None:
            train_n = trc
        if test_n is None:
            test_n = tec

    pairs_part = pairs_fp_name if pairs_fp_name else "pairs_<unknown>"

    def _fmt_n(x):
        return f"{int(x):,}" if isinstance(x, (int, np.integer)) else ("?" if x is None else str(x))

    print(f"\n[{split_name}] pairs={pairs_part} | train={_fmt_n(train_n)} | test={_fmt_n(test_n)} | HPO=frozen(enzyme-OOD)")

def _ext_labels_present_from_dir(out_dir: Path) -> bool:
    preds_fp = Path(out_dir) / "preds.csv"
    if not preds_fp.exists():
        return False
    try:
        df = pd.read_csv(preds_fp, nrows=5)
        return "y_true" in df.columns
    except Exception:
        return False

def _trackB_external_outdir_complete(out_dir: Path) -> tuple[bool, list[str]]:
    out_dir = Path(out_dir)
    missing = []

    for f in ["preds.csv", "y_score.npy", "bundle.json", "summary.json"]:
        if not (out_dir / f).exists():
            missing.append(f)

    has_labels = _ext_labels_present_from_dir(out_dir)
    if has_labels and not (out_dir / "eval.json").exists():
        missing.append("eval.json")

    return len(missing) == 0, missing

def _trackB_external_run_complete(
    run_dir: Path,
    *,
    universe_tags: list[str],
    ext_tags: list[str],
    ext_tags_by_universe: dict[str, list[str]],
) -> tuple[bool, list[str]]:
    run_dir = Path(run_dir)
    missing = []

    if not (run_dir / "run_manifest.json").exists():
        missing.append("run_manifest.json")

    track_root = run_dir / "trackB_external"
    if not track_root.exists():
        missing.append("trackB_external/")
        return False, missing

    for U in universe_tags:
        ext_tags_this = ext_tags_by_universe.get(U, ext_tags)
        for ext in ext_tags_this:
            out_dir = track_root / U / f"ext_{ext}"
            if not out_dir.exists():
                missing.append(str(out_dir.relative_to(run_dir)))
                continue
            ok, miss = _trackB_external_outdir_complete(out_dir)
            if not ok:
                missing.extend([f"{out_dir.relative_to(run_dir)}/{m}" for m in miss])

    return len(missing) == 0, missing

def _write_pred_eval_bundle(*, out_dir: Path, context: dict, df: pd.DataFrame, p: np.ndarray,
                           y: np.ndarray | None, w: np.ndarray | None,
                           thresholded: dict | None, threshold_policy: str,
                           make_plots: bool = True):
    """
    Writes:
      preds.csv, y_score.npy, eval.json, bundle.json, summary.json, plots/*
    Returns evald dict (or None if labels missing).
    """
    out_dir = _ensure_dir(out_dir)
    plots = _ensure_dir(out_dir / "plots")

    meta_cols = [c for c in ["pair_id","enzyme","enz_idx","sub_idx","source","organism"] if c in df.columns]
    preds = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=np.arange(len(df)))
    preds["prob_raw"] = np.asarray(p, float)

    if y is not None:
        preds["y_true"] = np.asarray(y, int)
    if w is not None:
        preds["weight"] = np.asarray(w, float)
    else:
        preds["weight"] = 1.0

    preds.to_csv(out_dir / "preds.csv", index=False)
    np.save(out_dir / "y_score.npy", np.asarray(p, float))

    if y is None:
        (out_dir / "bundle.json").write_text(json.dumps({"context": context, "eval": None}, indent=2))
        (out_dir / "summary.json").write_text(json.dumps({"context": context, "eval": None}, indent=2))
        return None

    y = np.asarray(y, int)
    w = preds["weight"].to_numpy(dtype=float)
    p = np.asarray(p, float)

    auroc = float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else float("nan")
    ap    = float(average_precision_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else float("nan")
    brier = float(brier_score_loss(y, p, sample_weight=w))
    pos_rate = float(np.average(y, weights=w))
    eps = 1e-15
    ll = float(log_loss(y, np.clip(p, eps, 1-eps), sample_weight=w))

    evald = dict(
        split=context.get("split", ""),
        n=int(len(y)),
        weight_sum=float(np.sum(w)),
        pos_rate_weighted=float(pos_rate),
        auroc_weighted=float(auroc),
        ap_weighted=float(ap),
        log_loss_weighted=float(ll),
        brier_weighted=float(brier),
        thresholded=thresholded,
    )
    (out_dir / "eval.json").write_text(json.dumps(evald, indent=2))

    ctx2 = dict(context)
    ctx2["threshold_policy"] = threshold_policy
    (out_dir / "bundle.json").write_text(json.dumps({"context": ctx2, "eval": evald}, indent=2))
    (out_dir / "summary.json").write_text(json.dumps({"context": ctx2, "eval": evald}, indent=2))

    # plots
    if make_plots and len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
        plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"{context.get('title','ROC')}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / "roc.png", dpi=160)
        plt.close()

        prec, rec, _ = precision_recall_curve(y, p, sample_weight=w)
        plt.figure()
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.hlines(pos_rate, 0, 1, linestyles="--", linewidth=1, label=f"Chance={pos_rate:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"{context.get('title','PR')}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / "pr.png", dpi=160)
        plt.close()

    if make_plots and REPORT_BINARY_METRICS and isinstance(thresholded, dict) and thresholded:
        for name, entry in thresholded.items():
            t = float(entry.get("threshold", np.nan))
            if not np.isfinite(t):
                continue
            rep = _threshold_report(y, p, w, t)

            disp = ConfusionMatrixDisplay(rep["cm_w"], display_labels=["neg","pos"])
            disp.plot(values_format=".1f")
            plt.title(f"{context.get('title','CM')} @ {name}={t:.3f}")
            plt.tight_layout()
            plt.savefig(plots / f"cm_counts__{name}.png", dpi=160)
            plt.close()

            cm_norm_true = confusion_matrix(y, rep["yhat"], labels=[0,1], sample_weight=w, normalize="true")
            disp = ConfusionMatrixDisplay(cm_norm_true, display_labels=["neg","pos"])
            disp.plot(values_format=".2f")
            plt.title(f"{context.get('title','CM norm(true)')} @ {name}={t:.3f}")
            plt.tight_layout()
            plt.savefig(plots / f"cm_norm_true__{name}.png", dpi=160)
            plt.close()

    return evald

def _smoke_check_ext_dir(out_dir: Path, has_labels: bool) -> tuple[bool, list[str]]:
    out_dir = Path(out_dir)
    need = ["preds.csv", "y_score.npy", "bundle.json", "summary.json"]
    if has_labels:
        need.append("eval.json")
    missing = [f for f in need if not (out_dir / f).exists()]

    # Optional: if labeled + two-class, expect ROC/PR plots (roc.png/pr.png)
    if has_labels and (out_dir / "preds.csv").exists():
        try:
            dfp = pd.read_csv(out_dir / "preds.csv")
            if "y_true" in dfp.columns and dfp["y_true"].nunique(dropna=True) > 1:
                for f in ["plots/roc.png", "plots/pr.png"]:
                    if not (out_dir / f).exists():
                        missing.append(f)
        except Exception:
            # non-fatal; only used for optional plot checks
            pass

    ok = (len(missing) == 0)
    return ok, missing

def vae_main_eval_prefix(*, run_root_tag: str, split_name: str) -> str:
    run_root_tag = str(run_root_tag).strip()
    split_name = str(split_name).strip()
    if run_root_tag == "trackA":
        return "trackA_internal/test"
    if run_root_tag == "trackB":
        return f"trackB/{split_name}/test"
    raise ValueError("run_root_tag must be 'trackA' or 'trackB'")

def vae_is_complete_run_dir(run_dir: Path, *, run_root_tag: str, split_name: str) -> bool:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return False
    pref = vae_main_eval_prefix(run_root_tag=run_root_tag, split_name=split_name)
    required = [
        run_dir / "run_manifest.json",
        run_dir / "model" / "model.pt",
        run_dir / "model" / "scaler.npz",
        run_dir / pref / "headline.json",
        run_dir / pref / "preds.csv",
    ]
    return all(p.exists() for p in required)

def _vae_headline_get_ap(headline_fp: Path) -> float | None:
    headline_fp = Path(headline_fp)
    if not headline_fp.exists():
        return None
    try:
        h = json.loads(headline_fp.read_text())
    except Exception:
        return None
    for k in ["ap_weighted", "ap", "average_precision", "avg_precision"]:
        v = h.get(k, None)
        if isinstance(v, (int, float)) and np.isfinite(v):
            return float(v)
    m = h.get("metrics", None)
    if isinstance(m, dict):
        for k in ["ap_weighted", "ap", "average_precision", "avg_precision"]:
            v = m.get(k, None)
            if isinstance(v, (int, float)) and np.isfinite(v):
                return float(v)
    return None

def vae_ext_dir_complete(out_dir: Path) -> bool:
    """
    Output completeness for a single evaluation directory.
    Accept:
      - preds.csv + y_score.npy + headline.json
    Also accept baseline bundles that include preds.csv + headline.json plus at least one eval marker.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return False

    preds_fp = out_dir / "preds.csv"
    head_fp  = out_dir / "headline.json"
    ysc_fp   = out_dir / "y_score.npy"

    if preds_fp.exists() and head_fp.exists() and ysc_fp.exists():
        return True

    if preds_fp.exists() and head_fp.exists():
        for extra in ["eval.json", "metrics.json", "y_true.npy", "confusion_matrix.png"]:
            if (out_dir / extra).exists():
                return True

    return False

def _vae_parquet_columns(fp: Path) -> list[str] | None:
    fp = Path(fp)
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(fp).schema.names)
    except Exception:
        return None

def vae_ext_is_complete(
    run_dir: Path,
    *,
    universe_tags,
    ext_tags,
    ext_tags_by_universe,
    require_overlap_audit: bool = True,
    require_sanity: bool = True,
    sanity_do_ablations: bool = True,
    sanity_do_permutations: bool = True,
    sanity_do_seen_unseen_2x2: bool = True,
    sanity_permute_substrate: bool = True,
) -> bool:
    """
    Deep-ish completeness check for the external VAE run directory.

    Tolerant behavior:
      - Missing ext parquet => that ext is not required.
      - Label-less ext => do not require eval bundles; require preds.csv+y_score.npy+headline.json.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return False

    if not (run_dir / "run_manifest.json").exists():
        return False

    ext_tags_by_universe = ext_tags_by_universe or {}

    # Require summary CSV iff at least one labeled ext parquet exists among intended sets
    any_labeled_ext = False
    for U in list(universe_tags):
        ext_list = ext_tags_by_universe.get(U, list(ext_tags))
        for ext in ext_list:
            ext_fp = Path(PROJ) / "data" / f"ext_{ext}.parquet"
            if not ext_fp.exists():
                continue
            cols = _vae_parquet_columns(ext_fp)
            if cols is None:
                any_labeled_ext = True
                break
            if any(c in cols for c in ["reaction","y","label","class"]):
                any_labeled_ext = True
                break
        if any_labeled_ext:
            break
    if any_labeled_ext and not (run_dir / "trackB_external_summary.csv").exists():
        return False

    for U in list(universe_tags):
        U_root = run_dir / "trackB_external" / str(U)

        # Full model bundle marker
        if not ((U_root / "model" / "model.pt").exists() and (U_root / "model" / "scaler.npz").exists()):
            return False

        # Ablation model bundles (if required)
        if require_sanity and sanity_do_ablations:
            for mode in ["enzyme_only", "substrate_only"]:
                mroot = U_root / "model_ablations" / mode
                if not ((mroot / "model" / "model.pt").exists() and (mroot / "model" / "scaler.npz").exists()):
                    return False

        ext_list = ext_tags_by_universe.get(U, list(ext_tags))
        for ext in ext_list:
            ext_fp = Path(PROJ) / "data" / f"ext_{ext}.parquet"
            if not ext_fp.exists():
                continue

            out_dir = run_dir / "trackB_external" / str(U) / f"ext_{ext}"
            if not out_dir.exists():
                return False

            if require_overlap_audit and not (out_dir / "overlap_audit.json").exists():
                return False

            cols = _vae_parquet_columns(ext_fp) or []
            labels_present = any(c in cols for c in ["reaction","y","label","class"])

            if labels_present:
                if not vae_ext_dir_complete(out_dir):
                    return False

                if require_sanity:
                    # ablations
                    if sanity_do_ablations:
                        for mode in ["enzyme_only", "substrate_only"]:
                            d = out_dir / "sanity" / "ablations" / mode
                            if not vae_ext_dir_complete(d):
                                return False

                    # permutations
                    if sanity_do_permutations:
                        d = out_dir / "sanity" / "permute_enz"
                        if not vae_ext_dir_complete(d):
                            return False
                        if sanity_permute_substrate:
                            d = out_dir / "sanity" / "permute_sub"
                            if not vae_ext_dir_complete(d):
                                return False

                    # 2×2
                    if sanity_do_seen_unseen_2x2:
                        cfp = out_dir / "sanity" / "seen_unseen_2x2_counts.json"
                        if not cfp.exists():
                            return False
                        try:
                            counts = json.loads(cfp.read_text())
                        except Exception:
                            return False
                        for q in ["E_seen__S_seen","E_seen__S_unseen","E_unseen__S_seen","E_unseen__S_unseen"]:
                            n = int(counts.get(q, 0) or 0)
                            if n <= 0:
                                continue
                            qdir = out_dir / "sanity" / "seen_unseen_2x2" / q
                            if not vae_ext_dir_complete(qdir):
                                return False
            else:
                # label-less: require main preds-only + (optionally) 2×2 counts
                if not ((out_dir / "preds.csv").exists() and (out_dir / "y_score.npy").exists() and (out_dir / "headline.json").exists()):
                    return False
                try:
                    h = json.loads((out_dir / "headline.json").read_text())
                    if h.get("labels_present", False) is not False:
                        return False
                except Exception:
                    return False

                if require_sanity and sanity_do_seen_unseen_2x2:
                    cfp = out_dir / "sanity" / "seen_unseen_2x2_counts.json"
                    if not cfp.exists():
                        return False

    return True

__all__ = [
    'weighted_ece',
    '_cm_rates_from_weighted_counts',
    '_as_threshold_dict',
    '_threshold_report',
    'postprocess_eval_dir',
    '_eval_headline',
    '_eval_and_write',
    '_bundle_smoke_check',
    '_quiet_bundle_ok',
    '_read_headline_json',
    '_fmt_track_line',
    '_cached_eval_dir',
    '_load_cached_headline',
    '_print_perf_line',
    '_trackB_eval_complete',
    '_repair_eval_bundle_from_preds',
    '_print_header_from_manifest_or_fallback',
    '_ext_labels_present_from_dir',
    '_trackB_external_outdir_complete',
    '_trackB_external_run_complete',
    '_write_pred_eval_bundle',
    '_smoke_check_ext_dir',
    'vae_main_eval_prefix',
    'vae_is_complete_run_dir',
    '_vae_headline_get_ap',
    'vae_ext_dir_complete',
    '_vae_parquet_columns',
    'vae_ext_is_complete',
]
