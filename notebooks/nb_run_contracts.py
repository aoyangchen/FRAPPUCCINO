"""
nb_run_contracts.py
===================

Helper routines for managing run directories, manifests, and logging for the
enzyme‑reactivity notebooks.

This module was extracted from the Phase 2 notebook to isolate side‑effectful
logic used when training and evaluating models. It provides utilities to:

* Generate stable configuration dictionaries and hashes for Track A and Track B runs.
* Locate existing run directories that match a configuration.
* Read and write manifests and backfill missing fields.
* Compute SHA‑1 hashes over strings, JSON objects and files.
* Tee standard output to a transcript file while preserving notebook output.

All functions here are side‑effect free at import time; they rely on the caller
to provide notebook‑level globals such as ``PROJ``.  See the notebook for the
expected calling conventions.  No file system mutations occur until the helpers
are invoked explicitly.
"""

from __future__ import annotations

import json
import hashlib
import platform
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch


def _now_tag():
    """Return a timestamp string suitable for run identifiers.

    The format is ``YYYYMMDD_HHMMSS`` in the local time zone.  It is
    used throughout the notebook to generate unique run folders and
    filenames.
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def _ensure_dir(p: Path) -> Path:
    """Create a directory (and any missing parents) if it does not exist.

    Parameters
    ----------
    p : Path or str
        Directory to create.

    Returns
    -------
    Path
        The resolved ``Path`` for ``p``.
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _read_json(fp: Path) -> dict | None:
    """Read a JSON file and return its contents as a dict.

    Invalid or missing files are silently ignored and ``None`` is returned.

    Parameters
    ----------
    fp : Path
        Path to a JSON file.

    Returns
    -------
    dict or None
        Parsed JSON object, or ``None`` if reading failed.
    """
    try:
        return json.loads(Path(fp).read_text())
    except Exception:
        return None

def _stable_json_dumps(obj) -> str:
    """Serialize an object to a JSON string with stable ordering.

    Uses sorted keys and compact separators to ensure that the same
    object produces the same SHA‑1 hash across runs.  Non‑serializable
    objects are converted to strings via ``default=str``.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)

def _sha1_text(s: str) -> str:
    """Return the SHA‑1 digest of a string.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        Hexadecimal SHA‑1 digest.
    """
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _sha1_file(fp: Path) -> str | None:
    """Return the SHA‑1 digest of a file's contents.

    Parameters
    ----------
    fp : Path
        Path to the file to hash.

    Returns
    -------
    str or None
        Hexadecimal digest if the file could be read; otherwise ``None``.
    """
    try:
        data = Path(fp).read_bytes()
    except Exception:
        return None
    return hashlib.sha1(data).hexdigest()

class TeeStdout:
    """Context manager that writes all text sent to ``sys.stdout`` to both the
    notebook output and a file.

    When entered, this class temporarily replaces ``sys.stdout`` with itself.
    Any writes are forwarded to the original ``sys.stdout`` and to the provided
    file handle.  Upon exit, the original ``sys.stdout`` is restored.  Use
    this context manager to capture a transcript of long‑running training or
    evaluation runs without suppressing notebook output.

    Parameters
    ----------
    fp : Path
        Path to the transcript file.  Parent directories are created automatically.
    mode : str, optional
        File open mode.  Defaults to ``"a"``, which appends to the file.
    session_header : str or None, optional
        If provided, this string is written as a header to the file at the start
        of the context.  This header is not emitted to ``sys.stdout``.
    """
    def __init__(self, fp: Path, mode: str = "a", session_header: str | None = None):
        self.fp = Path(fp)
        self.mode = mode
        self.session_header = session_header
        self._orig = None
        self._fh = None

    def __enter__(self):
        self.fp.parent.mkdir(parents=True, exist_ok=True)
        self._orig = sys.stdout
        self._fh = open(self.fp, self.mode, encoding="utf-8")
        if self.session_header:
            # Write header to file only (avoid extra notebook noise)
            try:
                self._fh.write(self.session_header + "\n")
                self._fh.flush()
            except Exception:
                pass
        sys.stdout = self
        return self

    def write(self, s):
        try:
            self._orig.write(s)
        except Exception:
            pass
        try:
            self._fh.write(s)
        except Exception:
            pass

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            self._fh.flush()
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb):
        try:
            sys.stdout = self._orig
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass

def _cfg_for_trackA(*,
    universe_tag: str,
    split_json: Path,
    emb_tag: str,
    emb_fp: Path,
    substrate_kind: str,
    substrate_fp: Path,
    sim_fp_fp: Path,
    hpo_source_universe: str,
    hpo_source_track: str,
    best_params_fp: Path | None,
    frozen_params: dict,
    enable_similarity_bins: bool,
    seed: int,
) -> dict:
    """Assemble a canonical configuration dictionary for Track A runs.

    This helper collects the various file paths, tags, hyperparameters
    and boolean flags required for a Track A experiment into a single
    dict.  It computes a SHA‑1 hash over the frozen parameters to aid
    in manifest comparison and reproducibility.  Many of the values are
    derived from global constants defined in the notebook (e.g.
    ``REPORT_BINARY_METRICS``, ``DEFAULT_THRESHOLD``), so callers should
    ensure that these globals are set appropriately before calling.

    Parameters
    ----------
    universe_tag : str
        Identifier for the training universe (e.g. ``"trainpool"``).
    split_json : Path
        Path to the split JSON describing train/test indices.
    emb_tag : str
        Identifier for the enzyme embedding to use (e.g. ``"esmc_600m"``).
    emb_fp : Path
        Path to the enzyme embedding array.
    substrate_kind : str
        Name of the substrate fingerprint representation.
    substrate_fp : Path
        Path to the substrate fingerprint array.
    sim_fp_fp : Path
        Path to the precomputed enzyme–substrate similarity matrix.
    hpo_source_universe : str
        Universe tag for hyperparameter optimization provenance.
    hpo_source_track : str
        Track identifier for the HPO source.
    best_params_fp : Path or None
        Path to the JSON file containing best hyperparameters; can be
        ``None`` if no prior HPO was performed.
    frozen_params : dict
        Dictionary of hyperparameters that should remain fixed across
        runs.
    enable_similarity_bins : bool
        Whether to enable binning by similarity in evaluation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        A configuration dictionary with a stable ``cfg_hash``.
    """
    flags = dict(
        # reporting / thresholding
        report_binary_metrics=bool(REPORT_BINARY_METRICS),
        default_threshold=float(DEFAULT_THRESHOLD),
        do_oof_threshold=bool(DO_OOF_THRESHOLD),
        n_splits_inner=int(N_SPLITS_INNER),
        seed=int(seed),

        # bundles / postprocess knobs that affect emitted files
        eval_write_canonical=bool(EVAL_WRITE_CANONICAL),
        eval_t_report=float(EVAL_T_REPORT),
        eval_do_calib_diag=bool(EVAL_DO_CALIB_DIAG),
        eval_do_thr_sweep=bool(EVAL_DO_THR_SWEEP),
        eval_do_cm_report=bool(EVAL_DO_CM_REPORT),
        eval_do_per_enzyme=bool(EVAL_DO_PER_ENZYME),
        eval_ece_bins=int(EVAL_ECE_BINS),
        eval_sweep_n_t=int(EVAL_SWEEP_N_T),

        # extras
        do_sub_seen_unseen=bool(DO_SUBSTRATE_SEEN_UNSEEN_BREAKDOWN),
        do_sanity_checks=bool(DO_SANITY_CHECKS),
        do_sanity_ablations=bool(DO_SANITY_ABLATIONS),
        do_sanity_permute=bool(DO_SANITY_PERMUTE_TEST),
        sanity_ablation_n_estimators_cap=int(SANITY_ABLATION_N_ESTIMATORS_CAP) if SANITY_ABLATION_N_ESTIMATORS_CAP is not None else None,
        enable_similarity_bins=bool(enable_similarity_bins),
    )

    split_sig = _sha1_file(split_json)

    cfg = dict(
        track="A_internal_enzyme_OOD",
        universe_tag=str(universe_tag),
        split_json=str(Path(split_json)),
        split_json_sig=str(split_sig) if split_sig else None,

        emb_tag=str(emb_tag),
        emb_fp=str(Path(emb_fp)),
        substrate_kind=str(substrate_kind),
        substrate_fp=str(Path(substrate_fp)),
        sim_fp_fp=str(Path(sim_fp_fp)),

        hpo_source_universe=str(hpo_source_universe),
        hpo_source_track=str(hpo_source_track),
        best_params_source_fp=str(best_params_fp) if best_params_fp else None,

        frozen_params_hash=_sha1_text(_stable_json_dumps(frozen_params)),
        flags=flags,
    )
    cfg["cfg_hash"] = _sha1_text(_stable_json_dumps(cfg))
    return cfg

def _manifest_matches_cfg(man: dict, cfg: dict) -> bool:
    """Determine whether a manifest corresponds to a given configuration.

    The comparison is conservative: it first checks for an exact
    ``cfg_hash`` match.  If the hash is absent, it compares a subset
    of critical fields (universe tag, embedding path, and split
    signature) to decide whether the existing manifest can be reused.

    Parameters
    ----------
    man : dict
        Manifest dictionary loaded from a previous run.
    cfg : dict
        Configuration dictionary as produced by ``_cfg_for_trackA``.

    Returns
    -------
    bool
        ``True`` if the manifest matches the configuration, else
        ``False``.
    """
    if not isinstance(man, dict):
        return False
    if man.get("track") != "A_internal_enzyme_OOD":
        return False

    # Fast path: cfg_hash present
    if ("cfg_hash" in man) and (man.get("cfg_hash") == cfg.get("cfg_hash")):
        return True

    # Legacy-ish path: compare key fields conservatively
    # Universe
    if str(man.get("universe")) != str(cfg.get("universe_tag")):
        return False

    # Emb tag/fp
    if str(man.get("emb_fp")) != str(cfg.get("emb_fp")):
        return False

    # Split signature or path
    cfg_sig = cfg.get("split_json_sig")
    man_sig = man.get("split_json_sig")
    if cfg_sig and man_sig:
        if str(cfg_sig) != str(man_sig):
            return False
    else:
        if str(man.get("split_json")) != str(cfg.get("split_json")):
            return False

    # Substrate features + sim fp
    man_sub_fp = man.get("fp_fp", man.get("substrate_fp"))
    if str(man_sub_fp) != str(cfg.get("substrate_fp")):
        return False
    if str(man.get("sim_fp_fp")) != str(cfg.get("sim_fp_fp")):
        return False

    # substrate_kind (if present)
    if "substrate_kind" in man:
        if str(man.get("substrate_kind")) != str(cfg.get("substrate_kind")):
            return False

    # HPO provenance (if present)
    if str(man.get("hpo_source_universe", "")) != str(cfg.get("hpo_source_universe", "")):
        return False
    if str(man.get("hpo_source_track", "")) != str(cfg.get("hpo_source_track", "")):
        return False

    # Compare frozen params hash if available
    man_params = man.get("frozen_params")
    if isinstance(man_params, dict):
        man_hash = _sha1_text(_stable_json_dumps(man_params))
        if man_hash != cfg.get("frozen_params_hash"):
            return False

    # Compare key flags that typically exist in legacy manifests
    for k_cfg, k_man in [
        ("report_binary_metrics", "report_binary_metrics"),
        ("do_oof_threshold", "did_oof_threshold"),
        ("do_sub_seen_unseen", "did_sub_seen_unseen_breakdown"),
        ("do_sanity_checks", "did_sanity_checks"),
    ]:
        cfg_val = cfg.get("flags", {}).get(k_cfg)
        man_val = man.get(k_man, None)
        if man_val is None:
            # cannot verify → be conservative
            return False
        if bool(cfg_val) != bool(man_val):
            return False

    return True

def find_existing_trackA_run_dir(*,
    proj: Path,
    cfg: dict,
    policy: str = "latest_mtime",
) -> Path | None:
    runs_root = Path(proj) / "metrics" / "runs"
    if not runs_root.exists():
        return None

    cands = []
    for d in runs_root.glob("trackA__*"):
        if not d.is_dir():
            continue
        mf = d / "run_manifest.json"
        if not mf.exists():
            continue
        try:
            man = json.loads(mf.read_text())
        except Exception:
            continue
        if _manifest_matches_cfg(man, cfg):
            cands.append(d)

    if not cands:
        return None

    if policy == "latest_stamp":
        def _stamp(d):
            mf = d / "run_manifest.json"
            try:
                man = json.loads(mf.read_text())
                return str(man.get("stamp", ""))
            except Exception:
                return ""
        cands = sorted(cands, key=lambda p: _stamp(p), reverse=True)
        return cands[0]

    # default: mtime
    return max(cands, key=lambda p: p.stat().st_mtime)

def _read_manifest(run_dir: Path) -> dict:
    fp = Path(run_dir) / "manifest.json"
    if not fp.exists():
        return {}
    try:
        obj = json.loads(fp.read_text())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _norm_path_str(x) -> Optional[str]:
    if x is None or str(x).strip() == "":
        return None
    try:
        return str(Path(x).resolve())
    except Exception:
        return str(Path(x))

def _trackB_run_ids(*, universe_tag: str, split_name: str, emb_tag: str, substrate_kind: str) -> Tuple[str, str]:
    universe_tag = str(universe_tag).strip()
    split_name = str(split_name).strip()
    emb_tag = str(emb_tag).strip()
    substrate_kind = str(substrate_kind).strip()

    new_id = f"trackB__{universe_tag}__{split_name}__{emb_tag}__{substrate_kind}"
    legacy_id = f"trackB__{universe_tag}__{split_name}__{emb_tag}"
    return new_id, legacy_id

def _resolve_trackB_run_dir(*, proj: Path, universe_tag: str, split_name: str, emb_tag: str, substrate_kind: str) -> Path:
    """
    Strict deterministic resolver.

    For substrate-specific benchmarking, NEVER fall back to the legacy run_id.
    This prevents Morgan / MolEncoder from silently aliasing to the same cached run.
    """
    proj = Path(proj)
    substrate_kind = str(substrate_kind).strip()
    need(len(substrate_kind) > 0, "substrate_kind must be non-empty for Track B internal benchmarking")

    new_id, _legacy_id = _trackB_run_ids(
        universe_tag=universe_tag,
        split_name=split_name,
        emb_tag=emb_tag,
        substrate_kind=substrate_kind,
    )
    return proj / "metrics" / "runs" / new_id

def _trackB_cfg_hash(*,
                     universe_tag: str,
                     split_name: str,
                     split_json_fp: Path,
                     cv_group_col: Optional[str],
                     emb_tag: str,
                     emb_fp: Path,
                     substrate_kind: str,
                     substrate_fp: Path,
                     do_oof_threshold: bool,
                     report_binary_metrics: bool,
                     default_threshold: float,
                     n_splits_inner: int,
                     frozen_params: dict) -> str:
    payload = {
        "universe_tag": str(universe_tag),
        "split_name": str(split_name),
        "split_json_fp": str(Path(split_json_fp)),
        "cv_group_col": cv_group_col,
        "emb_tag": str(emb_tag),
        "emb_fp": str(Path(emb_fp)),
        "substrate_kind": str(substrate_kind),
        "substrate_fp": str(Path(substrate_fp)),
        "do_oof_threshold": bool(do_oof_threshold),
        "report_binary_metrics": bool(report_binary_metrics),
        "default_threshold": float(default_threshold),
        "n_splits_inner": int(n_splits_inner),
        "frozen_params": frozen_params,
    }
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _trackB_manifest_matches_request(
    run_dir: Path,
    *,
    universe_tag: str,
    split_name: str,
    emb_tag: str,
    emb_fp: Path,
    substrate_kind: str,
    substrate_fp: Path,
    cfg_hash: str,
    allow_missing_cfg_hash: bool = True,
) -> Tuple[bool, str]:
    man = _read_manifest(run_dir)
    if not man:
        return False, "missing or unreadable manifest.json"

    expected_run_id = Path(run_dir).name

    checks = {
        "run_id": (str(man.get("run_id", "")), expected_run_id),
        "split_name": (str(man.get("split_name", "")), str(split_name)),
        "emb_tag": (str(man.get("emb_tag", "")), str(emb_tag)),
        "substrate_kind": (str(man.get("substrate_kind", "")), str(substrate_kind)),
        "emb_fp": (_norm_path_str(man.get("emb_fp", None)), _norm_path_str(emb_fp)),
        "substrate_fp": (_norm_path_str(man.get("substrate_fp", None)), _norm_path_str(substrate_fp)),
    }

    # support either "universe" or "universe_tag"
    man_universe = man.get("universe_tag", man.get("universe", ""))
    checks["universe_tag"] = (str(man_universe), str(universe_tag))

    mismatches = []
    for k, (got, want) in checks.items():
        if got != want:
            mismatches.append(f"{k}: got={got!r}, want={want!r}")

    got_cfg_hash = str(man.get("cfg_hash", "")).strip()
    if got_cfg_hash == "":
        if not allow_missing_cfg_hash:
            mismatches.append(f"cfg_hash: got={got_cfg_hash!r}, want={cfg_hash!r}")
    elif got_cfg_hash != str(cfg_hash):
        mismatches.append(f"cfg_hash: got={got_cfg_hash!r}, want={cfg_hash!r}")

    if mismatches:
        return False, "; ".join(mismatches)

    return True, "ok"

def _maybe_backfill_trackB_manifest(run_dir: Path, updates: dict):
    fp = Path(run_dir) / "manifest.json"
    if not fp.exists():
        return
    try:
        obj = json.loads(fp.read_text())
        if not isinstance(obj, dict):
            return
    except Exception:
        return

    changed = False
    for k, v in updates.items():
        if obj.get(k, None) in [None, ""] and v is not None:
            obj[k] = v
            changed = True

    if changed:
        fp.write_text(json.dumps(obj, indent=2))

def _trackB_external_cfg_hash(*,
    emb_tag: str,
    emb_fp: Path,
    substrate_kind: str,
    substrate_fp: Path,
    universe_tags: list[str],
    ext_tags: list[str],
    ext_tags_by_universe: dict[str, list[str]],
    report_binary_metrics: bool,
    default_threshold: float,
    do_train_oof_threshold: bool,
    do_ext_oracle_maxf1: bool,
    do_overlap_audit: bool,
    filter_overlap_from_ext: bool,
    do_trackb_sanity: bool,
    sanity_flags: dict,
) -> str:
    payload = {
        "emb_tag": str(emb_tag),
        "emb_fp": _norm_path_str(emb_fp),
        "substrate_kind": str(substrate_kind),
        "substrate_fp": _norm_path_str(substrate_fp),
        "universe_tags": [str(x) for x in universe_tags],
        "ext_tags": [str(x) for x in ext_tags],
        "ext_tags_by_universe": _normalize_ext_tags_by_universe(ext_tags_by_universe),
        "report_binary_metrics": bool(report_binary_metrics),
        "default_threshold": float(default_threshold),
        "do_train_oof_threshold": bool(do_train_oof_threshold),
        "do_ext_oracle_maxf1": bool(do_ext_oracle_maxf1),
        "do_overlap_audit": bool(do_overlap_audit),
        "filter_overlap_from_ext": bool(filter_overlap_from_ext),
        "do_trackb_sanity": bool(do_trackb_sanity),
        "sanity_flags": sanity_flags,
    }
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _trackB_external_manifest_matches_request(
    run_dir: Path,
    *,
    emb_tag: str,
    emb_fp: Path,
    substrate_kind: str,
    substrate_fp: Path,
    universe_tags: list[str],
    ext_tags: list[str],
    ext_tags_by_universe: dict[str, list[str]],
    report_binary_metrics: bool,
    default_threshold: float,
    do_train_oof_threshold: bool,
    do_ext_oracle_maxf1: bool,
    do_trackb_sanity: bool,
    sanity_flags: dict,
    cfg_hash: str,
    allow_missing_cfg_hash: bool = True,
) -> tuple[bool, str]:
    fp = Path(run_dir) / "run_manifest.json"
    if not fp.exists():
        return False, "missing run_manifest.json"

    try:
        man = json.loads(fp.read_text())
    except Exception as e:
        return False, f"unreadable run_manifest.json: {e!r}"

    checks = {
        "emb_tag": (str(man.get("emb_tag", "")), str(emb_tag)),
        "emb_fp": (_norm_path_str(man.get("emb_fp", None)), _norm_path_str(emb_fp)),
        "substrate_kind": (str(man.get("substrate_kind", "")), str(substrate_kind)),
        "substrate_fp": (_norm_path_str(man.get("substrate_fp", None)), _norm_path_str(substrate_fp)),
        "universes": ([str(x) for x in man.get("universes", [])], [str(x) for x in universe_tags]),
        "ext_tags": ([str(x) for x in man.get("ext_tags", [])], [str(x) for x in ext_tags]),
        "ext_tags_by_universe": (
            _normalize_ext_tags_by_universe(man.get("ext_tags_by_universe", {})),
            _normalize_ext_tags_by_universe(ext_tags_by_universe),
        ),
        "report_binary_metrics": (bool(man.get("report_binary_metrics", False)), bool(report_binary_metrics)),
        "default_threshold": (float(man.get("default_threshold", float("nan"))), float(default_threshold)),
        "did_train_oof_threshold": (bool(man.get("did_train_oof_threshold", False)), bool(do_train_oof_threshold)),
        "did_ext_oracle_maxf1": (bool(man.get("did_ext_oracle_maxf1", False)), bool(do_ext_oracle_maxf1)),
        "did_trackB_sanity": (bool(man.get("did_trackB_sanity", False)), bool(do_trackb_sanity)),
        "sanity_flags": (man.get("sanity_flags", {}), sanity_flags),
    }

    mismatches = []
    for k, (got, want) in checks.items():
        if got != want:
            mismatches.append(f"{k}: got={got!r}, want={want!r}")

    if "did_overlap_audit" in man:
        got = bool(man.get("did_overlap_audit"))
        want = bool(DO_OVERLAP_AUDIT)
        if got != want:
            mismatches.append(f"did_overlap_audit: got={got!r}, want={want!r}")
    if "did_filter_overlap_from_ext" in man:
        got = bool(man.get("did_filter_overlap_from_ext"))
        want = bool(FILTER_OVERLAP_FROM_EXT)
        if got != want:
            mismatches.append(f"did_filter_overlap_from_ext: got={got!r}, want={want!r}")

    got_cfg_hash = str(man.get("cfg_hash", "")).strip()
    if got_cfg_hash == "":
        if not allow_missing_cfg_hash:
            mismatches.append(f"cfg_hash: got={got_cfg_hash!r}, want={cfg_hash!r}")
    elif got_cfg_hash != str(cfg_hash):
        mismatches.append(f"cfg_hash: got={got_cfg_hash!r}, want={cfg_hash!r}")

    if mismatches:
        return False, "; ".join(mismatches)
    return True, "ok"

def _pick_latest_trackB_external_run(
    proj: Path,
    *,
    emb_tag: str,
    emb_fp: Path,
    substrate_kind: str,
    substrate_fp: Path,
    universe_tags: list[str],
    ext_tags: list[str],
    ext_tags_by_universe: dict[str, list[str]],
    report_binary_metrics: bool,
    default_threshold: float,
    do_train_oof_threshold: bool,
    do_ext_oracle_maxf1: bool,
    do_trackb_sanity: bool,
    sanity_flags: dict,
    cfg_hash: str,
) -> Path:
    runs = Path(proj) / "metrics" / "runs"
    if not runs.exists():
        raise FileNotFoundError(f"Missing runs dir: {runs}")

    emb_tag = str(emb_tag).strip()
    substrate_kind = str(substrate_kind).strip()
    pattern = f"trackB__external__{emb_tag}__{substrate_kind}__*"
    cands = [p for p in runs.glob(pattern) if p.is_dir()]

    good = []
    for p in cands:
        ok_manifest, _ = _trackB_external_manifest_matches_request(
            p,
            emb_tag=emb_tag,
            emb_fp=emb_fp,
            substrate_kind=substrate_kind,
            substrate_fp=substrate_fp,
            universe_tags=universe_tags,
            ext_tags=ext_tags,
            ext_tags_by_universe=ext_tags_by_universe,
            report_binary_metrics=report_binary_metrics,
            default_threshold=default_threshold,
            do_train_oof_threshold=do_train_oof_threshold,
            do_ext_oracle_maxf1=do_ext_oracle_maxf1,
            do_trackb_sanity=do_trackb_sanity,
            sanity_flags=sanity_flags,
            cfg_hash=cfg_hash,
            allow_missing_cfg_hash=True,
        )
        if not ok_manifest:
            continue

        ok_complete, _ = _trackB_external_run_complete(
            p,
            universe_tags=universe_tags,
            ext_tags=ext_tags,
            ext_tags_by_universe=ext_tags_by_universe,
        )
        if ok_complete:
            good.append(p)

    if not good:
        raise FileNotFoundError(
            f"No complete compatible TrackB external runs found under {runs} "
            f"for emb_tag={emb_tag}, substrate_kind={substrate_kind}"
        )

    return max(good, key=lambda p: p.stat().st_mtime)

def _resolve_trackB_external_run_dir(
    proj: Path,
    *,
    run_id: str,
    emb_tag: str,
    emb_fp: Path,
    substrate_kind: str,
    substrate_fp: Path,
    universe_tags: list[str],
    ext_tags: list[str],
    ext_tags_by_universe: dict[str, list[str]],
    report_binary_metrics: bool,
    default_threshold: float,
    do_train_oof_threshold: bool,
    do_ext_oracle_maxf1: bool,
    do_trackb_sanity: bool,
    sanity_flags: dict,
    cfg_hash: str,
) -> Path | None:
    rid = str(run_id).strip()

    if rid.lower() == "new":
        return None

    if rid.lower() == "latest":
        return _pick_latest_trackB_external_run(
            proj,
            emb_tag=emb_tag,
            emb_fp=emb_fp,
            substrate_kind=substrate_kind,
            substrate_fp=substrate_fp,
            universe_tags=universe_tags,
            ext_tags=ext_tags,
            ext_tags_by_universe=ext_tags_by_universe,
            report_binary_metrics=report_binary_metrics,
            default_threshold=default_threshold,
            do_train_oof_threshold=do_train_oof_threshold,
            do_ext_oracle_maxf1=do_ext_oracle_maxf1,
            do_trackb_sanity=do_trackb_sanity,
            sanity_flags=sanity_flags,
            cfg_hash=cfg_hash,
        )

    p = Path(proj) / "metrics" / "runs" / rid
    if not p.exists():
        raise FileNotFoundError(f"RUN_ID not found: {p}")

    ok_manifest, msg_manifest = _trackB_external_manifest_matches_request(
        p,
        emb_tag=emb_tag,
        emb_fp=emb_fp,
        substrate_kind=substrate_kind,
        substrate_fp=substrate_fp,
        universe_tags=universe_tags,
        ext_tags=ext_tags,
        ext_tags_by_universe=ext_tags_by_universe,
        report_binary_metrics=report_binary_metrics,
        default_threshold=default_threshold,
        do_train_oof_threshold=do_train_oof_threshold,
        do_ext_oracle_maxf1=do_ext_oracle_maxf1,
        do_trackb_sanity=do_trackb_sanity,
        sanity_flags=sanity_flags,
        cfg_hash=cfg_hash,
        allow_missing_cfg_hash=True,
    )
    if not ok_manifest:
        raise AssertionError(
            f"Explicit RUN_ID does not match current external Track B request:\n"
            f"  run_dir={p}\n"
            f"  reason={msg_manifest}"
        )

    ok_complete, missing = _trackB_external_run_complete(
        p,
        universe_tags=universe_tags,
        ext_tags=ext_tags,
        ext_tags_by_universe=ext_tags_by_universe,
    )
    if not ok_complete:
        raise AssertionError(
            f"Explicit RUN_ID is incomplete for current external Track B request:\n"
            f"  run_dir={p}\n"
            f"  missing={missing}"
        )

    return p

def vae_now_tag():
    return time.strftime("%Y%m%d_%H%M%S")

def vae_cfg_hash(cfg: dict, n: int = 8) -> str:
    s = _stable_json_dumps(cfg).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:n]

def vae_make_run_id(*, run_root_tag: str, universe: str, split_name: str, emb_tag: str, cfg: dict) -> str:
    """
    Track A: timestamped (snapshot-style)
    Track B: deterministic per cfg hash (cache-style)
    """
    run_root_tag = str(run_root_tag).strip()
    universe = str(universe).strip()
    split_name = str(split_name).strip()
    emb_tag = str(emb_tag).strip()
    h = vae_cfg_hash(cfg, n=8)

    if run_root_tag == "trackA":
        return f"trackA__{universe}__{split_name}__{emb_tag}__vae__{vae_now_tag()}__cfg-{h}"
    if run_root_tag == "trackB":
        return f"trackB__{universe}__{split_name}__{emb_tag}__vae__cfg-{h}"
    raise ValueError("run_root_tag must be 'trackA' or 'trackB'")

def find_existing_run_dir_by_cfg_hash(
    *,
    run_root_tag: str,
    universe: str,
    split_name: str,
    emb_tag: str,
    cfg_hash: str,
    proj: Path | None = None,
    policy: str | None = None,
) -> Path | None:
    """
    Scan PROJ/metrics/runs for an existing COMPLETE run matching:
      (run_root_tag, universe, split_name, emb_tag, cfg_hash)

    Patterns (as requested):
      - trackA__{universe}__{split_name}__{emb_tag}__vae__*__cfg-{cfg_hash}
      - trackB__{universe}__{split_name}__{emb_tag}__vae__cfg-{cfg_hash}
    """
    proj = Path(PROJ if proj is None else proj)
    policy = str(VAE_CACHE_POLICY if policy is None else policy)

    run_root_tag = str(run_root_tag).strip()
    universe = str(universe).strip()
    split_name = str(split_name).strip()
    emb_tag = str(emb_tag).strip()
    cfg_hash = str(cfg_hash).strip()

    runs_root = proj / "metrics" / "runs"

    if run_root_tag == "trackA":
        pat = f"trackA__{universe}__{split_name}__{emb_tag}__vae__*__cfg-{cfg_hash}"
    elif run_root_tag == "trackB":
        pat = f"trackB__{universe}__{split_name}__{emb_tag}__vae__cfg-{cfg_hash}"
    else:
        raise ValueError("run_root_tag must be 'trackA' or 'trackB'")

    cands = [p for p in runs_root.glob(pat) if p.is_dir()]
    if not cands:
        return None

    cands = [p for p in cands if vae_is_complete_run_dir(p, run_root_tag=run_root_tag, split_name=split_name)]
    if not cands:
        return None

    if policy == "best_ap":
        scored = []
        for p in cands:
            pref = vae_main_eval_prefix(run_root_tag=run_root_tag, split_name=split_name)
            ap = _vae_headline_get_ap(p / pref / "headline.json")
            ap = float(ap) if ap is not None else float("-inf")
            scored.append((ap, float(p.stat().st_mtime), p))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return scored[0][2]

    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

def vae_ensure_run_dir(run_id: str, *, proj: Path, force: bool) -> Path:
    """
    Create run_dir = PROJ/metrics/runs/<run_id>.
    If exists and populated:
      - force=True  -> delete + recreate
      - force=False -> return existing (caller may treat as cached)
    """
    run_dir = Path(proj) / "metrics" / "runs" / str(run_id)
    if run_dir.exists() and any(run_dir.iterdir()):
        if force:
            import shutil
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def vae_write_cfg(run_dir: Path, cfg: dict) -> str:
    fp = Path(run_dir) / "cfg.json"
    fp.write_text(json.dumps(cfg, indent=2, sort_keys=True))
    return str(fp)

def vae_write_model_bundle(run_dir: Path, *, model, scal: dict, cfg: dict) -> dict:
    mdir = _ensure_dir(Path(run_dir) / "model")
    model_fp = mdir / "model.pt"
    torch.save(model.state_dict(), model_fp)
    scaler_fp = mdir / "scaler.npz"
    np.savez(scaler_fp, **scal)
    (mdir / "model_meta.json").write_text(json.dumps({
        "cfg_hash": vae_cfg_hash(cfg),
        "scal_meta": {"d_enz": int(scal["d_enz"]), "d_fp": int(scal["d_fp"])},
        "stamp": vae_now_tag(),
    }, indent=2))
    return {"model_pt": str(model_fp), "scaler_npz": str(scaler_fp), "model_dir": str(mdir)}

def vae_write_latents(run_dir: Path, *, mu_tr, mu_te, z_tr, z_te) -> dict:
    ldir = _ensure_dir(Path(run_dir) / "latents")
    mu_tr_fp = ldir / "mu_train.npy"
    mu_te_fp = ldir / "mu_test.npy"
    z_tr_fp  = ldir / "z_train.npy"
    z_te_fp  = ldir / "z_test.npy"
    np.save(mu_tr_fp, mu_tr); np.save(mu_te_fp, mu_te)
    np.save(z_tr_fp,  z_tr);  np.save(z_te_fp,  z_te)
    return {
        "latents_dir": str(ldir),
        "mu_train": str(mu_tr_fp), "mu_test": str(mu_te_fp),
        "z_train": str(z_tr_fp),   "z_test": str(z_te_fp),
    }

def vae_write_training_log(run_dir: Path, train_log_df: pd.DataFrame) -> str:
    tdir = _ensure_dir(Path(run_dir) / "training")
    fp = tdir / "training_log.csv"
    train_log_df.to_csv(fp, index=False)
    return str(fp)

def vae_env_fingerprint() -> dict:
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        cuda_version=getattr(torch.version, "cuda", None),
        device_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        cudnn_version=(torch.backends.cudnn.version() if torch.cuda.is_available() else None),
    )

def vae_write_manifest(run_dir: Path, manifest: dict) -> str:
    fp = Path(run_dir) / "run_manifest.json"
    fp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return str(fp)

def vae_make_external_run_id(*, emb_tag: str, cfg: dict, policy: str = "deterministic") -> str:
    """
    Stable run_id for external VAE benchmarking.

    Must use the notebook's cfg hash helper: vae_cfg_hash(cfg, n=8).
    - policy="deterministic": no timestamp -> stable cache key
    - policy="timestamped": include vae_now_tag() -> allow multiple distinct runs
    """
    h = vae_cfg_hash(cfg, n=8)
    policy = str(policy or "deterministic").strip().lower()
    if policy == "deterministic":
        return f"trackB__external__{emb_tag}__vae__cfg-{h}"
    if policy == "timestamped":
        return f"trackB__external__{emb_tag}__vae__{vae_now_tag()}__cfg-{h}"
    raise ValueError(f"Unknown policy={policy!r} (expected 'deterministic' or 'timestamped').")

def vae_transcript_fp(run_dir: Path) -> Path:
    return Path(run_dir) / "console_transcript.txt"

def vae_print_transcript_if_exists(run_dir: Path) -> bool:
    fp = vae_transcript_fp(run_dir)
    if not fp.exists():
        return False
    try:
        txt = fp.read_text()
        sys.stdout.write(txt)
        if txt and (not txt.endswith("\n")):
            sys.stdout.write("\n")
        return True
    except Exception:
        return False

def mmvae_make_run_id(*, run_root_tag: str, universe: str, split_name: str, emb_tag: str, cfg: dict) -> str:
    """
    Track A: timestamped (snapshot-style)
    Track B: deterministic per cfg hash (cache-style)
    """
    run_root_tag = str(run_root_tag).strip()
    universe = str(universe).strip()
    split_name = str(split_name).strip()
    emb_tag = str(emb_tag).strip()
    h = vae_cfg_hash(cfg, n=8)

    if run_root_tag == "trackA":
        return f"trackA__{universe}__{split_name}__{emb_tag}__mmvae__{vae_now_tag()}__cfg-{h}"
    if run_root_tag == "trackB":
        return f"trackB__{universe}__{split_name}__{emb_tag}__mmvae__cfg-{h}"
    raise ValueError("run_root_tag must be 'trackA' or 'trackB'")

def find_existing_mmvae_run_dir_by_cfg_hash(
    *,
    run_root_tag: str,
    universe: str,
    split_name: str,
    emb_tag: str,
    cfg_hash: str,
    proj: Path | None = None,
    policy: str | None = None,
) -> Path | None:
    """
    Scan PROJ/metrics/runs for an existing COMPLETE run matching:
      (run_root_tag, universe, split_name, emb_tag, cfg_hash)

    Patterns (as requested):
      - trackA__{universe}__{split_name}__{emb_tag}__mmvae__*__cfg-{cfg_hash}
      - trackB__{universe}__{split_name}__{emb_tag}__mmvae__cfg-{cfg_hash}
    """
    proj = Path(PROJ if proj is None else proj)
    policy = str(MMVAE_CACHE_POLICY if policy is None else policy)

    run_root_tag = str(run_root_tag).strip()
    universe = str(universe).strip()
    split_name = str(split_name).strip()
    emb_tag = str(emb_tag).strip()
    cfg_hash = str(cfg_hash).strip()

    runs_root = proj / "metrics" / "runs"

    if run_root_tag == "trackA":
        pat = f"trackA__{universe}__{split_name}__{emb_tag}__mmvae__*__cfg-{cfg_hash}"
    elif run_root_tag == "trackB":
        pat = f"trackB__{universe}__{split_name}__{emb_tag}__mmvae__cfg-{cfg_hash}"
    else:
        raise ValueError("run_root_tag must be 'trackA' or 'trackB'")

    cands = [p for p in runs_root.glob(pat) if p.is_dir()]
    if not cands:
        return None

    cands = [p for p in cands if vae_is_complete_run_dir(p, run_root_tag=run_root_tag, split_name=split_name)]
    if not cands:
        return None

    if policy == "best_ap":
        scored = []
        for p in cands:
            pref = vae_main_eval_prefix(run_root_tag=run_root_tag, split_name=split_name)
            ap = _vae_headline_get_ap(p / pref / "headline.json")
            ap = float(ap) if ap is not None else float("-inf")
            scored.append((ap, float(p.stat().st_mtime), p))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return scored[0][2]

    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

def mmvae_make_external_run_id(*, emb_tag: str, cfg: dict, policy: str = "deterministic") -> str:
    """
    Stable run_id for external MMVAE benchmarking.

    Must use the notebook's cfg hash helper: vae_cfg_hash(cfg, n=8).
    - policy="deterministic": no timestamp -> stable cache key
    - policy="timestamped": include vae_now_tag() -> allow multiple distinct runs
    """
    h = vae_cfg_hash(cfg, n=8)
    policy = str(policy or "deterministic").strip().lower()
    if policy == "deterministic":
        return f"trackB__external__{emb_tag}__mmvae__cfg-{h}"
    if policy == "timestamped":
        return f"trackB__external__{emb_tag}__mmvae__{vae_now_tag()}__cfg-{h}"
    raise ValueError(f"Unknown policy={policy!r} (expected 'deterministic' or 'timestamped').")

__all__ = [
    '_now_tag',
    '_ensure_dir',
    '_read_json',
    '_stable_json_dumps',
    '_sha1_text',
    '_sha1_file',
    'TeeStdout',
    '_cfg_for_trackA',
    '_manifest_matches_cfg',
    'find_existing_trackA_run_dir',
    '_read_manifest',
    '_norm_path_str',
    '_trackB_run_ids',
    '_resolve_trackB_run_dir',
    '_trackB_cfg_hash',
    '_trackB_manifest_matches_request',
    '_maybe_backfill_trackB_manifest',
    '_trackB_external_cfg_hash',
    '_trackB_external_manifest_matches_request',
    '_pick_latest_trackB_external_run',
    '_resolve_trackB_external_run_dir',
    'vae_now_tag',
    'vae_cfg_hash',
    'vae_make_run_id',
    'find_existing_run_dir_by_cfg_hash',
    'vae_ensure_run_dir',
    'vae_write_cfg',
    'vae_write_model_bundle',
    'vae_write_latents',
    'vae_write_training_log',
    'vae_env_fingerprint',
    'vae_write_manifest',
    'vae_make_external_run_id',
    'vae_transcript_fp',
    'vae_print_transcript_if_exists',
    'mmvae_make_run_id',
    'find_existing_mmvae_run_dir_by_cfg_hash',
    'mmvae_make_external_run_id',
]
