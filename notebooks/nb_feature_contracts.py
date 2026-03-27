"""
Phase 3 feature/split loader contracts.

This module centralizes loader and path resolver helpers that were
previously defined inside the notebook. The helpers are imported back
into the notebook at runtime and have their globals patched to bind
to the notebook's global state (e.g. PROJ, SPL, EMB_FP, FP_FP). The
helpers here do not perform any side effects at import time and rely
on the caller to manage global variables appropriately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def need(cond: bool, msg: str) -> None:
    """Lightweight assertion used by loader helpers."""
    if not cond:
        raise AssertionError(msg)


def _pick_first_existing(paths: Iterable[Path]) -> Path | None:
    """
    Return the first existing Path from an iterable of candidate paths.

    Parameters
    ----------
    paths : Iterable[Path]
        Candidate file system paths to check.

    Returns
    -------
    Path | None
        The first existing Path, or None if none of the candidates exist.
    """
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def _load_pairs_universe(universe_tag: str) -> pd.DataFrame:
    """
    Load the (enzyme, substrate) pairs dataset for the given universe tag.

    Expects the input parquet at: PROJ/data/pairs_<universe_tag>.parquet.
    Falls back to the double-underscore naming convention if the primary
    file is not found. The returned dataframe includes a ``_pairs_fp``
    attribute recording the source file path as a string.

    Parameters
    ----------
    universe_tag : str
        Name of the universe (e.g. ``trainpool``, ``multiplex``, etc.).

    Returns
    -------
    pd.DataFrame
        Loaded pairs dataframe with a ``_pairs_fp`` attribute.
    """
    universe_tag = str(universe_tag).strip()
    # ``PROJ`` is expected to be injected into globals at runtime by the notebook.
    proj: Path = globals()["PROJ"]  # type: ignore
    fp = proj / "data" / f"pairs_{universe_tag}.parquet"
    if not fp.exists():
        # fallback naming
        fp2 = proj / "data" / f"pairs__{universe_tag}.parquet"
        fp = fp2 if fp2.exists() else fp
    assert fp.exists(), f"Missing pairs parquet for universe={universe_tag}: {fp}"
    df = pd.read_parquet(fp).reset_index(drop=True)
    df.attrs["_pairs_fp"] = str(fp)
    return df


def _load_features() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load enzyme embeddings and substrate fingerprints from global paths.

    ``EMB_FP`` and ``FP_FP`` are expected to be present in the notebook's
    global namespace at call time. The arrays are loaded with ``numpy``
    ``load`` and returned in order (embeddings, fingerprints).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of (embeddings, fingerprints).
    """
    emb_fp: Path = globals()["EMB_FP"]  # type: ignore
    fp_fp: Path = globals()["FP_FP"]  # type: ignore
    assert emb_fp.exists(), f"Missing: {emb_fp}"
    assert fp_fp.exists(), f"Missing: {fp_fp}"
    embs = np.load(emb_fp)
    fps = np.load(fp_fp)
    return embs, fps


def _read_split_obj(split_json_fp: Path) -> Dict[str, Any]:
    """
    Read and parse a JSON split descriptor file.

    Adds a helper field ``_split_json_fp`` capturing the resolved path as
    a string. The ``split_json_fp`` must exist.

    Parameters
    ----------
    split_json_fp : Path
        Path to the JSON file describing the split.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON object with an added ``_split_json_fp`` key.
    """
    split_json_fp = Path(split_json_fp)
    need(split_json_fp.exists(), f"Missing split_json_fp: {split_json_fp}")
    obj: Dict[str, Any] = json.loads(split_json_fp.read_text())
    obj["_split_json_fp"] = str(split_json_fp)
    return obj


def _resolve_train_test_ids_from_split_obj(
    pairs: pd.DataFrame,
    obj: Dict[str, Any],
    split_json_fp: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve train/test indices from a split object.

    Supports multiple conventions:

    1. Explicit indices stored in JSON: ``{"train_ids": [...], "test_ids": [...]}``.
    2. Unified naming scheme: ``train_ids_<stem>.npy`` / ``test_ids_<stem>.npy``.
    3. Enzyme-based schema: ``{"train_enzymes": [...], "test_enzymes": [...]}``.
    4. Group-based schema: ``{"group_col": ..., "train_groups": [...], "test_groups": [...]}``.

    Parameters
    ----------
    pairs : pd.DataFrame
        The pairs dataframe with at least columns referenced by split schema.
    obj : Dict[str, Any]
        Parsed JSON object describing the split.
    split_json_fp : Path
        Path to the JSON file (used to resolve default stem).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays of indices corresponding to train and test samples.
    """
    import numpy as _np  # local alias to avoid name capture in globals

    if "train_ids" in obj and "test_ids" in obj:
        tr = _np.array(obj["train_ids"], dtype=_np.int64)
        te = _np.array(obj["test_ids"], dtype=_np.int64)
        return tr, te

    stem = str(obj.get("split_name", Path(split_json_fp).stem)).strip()
    # SPL is expected in the notebook globals
    spl_root: Path = globals()["SPL"]  # type: ignore
    cand_dirs = [Path(split_json_fp).parent, spl_root]
    for d in cand_dirs:
        tr_fp = d / f"train_ids_{stem}.npy"
        te_fp = d / f"test_ids_{stem}.npy"
        if tr_fp.exists() and te_fp.exists():
            tr = _np.load(tr_fp).astype(_np.int64, copy=False)
            te = _np.load(te_fp).astype(_np.int64, copy=False)
            return tr, te

    if "train_enzymes" in obj and "test_enzymes" in obj:
        need("enzyme" in pairs.columns, "pairs needs 'enzyme' col for enzyme-based split JSON")
        tr_enz = set(map(str, obj["train_enzymes"]))
        te_enz = set(map(str, obj["test_enzymes"]))
        enz = pairs["enzyme"].astype(str)
        tr = _np.where(enz.isin(tr_enz).to_numpy())[0].astype(_np.int64)
        te = _np.where(enz.isin(te_enz).to_numpy())[0].astype(_np.int64)
        return tr, te

    if "group_col" in obj and "train_groups" in obj and "test_groups" in obj:
        gcol = str(obj["group_col"])
        need(gcol in pairs.columns, f"pairs missing group_col='{gcol}' required by split JSON")
        tr_g = set(obj["train_groups"])
        te_g = set(obj["test_groups"])
        g = pairs[gcol]
        tr = _np.where(g.isin(tr_g).to_numpy())[0].astype(_np.int64)
        te = _np.where(g.isin(te_g).to_numpy())[0].astype(_np.int64)
        return tr, te

    raise AssertionError(
        f"Unrecognized split schema AND no train/test .npy found for stem='{stem}'. "
        f"JSON keys={list(obj.keys())} | split_json_fp={split_json_fp}"
    )


def _load_token_file(fp_str: str) -> np.ndarray:
    """
    Load a token file saved under the ``.npz`` format.

    Assumes the file contains a ``tokens`` array under the root key.

    Parameters
    ----------
    fp_str : str
        Path-like string pointing to the ``.npz`` file.

    Returns
    -------
    np.ndarray
        Array of tokens extracted from the file.
    """
    obj = np.load(Path(fp_str), allow_pickle=False)
    return obj["tokens"]


def load_split_ids(split_name: str, *, return_paths: bool = False):
    """
    Load train, test, and drop ID arrays from the ``SPL`` folder.

    This helper maps baseline-compatible names (e.g. ``enzymeOOD80``) to
    their corresponding file stems on disk. It expects that the train and
    test ``.npy`` files exist; the drop file is optional. When ``return_paths``
    is True, a dict of file paths used is returned along with the arrays.

    Parameters
    ----------
    split_name : str
        Logical split name or filename stem.
    return_paths : bool, optional
        When True, return a dict of resolved file paths under the key
        ``paths``. Default is False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]] or
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, str]]
        Arrays of train IDs, test IDs, and drop IDs (if present). When
        ``return_paths`` is True, the dict of file paths is appended.
    """
    name_to_stem = {
        "enzymeOOD80": "trainpool_A1_enzyme80",
        "A0_randomRow": "trainpool_A0_randomRow",
        "A0b_randomEnzCluster80": "trainpool_A0b_randomEnzCluster80",
        "A2_scaffoldOOD": "trainpool_A2_scaffoldOOD",
        "A3_doubleCold_cluster80xscafGroup": "trainpool_A3_doubleCold_cluster80xscafGroup",
    }
    stem = name_to_stem.get(split_name, split_name)
    # ``SPL`` is expected to be available in the notebook globals.
    spl_root: Path = globals()["SPL"]  # type: ignore
    tr_fp = spl_root / f"train_ids_{stem}.npy"
    te_fp = spl_root / f"test_ids_{stem}.npy"
    dr_fp = spl_root / f"drop_ids_{stem}.npy"
    assert tr_fp.exists() and te_fp.exists(), f"Missing split files for {split_name} (stem={stem})"
    train_ids = np.load(tr_fp).astype(np.int64, copy=False)
    test_ids = np.load(te_fp).astype(np.int64, copy=False)
    drop_ids = np.load(dr_fp).astype(np.int64, copy=False) if dr_fp.exists() else None
    if not return_paths:
        return train_ids, test_ids, drop_ids
    paths = {
        "stem": stem,
        "train_ids_fp": str(tr_fp),
        "test_ids_fp": str(te_fp),
        "drop_ids_fp": str(dr_fp) if dr_fp.exists() else None,
    }
    return train_ids, test_ids, drop_ids, paths
