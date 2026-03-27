from __future__ import annotations

import os
from pathlib import Path

"""
nb_contracts.py
===============

Path contracts for the FRAPPUCCINO GitHub/Colab workflow.

Source-controlled inputs live under ``REPO_ROOT/data``.
Notebook-generated artifacts live under ``PROJ_ROOT`` which defaults to
``REPO_ROOT/results``.

"""

# ---------------------------------------------------------------------------
# Canonical repo and project roots
# ---------------------------------------------------------------------------

REPO_NAME: str = "FRAPPUCCINO"

# Git checkout root. The notebook bootstrap cell should set
# FRAPPUCCINO_REPO_ROOT explicitly, but we keep a sensible Colab default.
REPO_ROOT: Path = Path(
    os.environ.get("FRAPPUCCINO_REPO_ROOT", f"/content/{REPO_NAME}")
)

# Working/output root for notebook-generated artifacts.
# Defaults to <repo>/results and should match the notebook's PROJ variable.
PROJ_ROOT: Path = Path(
    os.environ.get("FRAPPUCCINO_PROJ", str(REPO_ROOT / "results"))
)

# Compatibility alias retained for older code paths that may still import it.
PROJ_NAME: str = REPO_NAME

# Legacy compatibility placeholder. Google Drive is not part of the default
# GitHub/Colab workflow anymore. Code should not rely on this value.
MOUNT_POINT: str | None = None

# Canonical first-level subdirectories under PROJ_ROOT.
SUBDIRS: list[str] = [
    "data",
    "features",
    "splits",
    "models",
    "metrics",
    "logs",
    "reports",
]


def get_subdir(proj: Path, subdir: str) -> Path:
    """Return ``proj / subdir`` as a ``Path`` instance."""
    return Path(proj) / subdir


def ensure_subdirs(proj: Path, subdirs: list[str] | None = None) -> None:
    """Ensure that the canonical subdirectories exist under ``proj``."""
    for name in (subdirs or SUBDIRS):
        (Path(proj) / name).mkdir(parents=True, exist_ok=True)


__all__ = [
    "REPO_NAME",
    "REPO_ROOT",
    "PROJ_ROOT",
    "PROJ_NAME",
    "MOUNT_POINT",
    "SUBDIRS",
    "get_subdir",
    "ensure_subdirs",
]
