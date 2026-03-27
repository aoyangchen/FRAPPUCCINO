"""
nb_drive_io.py
================

This module encapsulates Google Drive mounting and download utilities used
in the GT1 Colab workflow.  Separating this logic into its own module
allows the main notebook to import high‑level helpers without embedding
implementation details inline.  All imports of ``google.colab.drive`` and
``gdown`` occur inside functions to avoid side effects at import time
and to permit parsing outside of a Colab environment.

Only the minimal functionality required by Phase 1 of the refactor is
provided here.  Future phases may extend this module, but no runtime
behavior from the original notebook should be altered.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

# Acquire a module‑level logger.  This mirrors the original notebook's
# usage of a global ``logger`` instance for progress reporting.  If the
# notebook configures logging elsewhere, messages from this module will
# integrate seamlessly.
logger = logging.getLogger(__name__)

def mount_drive_and_get_proj(use_drive: bool, proj_name: str, mount_point: str = "/content/drive") -> Path:
    """Mount Google Drive (if requested) and return the project root.

    This helper replicates the logic from the original notebook's
    initialization cell.  When ``use_drive`` is ``True``, the Google
    Drive is mounted at ``mount_point`` and the project root is constructed
    as ``Path(mount_point) / 'MyDrive' / proj_name``.  When ``use_drive``
    is ``False``, the project root falls back to ``Path('/content') /
    proj_name``.  The caller remains responsible for creating any
    subdirectories under the project root.

    Parameters
    ----------
    use_drive : bool
        Whether to mount Google Drive.  When ``False``, no mounting is
        attempted.
    proj_name : str
        Name of the project directory (e.g., ``'esi_baseline_gbdt'``).
    mount_point : str, optional
        Filesystem location where Drive will be mounted.  Defaults to
        ``'/content/drive'``.

    Returns
    -------
    Path
        The resolved project root path.
    """
    from pathlib import Path  # imported here to avoid polluting globals
    if use_drive:
        # Import within function to avoid errors when not in Colab
        try:
            from google.colab import drive  # type: ignore
        except Exception:
            raise RuntimeError(
                "google.colab.drive is unavailable; set use_drive=False when not running in Colab"
            )
        # Mount the user's Google Drive.  This mirrors the original
        # notebook's use of force_remount=True to avoid stale mounts.
        drive.mount(mount_point, force_remount=True)
        proj = Path(mount_point) / "MyDrive" / proj_name
    else:
        proj = Path("/content") / proj_name
    return proj


def download_from_drive(file_id: str, out_path: Path | str, quiet: bool = True) -> Path:
    """Download a file from Google Drive given its share‑link ID.

    This function wraps ``gdown.download`` with additional safeguards.
    The output path's parent directories are created automatically, and
    progress is logged via the module's logger.  The API surface is kept
    identical to the original notebook to preserve downstream behavior.

    Parameters
    ----------
    file_id : str
        Google Drive file ID extracted from a shareable link.
    out_path : Path | str
        Destination file path.  If given as a string, it is converted
        to a ``Path``.
    quiet : bool, optional
        Whether to suppress gdown's download progress bar.  Defaults to
        ``True``.

    Returns
    -------
    Path
        The resolved ``out_path`` after download.
    """
    from pathlib import Path as _Path  # local alias to avoid confusion
    import gdown  # imported here to ensure availability when used

    p = _Path(out_path)
    # Ensure the destination directory exists
    p.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info(f"Downloading {file_id} → {p}")
    gdown.download(url, str(p), quiet=quiet)
    return p


__all__ = [
    "mount_drive_and_get_proj",
    "download_from_drive",
]