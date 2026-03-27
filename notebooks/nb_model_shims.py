"""
nb_model_shims.py
=================

This module contains model definitions, training loops and convenience
functions that were originally embedded in the Phase 3 GT1 reactivity
notebook.  It supports both the baseline XGBoost models and more advanced
neural architectures (e.g. supervised variational autoencoders).  The code
is factored into helpers so that the notebook can import a clean API
without cluttering the interactive narrative.

None of the helpers execute any training or file system mutations at import
time.  Instead, callers are expected to bind notebook‑level globals (such as
``PROJ``, ``REPORT_BINARY_METRICS`` or ``FP_LEN``) into the imported
functions before invocation, following the binder pattern illustrated in the
notebook.  Do not call these helpers in isolation: they rely on side
variables and on data prepared in earlier notebook sections.

The module deliberately mirrors the original notebook semantics.  Do not
modify class names, function signatures, or control flow without ensuring
that downstream notebooks and evaluation scripts remain compatible.
"""

import os
import json
import time
import math
import shutil
import hashlib
import random
import contextlib
import io
import sys
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, log_loss, brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy
import functools

# Phase 4B: Add missing SupervisedVAE model and related helpers.
# These helpers mirror the original notebook implementation for the single‑tower VAE.
# They are defined here so that importing nb_model_shims provides a complete API
# without requiring notebook injection.

# Choose device at import time. Use CPU if CUDA is unavailable.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    """Seed all underlying randomness sources to make experiments repeatable.

    This helper sets the seed for Python’s ``random`` module, NumPy, and PyTorch,
    and, if CUDA is available, also sets the CUDA random seed.  It does not
    enforce deterministic behavior in cuDNN but enables the benchmark mode for
    potential speedups, mirroring the notebook’s original behavior.

    Parameters
    ----------
    seed : int
        The integer seed to use for all pseudorandom number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Allow CUDA benchmarking for performance; deterministic algorithms are not enforced.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def infer_dims(d_in: int, cfg: dict, mode: str):
    """
    Infer enzyme and fingerprint dimensions from the total input dimension and mode.

    Parameters
    ----------
    d_in : int
        Total input dimension (enzyme + fingerprint).
    cfg : dict
        Configuration dictionary containing FP_LEN.
    mode : str
        One of {"full", "enzyme_only", "substrate_only"}.

    Returns
    -------
    d_enz : int
        Dimension of the enzyme embedding.
    d_fp : int
        Dimension of the fingerprint vector.
    """
    mode = str(mode).strip().lower()
    if mode == "full":
        d_fp = int(cfg["FP_LEN"])
        d_enz = d_in - d_fp
        if d_enz <= 0:
            raise ValueError(f"Bad dims: d_in={d_in}, FP_LEN={d_fp} -> d_enz={d_enz}")
        return d_enz, d_fp
    if mode == "enzyme_only":
        return d_in, 0
    if mode == "substrate_only":
        return 0, d_in
    raise ValueError(mode)

def fit_scaler(X_tr: np.ndarray, d_enz: int):
    """
    Fit a simple mean/std scaler on the enzyme part of the input.

    Parameters
    ----------
    X_tr : np.ndarray
        Training features of shape (n_samples, d_in).
    d_enz : int
        Number of enzyme dimensions.

    Returns
    -------
    mu : np.ndarray
        Mean of the enzyme dimensions (shape (1, d_enz)).
    sd : np.ndarray
        Standard deviation of the enzyme dimensions (shape (1, d_enz)).
    """
    if d_enz > 0:
        mu = X_tr[:, :d_enz].mean(axis=0, keepdims=True).astype(np.float32)
        sd = (X_tr[:, :d_enz].std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    else:
        mu = np.zeros((1, 0), dtype=np.float32)
        sd = np.ones((1, 0), dtype=np.float32)
    return mu, sd

def prep_X(X: np.ndarray, d_enz: int, d_fp: int, enz_mu: np.ndarray, enz_sd: np.ndarray):
    """
    Apply preprocessing to inputs:
      - Standardize enzyme embeddings using provided mean and std.
      - Binarize fingerprint bits (threshold at 0.5).

    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, d_in).
    d_enz : int
        Number of enzyme dimensions.
    d_fp : int
        Number of fingerprint dimensions.
    enz_mu : np.ndarray
        Mean used for enzyme standardization.
    enz_sd : np.ndarray
        Std used for enzyme standardization.

    Returns
    -------
    X_prepped : np.ndarray
        Preprocessed features with same shape as X.
    """
    X = X.astype(np.float32, copy=True)
    if d_enz > 0:
        X[:, :d_enz] = (X[:, :d_enz] - enz_mu) / enz_sd
    if d_fp > 0:
        X[:, d_enz:] = (X[:, d_enz:] > 0.5).astype(np.float32)
    return X

def _make_train_val_split(n: int, val_frac: float, seed: int, groups=None):
    """
    Deterministically split indices into train/val sets.
    Supports optional group-aware splitting via GroupShuffleSplit.

    Parameters
    ----------
    n : int
        Number of samples.
    val_frac : float
        Fraction of samples to use for validation (bounded to [0.05, 0.3]).
    seed : int
        Random seed.
    groups : array-like, optional
        Group labels for group-aware splitting.

    Returns
    -------
    tr_idx : np.ndarray
        Indices for training.
    val_idx : np.ndarray
        Indices for validation.
    """
    idx = np.arange(int(n))
    val_frac = float(val_frac)
    # enforce sane bounds on val fraction
    if val_frac <= 0 or val_frac >= 0.5:
        val_frac = min(max(val_frac, 0.05), 0.3)

    if groups is None:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(idx)
        n_val = max(1, int(val_frac * n))
        val_idx = idx[:n_val]
        tr_idx  = idx[n_val:]
        return tr_idx, val_idx

    # group-aware split
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError(f"groups length mismatch: len(groups)={len(groups)} vs n={n}")

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=int(seed))
    tr_idx, val_idx = next(gss.split(idx, groups=groups))
    return np.asarray(tr_idx), np.asarray(val_idx)

class SupervisedVAE(nn.Module):
    """
    Single-tower supervised variational autoencoder.

    This implementation matches the inline 6.2 model from the Phase 4A notebook.
    """
    def __init__(self, d_enz: int, d_fp: int, z_dim: int, h_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.d_enz = int(d_enz)
        self.d_fp  = int(d_fp)
        d_in = self.d_enz + self.d_fp
        if d_in <= 0:
            raise ValueError("d_in must be > 0")

        # Encoder: MLP with n_layers hidden layers of size h_dim.
        enc = []
        d = d_in
        for _ in range(int(n_layers)):
            enc += [nn.Linear(d, int(h_dim)), nn.ReLU(), nn.Dropout(float(dropout))]
            d = int(h_dim)
        self.encoder = nn.Sequential(*enc)
        self.mu     = nn.Linear(int(h_dim), int(z_dim))
        self.logvar = nn.Linear(int(h_dim), int(z_dim))

        # Decoder trunk: MLP on latent z for reconstruction.
        dec = []
        d = int(z_dim)
        for _ in range(int(n_layers)):
            dec += [nn.Linear(d, int(h_dim)), nn.ReLU(), nn.Dropout(float(dropout))]
            d = int(h_dim)
        self.decoder_trunk = nn.Sequential(*dec)

        # Reconstruction heads (one per modality).
        self.dec_enz       = nn.Linear(int(h_dim), self.d_enz) if self.d_enz > 0 else None
        self.dec_fp_logits = nn.Linear(int(h_dim), self.d_fp)  if self.d_fp  > 0 else None

        # Classifier head: maps latent z to a binary logit.
        self.cls = nn.Sequential(
            nn.Linear(int(z_dim), int(h_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(h_dim), 1),
        )

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, sigma^2) where sigma = exp(0.5 * logvar).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, *, sample_z: bool = True, cls_use_mu: bool = False):
        """
        Forward pass of the VAE.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch_size, d_in).
        sample_z : bool, default True
            Whether to sample z from the approximate posterior (train mode).
            If False, use mu deterministically (e.g. for validation/inference).
        cls_use_mu : bool, default False
            If True, classifier always uses mu (decouples classifier from sampling noise).

        Returns
        -------
        dict
            Dictionary containing:
              - mu (torch.Tensor): mean of q(z|x), shape (batch_size, z_dim)
              - logvar (torch.Tensor): log variance of q(z|x), shape (batch_size, z_dim)
              - z (torch.Tensor): latent vector fed to decoder, shape (batch_size, z_dim)
              - enz_hat (torch.Tensor | None): reconstructed enzyme, shape (batch_size, d_enz) or None
              - fp_logits (torch.Tensor | None): reconstructed fingerprint logits, shape (batch_size, d_fp) or None
              - y_logit (torch.Tensor): classifier logit, shape (batch_size,)
        """
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)

        # sample latent once
        z_sample = self.reparam(mu, logvar)

        # decoder uses stochastic z during training or mu during deterministic passes
        z_dec = z_sample if sample_z else mu

        # classifier uses mu if cls_use_mu else z_dec
        z_cls = mu if cls_use_mu else z_dec

        # decode
        hd = self.decoder_trunk(z_dec)
        enz_hat   = self.dec_enz(hd) if self.dec_enz is not None else None
        fp_logits = self.dec_fp_logits(hd) if self.dec_fp_logits is not None else None
        # flatten classifier output
        y_logit   = self.cls(z_cls).squeeze(1)

        return dict(mu=mu, logvar=logvar, z=z_dec, enz_hat=enz_hat, fp_logits=fp_logits, y_logit=y_logit)

def compute_loss(out: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
                 cfg: dict, d_enz: int, d_fp: int, beta: float):
    """
    Compute weighted total loss, classification loss, reconstruction loss and KL divergence.

    Parameters
    ----------
    out : dict
        Output dictionary from SupervisedVAE.forward.
    x : torch.Tensor
        Input batch of shape (batch_size, d_in).
    y : torch.Tensor
        Binary labels (0/1) of shape (batch_size,).
    w : torch.Tensor
        Sample weights of shape (batch_size,).
    cfg : dict
        Configuration dict with alpha_recon.
    d_enz : int
        Enzyme dimension.
    d_fp : int
        Fingerprint dimension.
    beta : float
        Weight on KL term.

    Returns
    -------
    total : torch.Tensor
        Total scalar loss.
    cls : torch.Tensor
        Classification loss.
    rec : torch.Tensor
        Reconstruction loss.
    kl : torch.Tensor
        KL divergence.
    """
    rec_terms = []

    # reconstruction loss (enzyme)
    if d_enz > 0:
        enz_rec = F.mse_loss(out["enz_hat"], x[:, :d_enz], reduction="none").mean(dim=1)
        rec_terms.append(enz_rec)

    # reconstruction loss (fingerprints)
    if d_fp > 0:
        fp_rec = F.binary_cross_entropy_with_logits(out["fp_logits"], x[:, d_enz:], reduction="none").mean(dim=1)
        rec_terms.append(fp_rec)

    if not rec_terms:
        raise RuntimeError(f"Empty rec_terms (d_enz={d_enz}, d_fp={d_fp})")

    rec = torch.stack(rec_terms, dim=0).mean(dim=0).mean()

    # classification (weighted BCE)
    cls_raw = F.binary_cross_entropy_with_logits(out["y_logit"], y, reduction="none")
    w_sum = (w.sum() + 1e-12)
    cls = (cls_raw * w).sum() / w_sum

    # KL divergence between q(z|x) and prior N(0, I)
    mu, logvar = out["mu"], out["logvar"]
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()

    total = cls + float(cfg["alpha_recon"]) * rec + float(beta) * kl
    return total, cls, rec, kl

def train_supervised_vae(X_train, y_train, w_train, cfg: dict, mode: str = "full", *, groups=None):
    """
    Train a supervised VAE on the provided data.
    Implements early stopping based on validation AP (deterministic using mu).

    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_samples, d_in).
    y_train : np.ndarray
        Binary labels of shape (n_samples,).
    w_train : np.ndarray
        Sample weights of shape (n_samples,).
    cfg : dict
        Hyperparameter configuration (see notebook for fields).
    mode : str, default "full"
        One of {"full", "enzyme_only", "substrate_only"}.
    groups : array-like, optional
        Group labels for optional group-aware splitting.

    Returns
    -------
    model : SupervisedVAE
        Trained VAE model on train split.
    scal : dict
        Dictionary containing scaler parameters (enz_mu, enz_sd, d_enz, d_fp, mode).
    log_df : pd.DataFrame
        DataFrame of training/validation logs per epoch.
    best_epoch : int
        Epoch achieving best validation AP.
    """
    # ensure deterministic algorithms disabled for potential performance
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    set_seed(int(cfg["seed"]))

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    w_train = np.asarray(w_train, dtype=np.float32).reshape(-1)

    if len(X_train) != len(y_train) or len(X_train) != len(w_train):
        raise ValueError(f"Shape mismatch: X={len(X_train)}, y={len(y_train)}, w={len(w_train)}")

    d_in = int(X_train.shape[1])
    d_enz, d_fp = infer_dims(d_in, cfg, mode)

    # train/val split
    n = int(len(X_train))
    tr_idx, val_idx = _make_train_val_split(n, float(cfg["val_frac"]), int(cfg["seed"]), groups=groups)

    # fit scaler on train split
    enz_mu, enz_sd = fit_scaler(X_train[tr_idx], d_enz)
    Xtr = prep_X(X_train[tr_idx], d_enz, d_fp, enz_mu, enz_sd)
    Xva = prep_X(X_train[val_idx], d_enz, d_fp, enz_mu, enz_sd)

    ytr, wtr = y_train[tr_idx], w_train[tr_idx]
    yva, wva = y_train[val_idx], w_train[val_idx]

    # DataLoaders
    ds_tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(wtr))
    ds_va = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva), torch.from_numpy(wva))
    dl_tr = DataLoader(ds_tr, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=int(cfg["batch_size"]), shuffle=False, drop_last=False)

    # model and optimizer
    model = SupervisedVAE(
        d_enz=int(d_enz),
        d_fp=int(d_fp),
        z_dim=int(cfg["z_dim"]),
        h_dim=int(cfg["h_dim"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("use_amp", False)) and DEVICE == "cuda")

    # training loop with early stopping
    best_ap = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    log_rows = []

    max_epochs = int(cfg["max_epochs"])
    warm = int(cfg["kl_warmup_epochs"])
    beta_max = float(cfg["beta_kl"])
    patience = int(cfg["patience"])
    train_sample_z = bool(cfg.get("train_sample_z", True))

    for epoch in range(1, max_epochs + 1):
        model.train()

        beta = beta_max * min(1.0, epoch / max(1, warm))

        losses = []
        for xb, yb, wb in dl_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            wb = wb.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=bool(cfg.get("use_amp", False)) and DEVICE == "cuda"):
                out = model(
                    xb,
                    sample_z=train_sample_z,
                    cls_use_mu=bool(cfg.get("cls_use_mu", False)),
                )
                loss, cls_loss, rec_loss, kl_loss = compute_loss(out, xb, yb, wb, cfg, d_enz, d_fp, beta=beta)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu().item()))

        # validation (deterministic: sample_z=False)
        model.eval()
        yhat, ytrue, wts = [], [], []
        with torch.no_grad():
            for xb, yb, wb in dl_va:
                xb = xb.to(DEVICE, non_blocking=True)
                out = model(xb, sample_z=False)  # deterministic validation
                p = torch.sigmoid(out["y_logit"]).detach().cpu().numpy()
                yhat.append(p)
                ytrue.append(yb.numpy())
                wts.append(wb.numpy())

        yhat = np.concatenate(yhat).reshape(-1)
        ytrue = np.concatenate(ytrue).reshape(-1)
        wts = np.concatenate(wts).reshape(-1)

        # compute validation metrics
        ap = float("nan")
        au = float("nan")
        if len(np.unique(ytrue)) > 1:
            ap = float(average_precision_score(ytrue, yhat, sample_weight=wts))
            au = float(roc_auc_score(ytrue, yhat, sample_weight=wts))

        log_rows.append(
            dict(epoch=int(epoch), loss=float(np.mean(losses)), val_ap=float(ap), val_auroc=float(au),
                 beta=float(beta), train_sample_z=bool(train_sample_z))
        )

        if np.isfinite(ap) and (ap > best_ap + 1e-6):
            best_ap = float(ap)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = int(epoch)
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    if best_state is None:
        raise AssertionError("Training failed: no best_state captured (val AP never finite/improving).")

    model.load_state_dict(best_state)

    scal = dict(
        enz_mu=np.asarray(enz_mu, dtype=np.float32),
        enz_sd=np.asarray(enz_sd, dtype=np.float32),
        d_enz=int(d_enz),
        d_fp=int(d_fp),
        mode=str(mode),
    )

    return model, scal, pd.DataFrame(log_rows), best_epoch

def retrain_vae_full_train(X_train, y_train, w_train, cfg: dict, best_epoch: int,
                           scal: dict | None = None, mode: str = "full"):
    """
    Retrain a supervised VAE on the full training data for a fixed number of epochs.

    Parameters
    ----------
    X_train : np.ndarray
        Features on the full training set.
    y_train : np.ndarray
        Labels on the full training set.
    w_train : np.ndarray
        Sample weights on the full training set.
    cfg : dict
        Hyperparameter configuration.
    best_epoch : int
        Number of epochs to train (e.g. from early stopping).
    scal : dict, optional
        Scaler dictionary; if None, scaler is fitted on full training data.
    mode : str, default "full"
        Input mode.

    Returns
    -------
    model : SupervisedVAE
        Retrained model.
    scal_out : dict
        Updated scaler dictionary.
    """
    set_seed(int(cfg["seed"]))

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    w_train = np.asarray(w_train, dtype=np.float32).reshape(-1)

    if len(X_train) != len(y_train) or len(X_train) != len(w_train):
        raise ValueError(f"Shape mismatch: X={len(X_train)}, y={len(y_train)}, w={len(w_train)}")

    d_in = int(X_train.shape[1])
    d_enz, d_fp = infer_dims(d_in, cfg, mode)

    # Use provided scaler or fit on full train for enzyme dims
    if scal is None:
        enz_mu, enz_sd = fit_scaler(X_train, d_enz)
    else:
        enz_mu = np.asarray(scal.get("enz_mu"), dtype=np.float32)
        enz_sd = np.asarray(scal.get("enz_sd"), dtype=np.float32)

    X_prep = prep_X(X_train, d_enz, d_fp, enz_mu, enz_sd)

    ds = TensorDataset(torch.from_numpy(X_prep), torch.from_numpy(y_train), torch.from_numpy(w_train))
    dl = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=False)

    model = SupervisedVAE(
        d_enz=int(d_enz),
        d_fp=int(d_fp),
        z_dim=int(cfg["z_dim"]),
        h_dim=int(cfg["h_dim"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.get("use_amp", False)) and DEVICE == "cuda")

    warm = int(cfg["kl_warmup_epochs"])
    beta_max = float(cfg["beta_kl"])
    train_sample_z = bool(cfg.get("train_sample_z", True))

    # Training for exactly best_epoch epochs
    for epoch in range(1, int(best_epoch) + 1):
        model.train()
        beta = beta_max * min(1.0, epoch / max(1, warm))

        for xb, yb, wb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            wb = wb.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=bool(cfg.get("use_amp", False)) and DEVICE == "cuda"):
                out = model(
                    xb,
                    sample_z=train_sample_z,
                    cls_use_mu=bool(cfg.get("cls_use_mu", False)),
                )
                loss, _, _, _ = compute_loss(out, xb, yb, wb, cfg, d_enz, d_fp, beta=beta)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    scal_out = dict(
        enz_mu=np.asarray(enz_mu, dtype=np.float32),
        enz_sd=np.asarray(enz_sd, dtype=np.float32),
        d_enz=int(d_enz),
        d_fp=int(d_fp),
        mode=str(mode),
    )

    return model, scal_out

def predict_with_latent(model: SupervisedVAE, scal: dict, X: np.ndarray, *,
                        batch_size: int | None = None, use_mu: bool = True,
                        mc_samples: int = 0, cls_use_mu: bool = False):
    """
    Make predictions and optionally return latent representations.

    Parameters
    ----------
    model : SupervisedVAE
        Trained model.
    scal : dict
        Scaler dictionary containing enz_mu, enz_sd, d_enz, d_fp, mode.
    X : np.ndarray
        Input features of shape (n_samples, d_in).
    batch_size : int, optional
        Batch size for batched prediction; if None, no batching.
    use_mu : bool, default True
        If True and mc_samples <= 0, run deterministic inference (sample_z=False).
    mc_samples : int, default 0
        Number of MC samples for stochastic forward passes; averaged over samples.
    cls_use_mu : bool, default False
        If True, classifier always uses mu.

    Returns
    -------
    p : np.ndarray
        Predicted probabilities of shape (n_samples,).
    mu : np.ndarray
        Mean latent representation of shape (n_samples, z_dim).
    z : np.ndarray
        Latent vector used for decoder of shape (n_samples, z_dim).
    """
    X = np.asarray(X, dtype=np.float32)
    d_enz, d_fp = int(scal["d_enz"]), int(scal["d_fp"])
    enz_mu = np.asarray(scal["enz_mu"], dtype=np.float32)
    enz_sd = np.asarray(scal["enz_sd"], dtype=np.float32)

    Xp = prep_X(X, d_enz, d_fp, enz_mu, enz_sd)

    model.eval()

    def _forward_probs(xb_t: torch.Tensor):
        # deterministic single pass when use_mu and no MC sampling
        if use_mu and mc_samples <= 0:
            out = model(xb_t, sample_z=False, cls_use_mu=cls_use_mu)
            p = torch.sigmoid(out["y_logit"])
            return p, out
        # MC averaging
        K = int(mc_samples) if mc_samples and mc_samples > 0 else 1
        ps = []
        out_last = None
        for _ in range(K):
            out_last = model(xb_t, sample_z=True, cls_use_mu=cls_use_mu)
            ps.append(torch.sigmoid(out_last["y_logit"]))
        p = torch.stack(ps, dim=0).mean(dim=0)
        return p, out_last

    # Non-batched mode
    if batch_size is None:
        xb = torch.from_numpy(Xp).to(DEVICE)
        with torch.no_grad():
            p_t, out = _forward_probs(xb)
            p = p_t.detach().cpu().numpy()
            mu = out["mu"].detach().cpu().numpy()
            z = out["z"].detach().cpu().numpy()
        return p.reshape(-1), mu, z

    # Batched mode
    bs = int(batch_size)
    p_list, mu_list, z_list = [], [], []
    with torch.no_grad():
        for i0 in range(0, len(Xp), bs):
            xb = torch.from_numpy(Xp[i0:i0 + bs]).to(DEVICE)
            p_t, out = _forward_probs(xb)
            p_list.append(p_t.detach().cpu().numpy())
            mu_list.append(out["mu"].detach().cpu().numpy())
            z_list.append(out["z"].detach().cpu().numpy())
    p = np.concatenate(p_list, axis=0).reshape(-1)
    mu = np.concatenate(mu_list, axis=0)
    z = np.concatenate(z_list, axis=0)
    return p, mu, z

def _device_kwargs():
    # XGBoost will use GPU if available and configured; otherwise CPU.
    # Keep existing behavior minimal: rely on xgboost default device unless user set env.
    return {}

def _get_label_and_weight(df: pd.DataFrame):
    """
    Resolves label + weight columns from df.
    Expects a binary label column.
    """
    # Label (match original Track A behavior)
    label_cands = ["reaction", "y", "label", "class", "is_active"]
    label_col = next((c for c in label_cands if c in df.columns), None)
    if label_col is None:
        raise AssertionError(
            f"No label column found. Expected one of {label_cands}. "
            f"Have: {list(df.columns)[:50]}..."
        )

    y = df[label_col].to_numpy()
    # Robust cast (works for bool/int/0-1 strings)
    y = y.astype(int)

    uniq = set(np.unique(y).tolist())
    if not uniq.issubset({0, 1}):
        raise AssertionError(f"Labels must be binary {{0,1}}. Found: {sorted(uniq)} in col={label_col}")

    # Weight
    if "weight" in df.columns:
        w_col = "weight"
        w = df[w_col].to_numpy(dtype=float)
    elif "sample_weight" in df.columns:
        w_col = "sample_weight"
        w = df[w_col].to_numpy(dtype=float)
    elif "w" in df.columns:
        w_col = "w"
        w = df[w_col].to_numpy(dtype=float)
    else:
        w_col = None
        w = np.ones(len(df), dtype=float)

    return label_col, w_col, y, w

def _build_X(df_part: pd.DataFrame, embs: np.ndarray, fps: np.ndarray) -> np.ndarray:
    if not {"enz_idx", "sub_idx"}.issubset(df_part.columns):
        raise AssertionError("pairs df_part must contain enz_idx and sub_idx columns.")

    enz_i = df_part["enz_idx"].to_numpy(dtype=int)
    sub_i = df_part["sub_idx"].to_numpy(dtype=int)

    if enz_i.min() < 0 or enz_i.max() >= len(embs):
        raise AssertionError(f"enz_idx out of range: min={enz_i.min()}, max={enz_i.max()}, n_embs={len(embs)}")
    if sub_i.min() < 0 or sub_i.max() >= len(fps):
        raise AssertionError(f"sub_idx out of range: min={sub_i.min()}, max={sub_i.max()}, n_fps={len(fps)}")

    X = np.hstack([embs[enz_i], fps[sub_i]])
    return X

def _build_X_mode(df_part: pd.DataFrame, embs: np.ndarray, fps: np.ndarray, mode: str):
    """
    mode in:
      - full
      - enzyme_only
      - substrate_only
    """
    mode = str(mode).strip().lower()
    if mode == "full":
        return _build_X(df_part, embs, fps)

    if not {"enz_idx", "sub_idx"}.issubset(df_part.columns):
        raise AssertionError("pairs df_part must contain enz_idx and sub_idx columns.")

    enz_i = df_part["enz_idx"].to_numpy(dtype=int)
    sub_i = df_part["sub_idx"].to_numpy(dtype=int)

    if mode == "enzyme_only":
        return embs[enz_i]
    if mode == "substrate_only":
        return fps[sub_i]

    raise ValueError(f"Unknown mode: {mode}")

def _load_best_params(universe_tag: str, emb_tag: str, track: str):
    """
    Robust loader for best_params artifacts.

    Supports JSON shapes:
      (A) {"params": {...}, "suggested_n_estimators": 1234, ...}
      (B) direct params dict (legacy): {"max_depth": 8, ...}

    Search order:
      1) PROJ/metrics/hpo_{internal,external}/<run_name>/best_params_<emb_tag>.json (new layout, newest mtime)
      2) PROJ/metrics/hpo/best_params__{track}__{universe}__{emb_tag}.json (legacy)
      3) PROJ/metrics/hpo/best_params__{universe}__{emb_tag}.json
      4) PROJ/metrics/hpo/best_params_<emb_tag>.json
    """
    universe_tag = str(universe_tag).strip()
    emb_tag      = str(emb_tag).strip()
    track        = str(track).strip().lower()

    def _read_obj(fp: Path) -> dict:
        return json.loads(fp.read_text())

    def _parse_obj(obj: dict):
        meta = {}
        if isinstance(obj, dict) and "params" in obj and isinstance(obj["params"], dict):
            params = obj["params"]
            meta["best_obj"] = obj
            meta["suggested_n_estimators"] = obj.get("suggested_n_estimators", obj.get("suggested_n_estimator", None))
            return params, meta
        if isinstance(obj, dict):
            meta["best_obj"] = None
            meta["suggested_n_estimators"] = None
            return obj, meta
        raise ValueError("best_params JSON must be a dict")

    def _pick_newest(files):
        files = [Path(f) for f in files if f is not None and Path(f).exists()]
        if not files:
            return None
        return max(files, key=lambda p: p.stat().st_mtime)

    preferred_roots = []
    if track.startswith("int"):
        preferred_roots.append(PROJ / "metrics" / "hpo_internal")
    elif track.startswith("ext"):
        preferred_roots.append(PROJ / "metrics" / "hpo_external")
    preferred_roots += [PROJ / "metrics" / "hpo_internal", PROJ / "metrics" / "hpo_external"]

    run_globs = []
    if track.startswith("int"):
        run_globs += [f"{universe_tag}__internal_*"]
    elif track.startswith("ext"):
        run_globs += [f"{universe_tag}__external_*"]
    run_globs += [f"{universe_tag}__*"]

    cand_files = []
    for root in preferred_roots:
        if not root.exists():
            continue
        for rg in run_globs:
            for run_dir in root.glob(rg):
                if run_dir.is_dir():
                    cand_files.append(run_dir / f"best_params_{emb_tag}.json")

    fp = _pick_newest(cand_files)
    if fp is not None:
        obj = _read_obj(fp)
        params, meta = _parse_obj(obj)
        meta["source"] = "new_layout"
        return fp, params, meta

    legacy = [
        PROJ/"metrics"/"hpo"/f"best_params__{track}__{universe_tag}__{emb_tag}.json",
        PROJ/"metrics"/"hpo"/f"best_params__{universe_tag}__{emb_tag}.json",
        PROJ/"metrics"/"hpo"/f"best_params_{emb_tag}.json",
        PROJ/"metrics"/"hpo"/f"best_params_{emb_tag}",
    ]
    fp = _pick_first_existing(legacy)
    if fp is None:
        return None, None, {}

    obj = _read_obj(fp)
    params, meta = _parse_obj(obj)
    meta["source"] = "legacy_flat"
    return fp, params, meta

def _default_xgb_params():
    return dict(
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=2.0,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=0.5,
        reg_alpha=0.0,
        n_estimators=1000,
    )

def _fit_xgb(X, y, w, params: dict, seed: int = 0):
    kw = _device_kwargs()
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_jobs=os.cpu_count() or 2,
        random_state=seed,
        verbosity=0,
        **params,
        **kw,
    )
    clf.fit(X, y, sample_weight=w)
    return clf

def _weighted_pr_f1_sweep(y, p, w):
    prec, rec, thr = precision_recall_curve(y, p, sample_weight=w)
    prec_t, rec_t = prec[:-1], rec[:-1]  # last point has no threshold
    f1 = (2.0 * prec_t * rec_t) / (prec_t + rec_t + 1e-12)
    best_i = int(np.argmax(f1))
    t = float(thr[best_i])
    return t, float(f1[best_i]), prec_t, rec_t, thr, f1

def _maybe_write_identity_band_report(run_dir: Path, df_test: pd.DataFrame, y, p, w):
    bands_csv = PROJ / "splits" / "test_identity_bands.csv"
    if not bands_csv.exists() or ("enzyme" not in df_test.columns):
        return None

    dfb = pd.read_csv(bands_csv)
    if "band" not in dfb.columns and "identity_band" in dfb.columns:
        dfb = dfb.rename(columns={"identity_band": "band"})

    te_enzyme = df_test["enzyme"].astype(str).to_frame(name="enzyme")
    band = te_enzyme.merge(dfb, on="enzyme", how="left")["band"].fillna("60–80%").astype(str)

    rows = []
    for name in sorted(band.unique().tolist()):
        m = (band.values == name)
        if m.sum() == 0:
            continue
        yb, pb, wb = y[m], p[m], w[m]
        rows.append(dict(
            band=name,
            n_rows=int(m.sum()),
            n_enzymes=int(df_test.loc[m, "enzyme"].nunique()),
            pos_rate=float(np.average(yb, weights=wb)),
            auroc=float(roc_auc_score(yb, pb, sample_weight=wb)) if len(np.unique(yb)) > 1 else float("nan"),
            ap=float(average_precision_score(yb, pb, sample_weight=wb)) if len(np.unique(yb)) > 1 else float("nan"),
            brier=float(brier_score_loss(yb, pb, sample_weight=wb)),
        ))
    out = pd.DataFrame(rows)
    out_dir = _ensure_dir(run_dir / "trackA_internal" / "by_identity_band")
    out.to_csv(out_dir / "by_band.csv", index=False)
    return out

def _cap_params_for_sanity(params: dict, cap: int | None):
    p = dict(params)
    if cap is not None and "n_estimators" in p:
        try:
            p["n_estimators"] = int(min(int(p["n_estimators"]), int(cap)))
        except Exception:
            pass
    return p

def _slug(s: str) -> str:
    s = str(s).replace("–", "-").replace("—", "-")
    out = []
    for ch in s:
        out.append(ch if (ch.isalnum() or ch in ".-") else "_")
    tag = "".join(out)
    while "__" in tag:
        tag = tag.replace("__", "_")
    return tag.strip("_")

def _print_transcript_if_exists(run_dir: Path) -> bool:
    fp = Path(run_dir) / "console_transcript.txt"
    if not fp.exists():
        return False
    try:
        print(fp.read_text())
        return True
    except Exception:
        return False

def is_trackA_complete(run_dir: Path, *, do_sanity: bool, do_sub_seen_unseen: bool, enable_similarity_bins: bool) -> bool:
    run_dir = Path(run_dir)
    test_dir = run_dir / "trackA_internal" / "test"
    ok, _ = _quiet_bundle_ok(test_dir)
    if not ok:
        return False

    if do_sub_seen_unseen:
        # require at least one subset dir to be present + complete; if present, must be complete
        seen_dir = run_dir / "trackA_internal" / "test_sub_seen"
        unseen_dir = run_dir / "trackA_internal" / "test_sub_unseen"
        any_present = False

        if seen_dir.exists():
            any_present = True
            ok, _ = _quiet_bundle_ok(seen_dir)
            if not ok:
                return False
        if unseen_dir.exists():
            any_present = True
            ok, _ = _quiet_bundle_ok(unseen_dir)
            if not ok:
                return False

        if not any_present:
            return False

    if enable_similarity_bins:
        sim_sum = run_dir / "trackA_internal" / "test_quadrants" / "sim_bins" / "sim_bin_summary.csv"
        sim_root = run_dir / "trackA_internal" / "test_quadrants" / "sim_bins"
        if not (sim_sum.exists() or sim_root.exists()):
            return False

    if do_sanity:
        if DO_SANITY_ABLATIONS:
            for mode in ["enzyme_only", "substrate_only"]:
                d = run_dir / "trackA_internal" / "sanity" / mode
                if not d.exists():
                    return False
                ok, _ = _quiet_bundle_ok(d)
                if not ok:
                    return False
        if DO_SANITY_PERMUTE_TEST:
            d = run_dir / "trackA_internal" / "sanity" / "permute_enz_test"
            if not d.exists():
                return False
            ok, _ = _quiet_bundle_ok(d)
            if not ok:
                return False

        # sanity summary file is expected when sanity enabled
        sc = run_dir / "trackA_internal" / "sanity_checks.json"
        if not sc.exists():
            return False

    return True

def replay_trackA_from_artifacts(run_dir: Path, *, universe_tag: str, substrate_kind: str, emb_tag: str, split_json: Path | None = None) -> None:
    run_dir = Path(run_dir)

    man_fp = run_dir / "run_manifest.json"
    man = _read_json(man_fp) if man_fp.exists() else None
    man = man if isinstance(man, dict) else {}

    # Header line (the single allowed extra line in compute; replay uses it too)
    print(f"[Track A] Features: enzyme={emb_tag} substrate={substrate_kind} | sim_bins=morgan")

    # pairs + counts
    pairs_fp = man.get("pairs_fp", None)
    n_pairs = None
    if pairs_fp:
        try:
            n_pairs = int(man.get("n_pairs", None)) if man.get("n_pairs") is not None else None
        except Exception:
            n_pairs = None

    if pairs_fp and n_pairs is not None:
        print(f"[Track A] pairs={Path(pairs_fp).name} | rows={n_pairs:,}")
    else:
        # lightweight: load pairs table to recover rows + filename
        try:
            pairs = _load_pairs_universe(universe_tag)
            pairs_p = Path(pairs.attrs.get("_pairs_fp", "unknown"))
            print(f"[Track A] pairs={pairs_p.name} | rows={len(pairs):,}")
        except Exception:
            if pairs_fp:
                print(f"[Track A] pairs={Path(pairs_fp).name}")
            else:
                print("[Track A] pairs=<unknown>")

    # train/test counts
    n_train = man.get("n_train", None)
    n_test = man.get("n_test", None)
    if (n_train is None) or (n_test is None):
        # infer via split_json if possible
        try:
            sj = Path(split_json) if split_json is not None else Path(man.get("split_json"))
            pairs = _load_pairs_universe(universe_tag)
            spl = json.loads(sj.read_text())
            train_enz = set(map(str, spl.get("train_enzymes", [])))
            test_enz  = set(map(str, spl.get("test_enzymes", [])))
            enz = pairs["enzyme"].astype(str)
            tr = int(enz.isin(train_enz).sum())
            te = int(enz.isin(test_enz).sum())
            n_train, n_test = tr, te
        except Exception:
            n_train, n_test = None, None
    if (n_train is not None) and (n_test is not None):
        print(f"[Track A] TRAIN rows={int(n_train):,} | TEST rows={int(n_test):,}")

    # group info (best-effort)
    gsrc = man.get("groups_train_source", None)
    if gsrc == "cluster_id_80":
        print("[Track A] Using group=cluster_id_80 for inner CV.")
    elif gsrc == "enzyme":
        pass

    # HPO provenance (best-effort)
    bp = man.get("best_params_source_fp", None)
    if bp:
        print(f"[Track A] Loaded best_params from: {bp}")
    if man.get("suggested_n_estimators") is not None:
        print(f"[Track A] Using suggested_n_estimators={man.get('suggested_n_estimators')} for final fit.")

    # Frozen params prints (best-effort)
    if "frozen_params" in man and isinstance(man["frozen_params"], dict):
        print("[HPO] Frozen params for reuse:")
        print("  FROZEN_BP_FP =", man.get("frozen_best_params_source_fp"))
        print("  FROZEN_HPO_SOURCE =", man.get("frozen_hpo_source"))
        try:
            print("  FROZEN_PARAMS keys =", sorted(list(man["frozen_params"].keys())))
        except Exception:
            pass

    # substrate seen/unseen membership line (best-effort)
    sc_fp = run_dir / "trackA_internal" / "sanity_checks.json"
    if sc_fp.exists():
        try:
            sc = json.loads(sc_fp.read_text())
            sub = sc.get("substrate_seen_unseen", None)
            if isinstance(sub, dict) and ("n_sub_seen" in sub) and ("n_sub_unseen" in sub):
                print(f"[Track A] Test substrate membership: seen={int(sub['n_sub_seen']):,} | unseen={int(sub['n_sub_unseen']):,}")
        except Exception:
            pass

    # bundle checks (prints)
    _bundle_smoke_check(run_dir / "trackA_internal" / "test")
    for d in [
        run_dir / "trackA_internal" / "test_sub_seen",
        run_dir / "trackA_internal" / "test_sub_unseen",
    ]:
        if d.exists():
            _bundle_smoke_check(d)

    # main metric line
    h = _read_headline_json(run_dir / "trackA_internal" / "test")
    if isinstance(h, dict):
        prefer = "t_oof_f1" if ("t_oof_f1" in (h.get("thresholded") or {})) else "t0p5"
        if globals().get("REPORT_BINARY_METRICS", True) and ("_fmt_track_line" in globals()):
            print(f"[Track A] {universe_tag} internal_test: " + _fmt_track_line(h, prefer=prefer))
        else:
            print(f"[Track A] {universe_tag} internal_test: AUROC={h.get('auroc_weighted', float('nan')):.3f} | AP={h.get('ap_weighted', float('nan')):.3f}")

    # sim bins summary (best-effort)
    sim_sum = run_dir / "trackA_internal" / "test_quadrants" / "sim_bins" / "sim_bin_summary.csv"
    if sim_sum.exists():
        try:
            df = pd.read_csv(sim_sum)
            cols = [c for c in ["sim_bin","n","frac_of_test","pos_rate_weighted","auroc_weighted","ap_weighted","threshold_used","f1_weighted"] if c in df.columns]
            if cols:
                print("\n[A1 sim-bin] Summary (sorted):")
                print(df[cols].to_string(index=False))
            print("[A1 novelty] Wrote similarity-bin reports under trackA_internal/test_quadrants/sim_bins/")
        except Exception:
            pass

    # sanity prints (best-effort)
    if sc_fp.exists():
        try:
            sc = json.loads(sc_fp.read_text())
            ab = sc.get("ablations", {}) or {}
            for mode in ["enzyme_only", "substrate_only"]:
                hd = ((ab.get(mode) or {}).get("headline") or {})
                if isinstance(hd, dict) and ("auroc_weighted" in hd) and ("ap_weighted" in hd):
                    print(f"[Sanity] Ablation {mode}: AUROC={hd['auroc_weighted']:.3f} | AP={hd['ap_weighted']:.3f}")

            pt = sc.get("permutation_test", None)
            if isinstance(pt, dict) and isinstance(pt.get("headline"), dict) and isinstance(pt.get("delta_vs_full"), dict):
                hd = pt["headline"]
                d  = pt["delta_vs_full"]
                print(f"[Sanity] Permute enzymes in TEST: AUROC={hd.get('auroc_weighted', float('nan')):.3f} | "
                      f"AP={hd.get('ap_weighted', float('nan')):.3f} | ΔAP={d.get('delta_ap', float('nan')):+.3f}")

            print("[Sanity] Wrote:", sc_fp)
        except Exception:
            pass

    # DONE block
    model_fp = run_dir / "trackA_internal" / "model" / "model.json"
    print("[Track A] DONE")
    print("  RUN_DIR =", run_dir)
    if model_fp.exists():
        print("  Model   =", model_fp)
    hj = run_dir / "trackA_internal" / "test" / "internal_test_headline.json"
    if hj.exists():
        print("  Headline JSON =", hj)
    elif (run_dir / "trackA_internal" / "test" / "headline.json").exists():
        print("  Headline JSON =", run_dir / "trackA_internal" / "test" / "headline.json")
    if sc_fp.exists():
        print("  Sanity JSON   =", sc_fp)

def run_trackA_internal_xgb(*,
        proj: Path,
        universe_tag: str,
        split_json: Path | None,
        emb_tag: str,
        emb_fp: Path,
        substrate_kind: str,      # "morgan" or "molencoder" (used for run_id + logs)
        substrate_fp: Path,       # features used by the MODEL
        sim_fp_fp: Path,          # Morgan fingerprint matrix used ONLY for similarity bins analysis
        hpo_source_universe: str = "trainpool",
        hpo_source_track: str = "internal",
        force: bool = False,
        seed: int = 42,
        enable_similarity_bins: bool = True,
        include_substrate_kind_in_runid: bool = True,
        make_legacy_runid_alias: bool = True,
        ) -> Path:
    """Track A (internal enzyme-OOD) XGBoost with cache-first run reuse + transcript replay.

    - MODEL substrate features: `substrate_fp` (Morgan or MolEncoder)
    - Similarity bins analysis: ALWAYS uses Morgan from `sim_fp_fp`
    """
    proj = Path(proj)

    # Ensure helpers that rely on global PROJ / eval knobs work with the passed proj
    global PROJ, EVAL_DO_PER_ENZYME, EVAL_DO_THR_SWEEP, EVAL_DO_CALIB_DIAG
    PROJ = proj

    universe_tag = str(universe_tag).strip()
    emb_tag = str(emb_tag).strip()
    substrate_kind = str(substrate_kind).strip()

    emb_fp = Path(emb_fp)
    substrate_fp = Path(substrate_fp)
    sim_fp_fp = Path(sim_fp_fp)

    # Resolve split_json per notebook convention if not provided
    if split_json is None:
        if "ACTIVE_SPLIT" in globals() and isinstance(globals().get("ACTIVE_SPLIT"), dict) and globals()["ACTIVE_SPLIT"].get("split_json_fp"):
            split_json = Path(globals()["ACTIVE_SPLIT"]["split_json_fp"])
        else:
            split_json = proj / "splits" / f"{universe_tag}_enzyme80_split.json"
    split_json = Path(split_json)

    # ---- load best params early (needed for cfg matching AND for global FROZEN_PARAMS) ----
    bp_fp, best_params, meta = _load_best_params(hpo_source_universe, emb_tag, track=hpo_source_track)
    if best_params is None:
        best_params = _default_xgb_params()
        meta = {}
    else:
        # Apply suggested_n_estimators (if present)
        suggested = meta.get("suggested_n_estimators")
        if suggested is not None:
            best_params = dict(best_params)
            best_params["n_estimators"] = int(suggested)

    # Freeze hyperparameters for reuse in Track B (always set, even on cache hit)
    global FROZEN_PARAMS, FROZEN_META, FROZEN_BP_FP, FROZEN_HPO_SOURCE
    FROZEN_PARAMS = dict(best_params)
    FROZEN_META   = dict(meta)
    FROZEN_BP_FP  = str(bp_fp) if bp_fp else None
    FROZEN_HPO_SOURCE = dict(
        universe=hpo_source_universe,
        track=hpo_source_track,
        emb_tag=emb_tag,
        best_params_fp=FROZEN_BP_FP,
    )

    cfg = _cfg_for_trackA(
        universe_tag=universe_tag,
        split_json=split_json,
        emb_tag=emb_tag,
        emb_fp=emb_fp,
        substrate_kind=substrate_kind,
        substrate_fp=substrate_fp,
        sim_fp_fp=sim_fp_fp,
        hpo_source_universe=hpo_source_universe,
        hpo_source_track=hpo_source_track,
        best_params_fp=bp_fp,
        frozen_params=FROZEN_PARAMS,
        enable_similarity_bins=enable_similarity_bins,
        seed=seed,
    )

    # ---- cache locate ----
    existing = None
    if not force:
        existing = find_existing_trackA_run_dir(proj=proj, cfg=cfg, policy="latest_mtime")

    if (existing is not None) and (not force):
        existing = Path(existing)

        # If complete → replay (transcript preferred) and return
        if is_trackA_complete(
            existing,
            do_sanity=cfg["flags"]["do_sanity_checks"],
            do_sub_seen_unseen=cfg["flags"]["do_sub_seen_unseen"],
            enable_similarity_bins=cfg["flags"]["enable_similarity_bins"],
        ):
            if not _print_transcript_if_exists(existing):
                replay_trackA_from_artifacts(
                    existing,
                    universe_tag=universe_tag,
                    substrate_kind=substrate_kind,
                    emb_tag=emb_tag,
                    split_json=split_json,
                )
            return existing

        # Otherwise: resume into the existing dir
        RUN_DIR = existing
        RUN_ID = RUN_DIR.name
        resume_mode = True
    else:
        # Cache miss → create new run dir (timestamped, but matched later via manifest cfg_hash)
        stamp = _now_tag()
        if include_substrate_kind_in_runid:
            RUN_ID = f"trackA__{universe_tag}__enzymeOOD80__{emb_tag}__{substrate_kind}__{stamp}"
            LEGACY_RUN_ID = f"trackA__{universe_tag}__enzymeOOD80__{emb_tag}__{stamp}"
        else:
            RUN_ID = f"trackA__{universe_tag}__enzymeOOD80__{emb_tag}__{stamp}"
            LEGACY_RUN_ID = RUN_ID

        RUN_DIR = _ensure_dir(proj / "metrics" / "runs" / RUN_ID)
        resume_mode = False

    # Transcript tee (fresh or resume)
    transcript_fp = RUN_DIR / "console_transcript.txt"
    tee_mode = "a" if resume_mode else "w"
    header = None
    if resume_mode:
        header = f"\n\n===== RESUME {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ====="

    with TeeStdout(transcript_fp, mode=tee_mode, session_header=header):

        # One allowed extra log line
        print(f"[Track A] Features: enzyme={emb_tag} substrate={substrate_kind} | sim_bins=morgan")

        # -----------------------------
        # 2) Track A execution (compute/resume)
        # -----------------------------
        assert split_json.exists(), f"Missing: {split_json}"
        assert emb_fp.exists(), f"Missing: {emb_fp}"
        assert substrate_fp.exists(), f"Missing: {substrate_fp}"
        assert sim_fp_fp.exists(), f"Missing sim_fp_fp: {sim_fp_fp}"

        pairs = _load_pairs_universe(universe_tag)
        pairs_fp = Path(pairs.attrs.get("_pairs_fp", "unknown"))
        print(f"[Track A] pairs={pairs_fp.name} | rows={len(pairs):,}")

        # Load MODEL features (enzyme emb + substrate features)
        embs = np.load(emb_fp)
        fps  = np.load(substrate_fp)

        # Split → train_ids/test_ids by enzyme list
        spl = json.loads(split_json.read_text())
        train_enz = set(map(str, spl.get("train_enzymes", [])))
        test_enz  = set(map(str, spl.get("test_enzymes", [])))
        if (not train_enz) or (not test_enz):
            raise AssertionError(f"Split JSON missing train_enzymes/test_enzymes keys: {list(spl.keys())}")

        if "enzyme" not in pairs.columns:
            raise AssertionError("pairs table must have 'enzyme' column for enzyme-OOD split application.")

        enz = pairs["enzyme"].astype(str)
        train_mask = enz.isin(train_enz).to_numpy()
        test_mask  = enz.isin(test_enz).to_numpy()
        train_ids  = np.where(train_mask)[0]
        test_ids   = np.where(test_mask)[0]

        print(f"[Track A] TRAIN rows={len(train_ids):,} | TEST rows={len(test_ids):,}")
        if len(test_ids) == 0:
            raise AssertionError("TEST is empty after applying enzyme split. Check enzyme name harmonization.")

        # Optional: attach cluster_id_80 for group-aware CV on train
        groups_train = None
        clustermap_csv = proj / "splits" / "all_union_enzyme_cluster_id_80.csv"
        groups_train_source = None
        if clustermap_csv.exists():
            cmap = pd.read_csv(clustermap_csv)
            cmap["enzyme"] = cmap["enzyme"].astype(str).str.strip()
            pairs = pairs.merge(cmap, on="enzyme", how="left")
            if "cluster_id_80" in pairs.columns and pairs.loc[train_ids, "cluster_id_80"].notna().any():
                groups_train = pairs.loc[train_ids, "cluster_id_80"].fillna(-1).astype(int).to_numpy()
                groups_train_source = "cluster_id_80"
                print("[Track A] Using group=cluster_id_80 for inner CV.")
        else:
            # preserve original behavior
            print("[Track A] No clustermap found; inner CV will group by enzyme.")

        if groups_train is None:
            groups_train_source = "enzyme"
            groups_train = pairs.loc[train_ids, "enzyme"].astype(str).to_numpy()

        # Label + weights
        label_col, w_col, y_all, w_all = _get_label_and_weight(pairs)

        # Build matrices (FULL) using MODEL substrate features
        X_train = _build_X(pairs.iloc[train_ids], embs, fps)
        y_train = y_all[train_ids]
        w_train = w_all[train_ids]

        X_test  = _build_X(pairs.iloc[test_ids], embs, fps)
        y_test  = y_all[test_ids]
        w_test  = w_all[test_ids]

        # Print HPO provenance like compute path (even on resume)
        if bp_fp is None:
            print("[Track A] best_params not found; using fallback defaults.")
        else:
            print(f"[Track A] Loaded best_params from: {bp_fp}")
            suggested = meta.get("suggested_n_estimators")
            if suggested is not None:
                print(f"[Track A] Using suggested_n_estimators={suggested} for final fit.")

        # Print frozen params block (even on resume)
        assert isinstance(FROZEN_PARAMS, dict) and len(FROZEN_PARAMS) > 0
        assert "n_estimators" in FROZEN_PARAMS, "Expected n_estimators present after freezing."
        print("[HPO] Frozen params for reuse:")
        print("  FROZEN_BP_FP =", FROZEN_BP_FP)
        print("  FROZEN_HPO_SOURCE =", FROZEN_HPO_SOURCE)
        print("  FROZEN_PARAMS keys =", sorted(FROZEN_PARAMS.keys()))

        # Decide what thresholds to report
        thresholds_to_report = None
        oof_dir = None
        t_oof = None

        if REPORT_BINARY_METRICS:
            thresholds_to_report = {"t0p5": float(DEFAULT_THRESHOLD)}

        # Optional: compute/load OOF threshold on TRAIN only (no test touch)
        if DO_OOF_THRESHOLD:
            # reuse existing OOF threshold if present
            cand = RUN_DIR / "trackA_internal" / "train_oof" / "thresholds_oof.json"
            if cand.exists():
                try:
                    obj = json.loads(cand.read_text())
                    t_oof = float(obj.get("threshold_f1_oof", obj.get("t_oof_f1", np.nan)))
                except Exception:
                    t_oof = None

            if (t_oof is None) or (not np.isfinite(t_oof)):
                cv = StratifiedGroupKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=int(seed))
                p_oof = np.full(len(train_ids), np.nan, dtype=float)

                for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train, groups=groups_train), 1):
                    X_tr, y_tr, w_tr = X_train[tr_idx], y_train[tr_idx], w_train[tr_idx]
                    X_va = X_train[va_idx]
                    clf_tmp = _fit_xgb(X_tr, y_tr, w_tr, FROZEN_PARAMS, seed=int(seed) + fold)
                    p_oof[va_idx] = clf_tmp.predict_proba(X_va)[:, 1]

                if not np.isfinite(p_oof).all():
                    raise AssertionError("OOF prediction contains NaN/inf (unexpected).")

                t_oof, f1_oof_max, prec_t, rec_t, thr_grid, f1_grid = _weighted_pr_f1_sweep(y_train, p_oof, w_train)

                oof_dir = _ensure_dir(RUN_DIR / "trackA_internal" / "train_oof")
                np.save(oof_dir / "y_oof.npy", y_train.astype(int))
                np.save(oof_dir / "p_oof.npy", p_oof.astype(float))
                np.save(oof_dir / "w_oof.npy", w_train.astype(float))
                pd.DataFrame({"threshold": thr_grid, "precision": prec_t, "recall": rec_t, "f1": f1_grid}).to_csv(
                    oof_dir / "threshold_sweep_f1.csv", index=False
                )
                (oof_dir / "thresholds_oof.json").write_text(json.dumps({
                    "universe": universe_tag,
                    "emb_tag": emb_tag,
                    "criterion": "maximize_weighted_f1_on_train_oof",
                    "threshold_f1_oof": float(t_oof),
                    "f1_oof_max": float(f1_oof_max),
                    "n_train": int(len(train_ids)),
                    "split_json": str(split_json),
                    "stamp": _now_tag(),
                }, indent=2))

                plt.figure()
                plt.plot(thr_grid, f1_grid)
                plt.axvline(t_oof, linestyle="--")
                plt.xlabel("Threshold"); plt.ylabel("F1 (weighted via PR curve)")
                plt.title(f"TRAIN OOF: F1 vs threshold | chosen t={t_oof:.3f}")
                plt.tight_layout()
                plt.savefig(oof_dir / "f1_vs_threshold_oof.png", dpi=160)
                plt.close()

                print(f"[Track A] OOF threshold computed: t_oof={t_oof:.6f}")
            else:
                print(f"[Track A] OOF threshold computed: t_oof={t_oof:.6f}")

            if REPORT_BINARY_METRICS:
                thresholds_to_report = {"t_oof_f1": float(t_oof), **thresholds_to_report}

        # Fit or load model; then get p_test
        model_dir = _ensure_dir(RUN_DIR / "trackA_internal" / "model")
        model_fp  = model_dir / "model.json"

        booster = None
        if model_fp.exists():
            try:
                booster = xgb.Booster()
                booster.load_model(str(model_fp))
            except Exception:
                booster = None

        if booster is None:
            # train from scratch
            clf_final = _fit_xgb(X_train, y_train, w_train, FROZEN_PARAMS, seed=int(seed))
            clf_final.get_booster().save_model(str(model_fp))
            booster = xgb.Booster()
            booster.load_model(str(model_fp))
        else:
            # best-effort for downstream globals
            clf_final = None

        # Obtain p_test from cached preds if possible, else predict from model
        p_test = None
        cached_preds = RUN_DIR / "trackA_internal" / "test" / "preds.csv"
        if cached_preds.exists():
            try:
                dfp = pd.read_csv(cached_preds)
                if "prob_raw" in dfp.columns and len(dfp) == len(test_ids):
                    p_test = dfp["prob_raw"].to_numpy(dtype=float)
            except Exception:
                p_test = None

        if p_test is None:
            dm = xgb.DMatrix(X_test)
            p_test = booster.predict(dm)

        # -----------------------------
        # Track A EXTRA: test substrate seen vs unseen
        # -----------------------------
        sub_breakdown = None
        df_test_full = pairs.iloc[test_ids].reset_index(drop=True)
        df_train_full = pairs.iloc[train_ids].reset_index(drop=True)
        train_subs = set(df_train_full["sub_idx"].astype(int).tolist())

        if DO_SUBSTRATE_SEEN_UNSEEN_BREAKDOWN and ("sub_idx" in df_test_full.columns):
            m_sub_seen = df_test_full["sub_idx"].astype(int).isin(train_subs).to_numpy()
            m_sub_unseen = ~m_sub_seen

            print(f"[Track A] Test substrate membership: seen={int(m_sub_seen.sum()):,} | unseen={int(m_sub_unseen.sum()):,}")

            sub_breakdown = dict(
                n_test=int(len(df_test_full)),
                n_sub_seen=int(m_sub_seen.sum()),
                n_sub_unseen=int(m_sub_unseen.sum()),
                frac_unseen=float(m_sub_unseen.mean()) if len(df_test_full) else 0.0,
            )

            _prev_per_enz   = EVAL_DO_PER_ENZYME
            _prev_thr_sweep = EVAL_DO_THR_SWEEP
            _prev_calib     = EVAL_DO_CALIB_DIAG

            try:
                EVAL_DO_PER_ENZYME = False  # key speed-up for subsets

                # seen subset
                if m_sub_seen.sum() > 0:
                    seen_dir = RUN_DIR / "trackA_internal" / "test_sub_seen"
                    ok_seen, _ = _quiet_bundle_ok(seen_dir) if seen_dir.exists() else (False, None)
                    if not ok_seen:
                        _ = _eval_and_write(
                            run_dir=RUN_DIR,
                            split_name="internal_test__sub_seen",
                            df_part=df_test_full.loc[m_sub_seen].reset_index(drop=True),
                            y=y_test[m_sub_seen], w=w_test[m_sub_seen], p=p_test[m_sub_seen],
                            thresholds=thresholds_to_report,
                            prefix="trackA_internal/test_sub_seen"
                        )

                # unseen subset
                if m_sub_unseen.sum() > 0:
                    unseen_dir = RUN_DIR / "trackA_internal" / "test_sub_unseen"
                    ok_unseen, _ = _quiet_bundle_ok(unseen_dir) if unseen_dir.exists() else (False, None)
                    if not ok_unseen:
                        _ = _eval_and_write(
                            run_dir=RUN_DIR,
                            split_name="internal_test__sub_unseen",
                            df_part=df_test_full.loc[m_sub_unseen].reset_index(drop=True),
                            y=y_test[m_sub_unseen], w=w_test[m_sub_unseen], p=p_test[m_sub_unseen],
                            thresholds=thresholds_to_report,
                            prefix="trackA_internal/test_sub_unseen"
                        )

            finally:
                EVAL_DO_PER_ENZYME   = _prev_per_enz
                EVAL_DO_THR_SWEEP    = _prev_thr_sweep
                EVAL_DO_CALIB_DIAG   = _prev_calib

        # Main internal_test evaluation (FULL) — skip if already complete
        test_dir = RUN_DIR / "trackA_internal" / "test"
        ok_test, _ = _quiet_bundle_ok(test_dir) if test_dir.exists() else (False, None)

        EVAL_DO_PER_ENZYME = True
        if ok_test:
            headline_full = _read_headline_json(test_dir)
        else:
            headline_full = _eval_and_write(
                run_dir=RUN_DIR,
                split_name="internal_test",
                df_part=df_test_full,
                y=y_test, w=w_test, p=p_test,
                thresholds=thresholds_to_report,
                prefix="trackA_internal/test"
            )

        # --- bundle smoke checks (prints) ---
        _bundle_smoke_check(RUN_DIR / "trackA_internal" / "test")
        for d in [
            RUN_DIR / "trackA_internal" / "test_sub_seen",
            RUN_DIR / "trackA_internal" / "test_sub_unseen",
        ]:
            if d.exists():
                _bundle_smoke_check(d)

        # Track A summary line
        if REPORT_BINARY_METRICS and isinstance(headline_full, dict):
            print(f"[Track A] {universe_tag} internal_test: " + _fmt_track_line(
                headline_full,
                prefer=("t_oof_f1" if DO_OOF_THRESHOLD else "t0p5")
            ))
        elif isinstance(headline_full, dict):
            print(f"[Track A] {universe_tag} internal_test: "
                  f"AUROC={headline_full.get('auroc_weighted', float('nan')):.3f} | "
                  f"AP={headline_full.get('ap_weighted', float('nan')):.3f} | "
                  f"Brier={headline_full.get('brier_weighted', float('nan')):.3f} | "
                  f"LogLoss={headline_full.get('log_loss_weighted', float('nan')):.3f}")

        # Optional identity-band report (threshold-free)
        try:
            _ = _maybe_write_identity_band_report(
                RUN_DIR,
                df_test_full,
                y_test, p_test, w_test
            )
        except Exception:
            pass

        # -----------------------------
        # Similarity bins (analysis-only; ALWAYS Morgan)
        # -----------------------------
        SIM_BINS = [0.0, 0.4, 0.6, 0.8, 1.000001]
        SIM_BIN_LABELS = ["<0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]

        sim_sum_fp = RUN_DIR / "trackA_internal" / "test_quadrants" / "sim_bins" / "sim_bin_summary.csv"
        if enable_similarity_bins:
            if sim_sum_fp.exists():
                # replay a compact summary (best-effort, no recompute)
                try:
                    df_sim = pd.read_csv(sim_sum_fp)
                    print("\n[A1 sim-bin] Summary (sorted):")
                    cols_show = [c for c in [
                        "sim_bin","n","frac_of_test","pos_rate_weighted","auroc_weighted","ap_weighted","threshold_used","f1_weighted"
                    ] if c in df_sim.columns]
                    if cols_show:
                        print(df_sim[cols_show].to_string(index=False))
                    print("[A1 novelty] Wrote similarity-bin reports under trackA_internal/test_quadrants/sim_bins/")
                except Exception:
                    pass
            else:
                # compute bins (no retrain; uses cached preds)
                train_sub_pos = set(df_train_full.loc[(y_train == 1), "sub_idx"].astype(int).tolist())
                train_pos_sub_idx = np.array(sorted(train_sub_pos), dtype=np.int64)

                if len(train_pos_sub_idx) == 0:
                    print("[A1 novelty] No TRAIN-positive substrates; similarity bins skipped.")
                else:
                    fp_sim = np.load(sim_fp_fp)
                    if fp_sim.dtype != np.bool_:
                        fp_sim = (fp_sim > 0).astype(np.uint8)

                    test_sub_idx_unique = df_test_full["sub_idx"].astype(int).unique()
                    test_sub_idx_unique = np.array(test_sub_idx_unique, dtype=np.int64)

                    _POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

                    def _popcount_bytes(a_u8):
                        return _POPCNT[a_u8].sum(axis=-1, dtype=np.int32)

                    def _max_tanimoto_to_ref(query_idx, ref_idx, q_chunk=256, r_chunk=1024):
                        ref = np.packbits(fp_sim[ref_idx], axis=1)
                        ref_cnt = _popcount_bytes(ref).astype(np.int32)

                        out = np.zeros(len(query_idx), dtype=np.float32)
                        for i0 in range(0, len(query_idx), q_chunk):
                            qi = query_idx[i0:i0+q_chunk]
                            q = np.packbits(fp_sim[qi], axis=1)
                            q_cnt = _popcount_bytes(q).astype(np.int32)

                            best = np.zeros(len(qi), dtype=np.float32)
                            for j0 in range(0, len(ref), r_chunk):
                                r = ref[j0:j0+r_chunk]
                                r_cnt = ref_cnt[j0:j0+r_chunk]

                                inter = _popcount_bytes(np.bitwise_and(q[:, None, :], r[None, :, :]))
                                union = q_cnt[:, None] + r_cnt[None, :] - inter
                                sim = inter / np.maximum(union, 1)
                                best = np.maximum(best, sim.max(axis=1).astype(np.float32))
                            out[i0:i0+len(qi)] = best
                        return out

                    max_sim = _max_tanimoto_to_ref(test_sub_idx_unique, train_pos_sub_idx)
                    sim_map = dict(zip(test_sub_idx_unique.tolist(), max_sim.tolist()))
                    df_test_full["sub_max_sim_to_train_pos"] = df_test_full["sub_idx"].astype(int).map(sim_map)

                    df_test_full["sub_sim_bin"] = pd.cut(
                        df_test_full["sub_max_sim_to_train_pos"],
                        bins=SIM_BINS, labels=SIM_BIN_LABELS, right=False
                    )

                    sim_bin_rows = []
                    prefer_thr = ("t_oof_f1" if (REPORT_BINARY_METRICS and DO_OOF_THRESHOLD) else "t0p5")

                    _prev_per_enz = EVAL_DO_PER_ENZYME
                    try:
                        EVAL_DO_PER_ENZYME = False

                        for b in SIM_BIN_LABELS:
                            m = (df_test_full["sub_sim_bin"] == b).to_numpy()
                            n_b = int(m.sum())
                            if n_b == 0:
                                continue

                            n_pos = int((y_test[m] == 1).sum())
                            n_neg = int((y_test[m] == 0).sum())
                            w_pos = float(w_test[m][y_test[m] == 1].sum())
                            w_neg = float(w_test[m][y_test[m] == 0].sum())
                            print(f"[A1 sim-bin counts] bin={b:>8} | n={n_b:,} | n_pos={n_pos:,} n_neg={n_neg:,} | w_pos={w_pos:.3f} w_neg={w_neg:.3f}")

                            b_tag = _slug(b)

                            h = _eval_and_write(
                                run_dir=RUN_DIR,
                                split_name=f"internal_test__enz_unseen__sim_{b_tag}",
                                df_part=df_test_full.loc[m].reset_index(drop=True),
                                y=y_test[m], w=w_test[m], p=p_test[m],
                                thresholds=thresholds_to_report,
                                prefix=f"trackA_internal/test_quadrants/sim_bins/{b_tag}"
                            )

                            frac = n_b / max(1, len(df_test_full))
                            print(
                                f"[A1 sim-bin] bin={b:>8} | n={h['n']:,} ({frac:.3f} of test) | "
                                f"pos_w={h['pos_rate_weighted']:.3f} | " + _fmt_track_line(h, prefer=prefer_thr)
                            )

                            row = dict(
                                sim_bin=str(b),
                                sim_bin_tag=str(b_tag),
                                n=int(h["n"]),
                                frac_of_test=float(frac),
                                pos_rate_weighted=float(h["pos_rate_weighted"]),
                                auroc_weighted=float(h["auroc_weighted"]),
                                ap_weighted=float(h["ap_weighted"]),
                                brier_weighted=float(h["brier_weighted"]),
                                log_loss_weighted=float(h["log_loss_weighted"]),
                            )

                            thr = h.get("thresholded") or {}
                            if isinstance(thr, dict) and (prefer_thr in thr):
                                row["threshold_used"] = float(thr[prefer_thr].get("threshold", np.nan))
                                row["f1_weighted"]    = float(thr[prefer_thr].get("rates", {}).get("f1", np.nan))

                            sim_bin_rows.append(row)

                    finally:
                        EVAL_DO_PER_ENZYME = _prev_per_enz

                    if sim_bin_rows:
                        sim_df = pd.DataFrame(sim_bin_rows).sort_values("sim_bin_tag")
                        out_dir = _ensure_dir(RUN_DIR / "trackA_internal" / "test_quadrants" / "sim_bins")
                        sim_df.to_csv(out_dir / "sim_bin_summary.csv", index=False)

                        print("\n[A1 sim-bin] Summary (sorted):")
                        cols_show = [c for c in [
                            "sim_bin","n","frac_of_test","pos_rate_weighted","auroc_weighted","ap_weighted","threshold_used","f1_weighted"
                        ] if c in sim_df.columns]
                        print(sim_df[cols_show].to_string(index=False))
                    else:
                        print("[A1 sim-bin] No bins had any rows; nothing written.")

                    print("[A1 novelty] Wrote similarity-bin reports under trackA_internal/test_quadrants/sim_bins/")

        # -----------------------------
        # SANITY CHECKS (ablations + permutation test) — resume-capable
        # -----------------------------
        sanity_fp = RUN_DIR / "trackA_internal" / "sanity_checks.json"
        sanity = dict(
            universe=universe_tag,
            emb_tag=emb_tag,
            split_json=str(split_json),
            stamp=_now_tag(),
            substrate_seen_unseen=sub_breakdown,
            baseline_full=headline_full,
            ablations=None,
            permutation_test=None,
        )

        if DO_SANITY_CHECKS:
            sanity_root = _ensure_dir(RUN_DIR / "trackA_internal" / "sanity")
            (sanity_root / "README.txt").write_text(
                "Sanity checks for Track A:\n"
                "- Ablations: enzyme_only vs substrate_only (trained on same TRAIN split)\n"
                "- Permutation test: shuffle enzyme embeddings within TEST (substrates fixed)\n"
            )

            # ----- A) Ablations -----
            if DO_SANITY_ABLATIONS:
                ablations = {}
                params_sanity = _cap_params_for_sanity(FROZEN_PARAMS, SANITY_ABLATION_N_ESTIMATORS_CAP)

                for mode in ["enzyme_only", "substrate_only"]:
                    out_eval = RUN_DIR / "trackA_internal" / "sanity" / mode
                    ok_mode, _ = _quiet_bundle_ok(out_eval) if out_eval.exists() else (False, None)

                    if ok_mode:
                        h = _read_headline_json(out_eval)
                    else:
                        Xtr = _build_X_mode(pairs.iloc[train_ids], embs, fps, mode=mode)
                        Xte = _build_X_mode(pairs.iloc[test_ids],  embs, fps, mode=mode)

                        clf_m = _fit_xgb(Xtr, y_train, w_train, params_sanity, seed=int(seed) + (11 if mode=="enzyme_only" else 22))
                        pte_m = clf_m.predict_proba(Xte)[:, 1]

                        h = _eval_and_write(
                            run_dir=RUN_DIR,
                            split_name=f"internal_test__{mode}",
                            df_part=df_test_full,
                            y=y_test, w=w_test, p=pte_m,
                            thresholds=thresholds_to_report,
                            prefix=f"trackA_internal/sanity/{mode}"
                        )

                    if isinstance(h, dict):
                        ablations[mode] = dict(
                            params_used=params_sanity,
                            headline=h,
                        )
                        print(f"[Sanity] Ablation {mode}: AUROC={h['auroc_weighted']:.3f} | AP={h['ap_weighted']:.3f}")

                ablations["full_baseline"] = dict(
                    params_used=FROZEN_PARAMS,
                    headline=headline_full,
                )
                sanity["ablations"] = ablations

            # ----- B) Permutation test -----
            if DO_SANITY_PERMUTE_TEST:
                out_eval = RUN_DIR / "trackA_internal" / "sanity" / "permute_enz_test"
                ok_perm, _ = _quiet_bundle_ok(out_eval) if out_eval.exists() else (False, None)

                if ok_perm:
                    h_perm = _read_headline_json(out_eval)
                    if isinstance(h_perm, dict) and isinstance(headline_full, dict):
                        delta = dict(
                            delta_auroc=float(h_perm["auroc_weighted"] - headline_full["auroc_weighted"]),
                            delta_ap=float(h_perm["ap_weighted"] - headline_full["ap_weighted"]),
                            delta_logloss=float(h_perm["log_loss_weighted"] - headline_full["log_loss_weighted"]),
                            delta_brier=float(h_perm["brier_weighted"] - headline_full["brier_weighted"]),
                        )
                        sanity["permutation_test"] = dict(
                            seed=int(seed) + 999,
                            headline=h_perm,
                            delta_vs_full=delta,
                        )
                        print(f"[Sanity] Permute enzymes in TEST: AUROC={h_perm['auroc_weighted']:.3f} | AP={h_perm['ap_weighted']:.3f} "
                              f"| ΔAP={delta['delta_ap']:+.3f}")
                else:
                    if not {"enz_idx", "sub_idx"}.issubset(df_test_full.columns):
                        print("[Sanity] Permutation test skipped: missing enz_idx/sub_idx in test.")
                    else:
                        rng = np.random.default_rng(int(seed) + 999)
                        enz_test_idx = df_test_full["enz_idx"].to_numpy(dtype=int)
                        sub_test_idx = df_test_full["sub_idx"].to_numpy(dtype=int)
                        perm = rng.permutation(len(df_test_full))
                        enz_perm_idx = enz_test_idx[perm]
                        X_test_perm = np.hstack([embs[enz_perm_idx], fps[sub_test_idx]])
                        dm_perm = xgb.DMatrix(X_test_perm)
                        p_test_perm = booster.predict(dm_perm)

                        h_perm = _eval_and_write(
                            run_dir=RUN_DIR,
                            split_name="internal_test__permute_enz",
                            df_part=df_test_full.assign(enz_idx_shuffled=enz_perm_idx),
                            y=y_test, w=w_test, p=p_test_perm,
                            thresholds=thresholds_to_report,
                            prefix="trackA_internal/sanity/permute_enz_test"
                        )

                        delta = dict(
                            delta_auroc=float(h_perm["auroc_weighted"] - headline_full["auroc_weighted"]),
                            delta_ap=float(h_perm["ap_weighted"] - headline_full["ap_weighted"]),
                            delta_logloss=float(h_perm["log_loss_weighted"] - headline_full["log_loss_weighted"]),
                            delta_brier=float(h_perm["brier_weighted"] - headline_full["brier_weighted"]),
                        )

                        sanity["permutation_test"] = dict(
                            seed=int(seed) + 999,
                            headline=h_perm,
                            delta_vs_full=delta,
                        )

                        print(f"[Sanity] Permute enzymes in TEST: AUROC={h_perm['auroc_weighted']:.3f} | AP={h_perm['ap_weighted']:.3f} "
                              f"| ΔAP={delta['delta_ap']:+.3f}")

            # write/update sanity summary
            sanity_fp.write_text(json.dumps(sanity, indent=2))
            print("[Sanity] Wrote:", sanity_fp)

        # -----------------------------
        # Write / update run manifest (add cfg_hash + cfg for matching)
        # -----------------------------
        manifest_fp = RUN_DIR / "run_manifest.json"
        old = {}
        if manifest_fp.exists():
            try:
                old = json.loads(manifest_fp.read_text())
            except Exception:
                old = {}

        # Store some extra fields to aid replay/matching
        update = dict(
            run_id=RUN_ID,
            track="A_internal_enzyme_OOD",
            universe=universe_tag,
            pairs_fp=str(pairs_fp),
            n_pairs=int(len(pairs)),
            n_train=int(len(train_ids)),
            n_test=int(len(test_ids)),
            groups_train_source=groups_train_source,

            emb_fp=str(emb_fp),
            fp_fp=str(substrate_fp),
            substrate_kind=str(substrate_kind),
            sim_fp_fp=str(sim_fp_fp),
            split_json=str(split_json),
            split_json_sig=_sha1_file(split_json),

            # HPO provenance
            hpo_source_universe=hpo_source_universe,
            hpo_source_track=hpo_source_track,
            best_params_source_fp=str(bp_fp) if bp_fp else None,
            suggested_n_estimators=meta.get("suggested_n_estimators", None),

            frozen_best_params_source_fp=FROZEN_BP_FP,
            frozen_hpo_source=FROZEN_HPO_SOURCE,
            frozen_params=FROZEN_PARAMS,

            # reporting knobs
            report_binary_metrics=bool(REPORT_BINARY_METRICS),
            default_threshold=float(DEFAULT_THRESHOLD),
            did_oof_threshold=bool(DO_OOF_THRESHOLD),
            oof_threshold_source=str((RUN_DIR / "trackA_internal" / "train_oof" / "thresholds_oof.json")) if DO_OOF_THRESHOLD else None,
            reported_thresholds=thresholds_to_report,

            # extras
            did_sub_seen_unseen_breakdown=bool(DO_SUBSTRATE_SEEN_UNSEEN_BREAKDOWN),
            did_sanity_checks=bool(DO_SANITY_CHECKS),
            sanity_checks_fp=str(sanity_fp) if (DO_SANITY_CHECKS and sanity_fp.exists()) else None,

            internal_test_headline=headline_full,
            cfg=cfg,
            cfg_hash=cfg.get("cfg_hash"),
            console_transcript=str(transcript_fp),
            stamp=_now_tag(),
        )

        merged = dict(old)
        merged.update(update)
        manifest_fp.write_text(json.dumps(merged, indent=2))

        # Backward-compat: create legacy alias run dir name (old 5-field pattern) → points to this run
        if make_legacy_runid_alias and include_substrate_kind_in_runid and (not resume_mode):
            legacy_dir = proj / "metrics" / "runs" / LEGACY_RUN_ID
            if not legacy_dir.exists():
                try:
                    legacy_dir.symlink_to(RUN_DIR, target_is_directory=True)
                except Exception:
                    pass

        print("[Track A] DONE")
        print("  RUN_DIR =", RUN_DIR)
        print("  Model   =", model_fp)
        print("  Headline JSON =", RUN_DIR / "trackA_internal/test/internal_test_headline.json")
        if DO_SANITY_CHECKS and sanity_fp.exists():
            print("  Sanity JSON   =", sanity_fp)

    return RUN_DIR

def _assert_bundle_ok(run_dir: Path):
    run_dir = Path(run_dir)
    with contextlib.redirect_stdout(_io.StringIO()):
        ok, missing = _bundle_smoke_check(run_dir / "trackA_internal" / "test")
    assert ok, f"Bundle check failed (trackA_internal/test): missing={missing} | run_dir={run_dir}"

def need(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)

def _infer_train_test_counts_from_split_json(split_json_fp: Path) -> tuple[int | None, int | None]:
    """
    Best-effort count inference without loading pairs.
    Handles explicit ids in JSON and train_ids_*.npy scheme.
    """
    split_json_fp = Path(split_json_fp)
    if not split_json_fp.exists():
        return None, None
    try:
        obj = json.loads(split_json_fp.read_text())
    except Exception:
        return None, None

    if isinstance(obj, dict) and ("train_ids" in obj) and ("test_ids" in obj):
        try:
            return int(len(obj["train_ids"])), int(len(obj["test_ids"]))
        except Exception:
            return None, None

    stem = str(obj.get("split_name", split_json_fp.stem)).strip() if isinstance(obj, dict) else split_json_fp.stem
    for d in [split_json_fp.parent, SPL]:
        tr_fp = d / f"train_ids_{stem}.npy"
        te_fp = d / f"test_ids_{stem}.npy"
        if tr_fp.exists() and te_fp.exists():
            try:
                tr = np.load(tr_fp)
                te = np.load(te_fp)
                return int(len(tr)), int(len(te))
            except Exception:
                return None, None
    return None, None

def _pick_cv_splitter(y_train, groups_train, n_splits, seed):
    if groups_train is not None:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed), True
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), False

def _get_groups_for_oof(pairs: pd.DataFrame, train_ids: np.ndarray, cv_group_col: Optional[str]):
    if cv_group_col is None:
        return None
    need(cv_group_col in pairs.columns, f"cv_group_col='{cv_group_col}' missing in pairs.")
    g = pairs.loc[train_ids, cv_group_col]
    if pd.api.types.is_numeric_dtype(g):
        return g.fillna(-1).astype(int).to_numpy()
    return g.fillna("NA").astype(str).to_numpy()

def _maybe_print_transcript(run_dir: Path) -> bool:
    fp = Path(run_dir) / "console_transcript.txt"
    if not fp.exists():
        return False
    try:
        print(fp.read_text())
        return True
    except Exception:
        return False

def _trackB_preflight_check_for_legacy_aliases(*, proj: Path, emb_tag: str, eval_specs: list[dict], substrate_kinds=("morgan", "molencoder")):
    """
    Fail early if a substrate-specific Track B run path is a symlink or carries a legacy substrate-agnostic manifest.
    """
    proj = Path(proj)
    bad = []

    for spec in eval_specs:
        universe_tag = str(spec["universe_tag"]).strip()
        split_name = str(spec["split_name"]).strip()
        for substrate_kind in substrate_kinds:
            run_dir = _resolve_trackB_run_dir(
                proj=proj,
                universe_tag=universe_tag,
                split_name=split_name,
                emb_tag=emb_tag,
                substrate_kind=substrate_kind,
            )
            if not (run_dir.exists() or run_dir.is_symlink()):
                continue

            if run_dir.is_symlink():
                bad.append((str(run_dir), "symlinked_run_dir"))
                continue

            man = _read_manifest(run_dir)
            if not man:
                continue

            _new_id, legacy_id = _trackB_run_ids(
                universe_tag=universe_tag,
                split_name=split_name,
                emb_tag=emb_tag,
                substrate_kind=substrate_kind,
            )
            man_run_id = str(man.get("run_id", "")).strip()
            man_sub_kind = str(man.get("substrate_kind", "")).strip()
            man_sub_fp = str(man.get("substrate_fp", "")).strip()

            if man_run_id == legacy_id and (man_sub_kind == "" or man_sub_fp == ""):
                bad.append((str(run_dir), "legacy_manifest_without_substrate_identity"))

    if bad:
        msg = ["Refusing to run Track B with contaminated legacy/aliased substrate-specific caches."]
        msg.append("Move these run directories aside (or delete them) and rerun this cell:")
        for p, reason in bad:
            msg.append(f"  - {p} | reason={reason}")
        raise AssertionError("\n".join(msg))

def run_frozen_eval(
    *,
    universe_tag: str,
    split_name: str,
    split_json_fp: Path,
    cv_group_col: Optional[str],
    emb_tag: str,
    emb_fp: Path,
    substrate_kind: str,
    substrate_fp: Path,
    scaffold_map_fp: Optional[Path] = None,
    do_oof_threshold: bool = False,
    report_binary_metrics: bool = True,
    default_threshold: float = 0.5,
    n_splits_inner: int = 5,
    force: bool = False,
):
    universe_tag = str(universe_tag).strip()
    split_name   = str(split_name).strip()
    emb_tag      = str(emb_tag).strip()
    substrate_kind = str(substrate_kind).strip()

    emb_fp = Path(emb_fp)
    substrate_fp = Path(substrate_fp)
    need(emb_fp.exists(), f"Missing emb_fp: {emb_fp}")
    need(substrate_fp.exists(), f"Missing substrate_fp: {substrate_fp}")

    cfg_hash = _trackB_cfg_hash(
        universe_tag=universe_tag,
        split_name=split_name,
        split_json_fp=Path(split_json_fp),
        cv_group_col=cv_group_col,
        emb_tag=emb_tag,
        emb_fp=emb_fp,
        substrate_kind=substrate_kind,
        substrate_fp=substrate_fp,
        do_oof_threshold=do_oof_threshold,
        report_binary_metrics=report_binary_metrics,
        default_threshold=default_threshold,
        n_splits_inner=n_splits_inner,
        frozen_params=FROZEN_PARAMS,
    )

    # --- strict deterministic run dir (substrate-specific only) ---
    run_dir = _resolve_trackB_run_dir(
        proj=PROJ,
        universe_tag=universe_tag,
        split_name=split_name,
        emb_tag=emb_tag,
        substrate_kind=substrate_kind,
    )
    run_id = run_dir.name
    eval_dir = _cached_eval_dir(run_dir, split_name)
    manifest_fp = run_dir / "manifest.json"

    # ---- cache safety gate ----
    if run_dir.exists() and run_dir.is_symlink():
        raise AssertionError(
            f"[{split_name}] Refusing to use symlinked Track B run_dir: {run_dir}\n"
            "This usually indicates legacy aliasing between substrate representations.\n"
            "Move this path aside and rerun."
        )

    if run_dir.exists() and (not force):
        ok_complete, _ = _trackB_eval_complete(run_dir, split_name)

        if manifest_fp.exists():
            man_ok, man_msg = _trackB_manifest_matches_request(
                run_dir=run_dir,
                universe_tag=universe_tag,
                split_name=split_name,
                emb_tag=emb_tag,
                emb_fp=emb_fp,
                substrate_kind=substrate_kind,
                substrate_fp=substrate_fp,
                cfg_hash=cfg_hash,
                allow_missing_cfg_hash=True,
            )
            need(
                man_ok,
                f"[{split_name}] Refusing cached Track B run_dir due to manifest mismatch:\n"
                f"  run_dir={run_dir}\n"
                f"  reason={man_msg}\n"
                "Move this run directory aside or rerun with force=True."
            )
            _maybe_backfill_trackB_manifest(run_dir, {
                "cfg_hash": cfg_hash,
                "universe_tag": universe_tag,
            })
        elif ok_complete:
            raise AssertionError(
                f"[{split_name}] Refusing complete cached Track B run_dir without manifest.json:\n"
                f"  run_dir={run_dir}\n"
                "This looks like a legacy or contaminated cache. Move it aside and rerun."
            )

    # ---- cache hit gate ----
    if run_dir.exists() and (not force):
        ok, missing = _trackB_eval_complete(run_dir, split_name)
        if ok:
            print(f"[skip] {split_name}: cached run_dir complete: {run_dir}")

            # Transcript replay preferred
            if _maybe_print_transcript(run_dir):
                return run_dir

            # Artifact-based replay
            _print_header_from_manifest_or_fallback(split_name=split_name, run_dir=run_dir, split_json_fp=Path(split_json_fp))
            _bundle_smoke_check(eval_dir)
            headline_cached = _load_cached_headline(run_dir, split_name)
            _print_perf_line(split_name, headline_cached)
            return run_dir

        # incomplete: try repair (no retrain) then re-check
        if eval_dir.exists():
            _repair_eval_bundle_from_preds(eval_dir)
            ok2, missing2 = _trackB_eval_complete(run_dir, split_name)
            if ok2:
                print(f"[skip] {split_name}: cached run_dir complete: {run_dir}")
                if _maybe_print_transcript(run_dir):
                    return run_dir
                _print_header_from_manifest_or_fallback(split_name=split_name, run_dir=run_dir, split_json_fp=Path(split_json_fp))
                _bundle_smoke_check(eval_dir)
                headline_cached = _load_cached_headline(run_dir, split_name)
                _print_perf_line(split_name, headline_cached)
                return run_dir
        # fall through to compute (acceptable fallback)

    # ---- compute / recompute (force or incomplete) ----
    transcript_fp = run_dir / "console_transcript.txt"
    session_header = f"\n\n===== RUN {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ====="

    # If user explicitly forces recompute, allow wipe
    if force and (run_dir.exists() or run_dir.is_symlink()):
        if run_dir.is_symlink():
            run_dir.unlink()
        else:
            shutil.rmtree(run_dir, ignore_errors=True)

    run_dir.mkdir(parents=True, exist_ok=True)

    with TeeStdout(transcript_fp, mode=("a" if transcript_fp.exists() else "w"), session_header=session_header):

        pairs = _load_pairs_universe(universe_tag)
        pairs_fp = Path(pairs.attrs.get("_pairs_fp", "unknown"))

        split_obj = _read_split_obj(Path(split_json_fp))

        # Attach cluster_id_80 if needed by cv_group_col or split group_col
        needs_cluster = (cv_group_col == "cluster_id_80") or (split_obj.get("group_col") == "cluster_id_80")
        if needs_cluster and ("cluster_id_80" not in pairs.columns):
            need(Path(CLUSTERMAP_CSV).exists(), f"Missing CLUSTERMAP_CSV: {CLUSTERMAP_CSV}")
            cmap = pd.read_csv(CLUSTERMAP_CSV)
            cmap["enzyme"] = cmap["enzyme"].astype(str).str.strip()
            need("enzyme" in pairs.columns, "pairs missing 'enzyme' needed to merge cluster map.")
            pairs = pairs.merge(cmap, on="enzyme", how="left")

        # Explicit feature loading (CRITICAL: do NOT call _load_features())
        embs = np.load(emb_fp)
        subs = np.load(substrate_fp)

        label_col, w_col, y_all, w_all = _get_label_and_weight(pairs)

        train_ids, test_ids = _resolve_train_test_ids_from_split_obj(pairs, split_obj, Path(split_json_fp))
        need(len(train_ids) > 0 and len(test_ids) > 0, f"[{split_name}] Empty train/test after applying split.")
        need(len(np.intersect1d(train_ids, test_ids)) == 0, f"[{split_name}] train_ids ∩ test_ids not empty (leakage).")

        # Build data
        X_train = _build_X(pairs.iloc[train_ids], embs, subs)
        y_train = y_all[train_ids]
        w_train = w_all[train_ids]

        X_test  = _build_X(pairs.iloc[test_ids], embs, subs)
        y_test  = y_all[test_ids]
        w_test  = w_all[test_ids]

        # Threshold reporting
        thresholds_to_report = None
        if report_binary_metrics:
            thresholds_to_report = {"t0p5": float(default_threshold)}

        # Optional: OOF threshold (TRAIN only; no test touch)
        if do_oof_threshold:
            groups_train = _get_groups_for_oof(pairs, train_ids, cv_group_col)
            splitter, is_group = _pick_cv_splitter(y_train, groups_train, n_splits_inner, SEED)

            p_oof = np.full(len(train_ids), np.nan, dtype=float)
            split_iter = (
                splitter.split(X_train, y_train, groups=groups_train)
                if is_group else splitter.split(X_train, y_train)
            )

            for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
                clf = _fit_xgb(X_train[tr_idx], y_train[tr_idx], w_train[tr_idx], FROZEN_PARAMS, seed=SEED + fold)
                p_oof[va_idx] = clf.predict_proba(X_train[va_idx])[:, 1]

            need(np.isfinite(p_oof).all(), f"[{split_name}] OOF preds contain NaN/inf (unexpected).")
            t_oof, f1_oof_max, *_ = _weighted_pr_f1_sweep(y_train, p_oof, w_train)
            thresholds_to_report = {"t_oof_f1": float(t_oof), **(thresholds_to_report or {})}

            (run_dir / "oof_threshold.json").write_text(json.dumps({
                "split_name": split_name,
                "criterion": "maximize_weighted_f1_on_train_oof",
                "t_oof_f1": float(t_oof),
                "f1_oof_max": float(f1_oof_max),
                "cv_group_col": cv_group_col,
                "n_splits": int(n_splits_inner),
                "stamp": _now_tag(),
            }, indent=2))

        print(f"\n[{split_name}] pairs={pairs_fp.name} | train={len(train_ids):,} | test={len(test_ids):,} | HPO=frozen(enzyme-OOD)")

        clf = _fit_xgb(X_train, y_train, w_train, FROZEN_PARAMS, seed=SEED)
        p_test = clf.predict_proba(X_test)[:, 1]

        df_test = pairs.iloc[test_ids].reset_index(drop=True)
        headline = _eval_and_write(
            run_dir=run_dir,
            split_name=f"{split_name}__test",
            df_part=df_test,
            y=y_test, w=w_test, p=p_test,
            thresholds=thresholds_to_report,
            prefix=f"trackB/{split_name}/test"
        )

        # Manifest (strict substrate-specific provenance)
        (run_dir / "manifest.json").write_text(json.dumps({
            "run_id": run_id,
            "run_id_new": f"trackB__{universe_tag}__{split_name}__{emb_tag}__{substrate_kind}",
            "run_id_legacy": f"trackB__{universe_tag}__{split_name}__{emb_tag}",
            "universe": universe_tag,
            "universe_tag": universe_tag,
            "split_name": split_name,
            "pairs_fp": str(pairs_fp),
            "pairs_fp_name": str(pairs_fp.name),
            "train_n": int(len(train_ids)),
            "test_n": int(len(test_ids)),
            "eval_dir": str(eval_dir),
            "split_json_fp": str(Path(split_json_fp)),
            "split_json_keys": sorted(list(split_obj.keys())),

            "emb_tag": emb_tag,
            "emb_fp": str(emb_fp),
            "substrate_kind": substrate_kind,
            "substrate_fp": str(substrate_fp),
            "cfg_hash": cfg_hash,

            # frozen provenance (manifest only)
            "frozen_best_params_source_fp": FROZEN_BP_FP,
            "frozen_hpo_source": FROZEN_HPO_SOURCE,
            "hpo_frozen_best_params_fp": FROZEN_BP_FP,
            "hpo_frozen_source": FROZEN_HPO_SOURCE,

            # training recipe
            "frozen_params": FROZEN_PARAMS,

            "thresholds_reported": thresholds_to_report,
            "cv_group_col_for_oof_threshold": cv_group_col,
            "headline": headline,
            "stamp": _now_tag(),
        }, indent=2))

        # Print bundle check lines (to match cache replay)
        _bundle_smoke_check(eval_dir)

        # Acceptance gate: canonical bundle must be complete
        ok, missing = _trackB_eval_complete(run_dir, split_name)
        assert ok, f"[{split_name}] canonical bundle incomplete: missing={missing} | eval_dir={eval_dir}"

        _print_perf_line(split_name, headline)

    return run_dir

def run_trackB_suite(*,
        proj: Path,
        emb_tag: str,
        emb_fp: Path,
        substrate_kind: str,      # "morgan" or "molencoder"
        substrate_fp: Path,
        eval_specs: list[dict],
        force: bool,
        report_binary_metrics: bool,
        do_oof_threshold: bool,
        default_threshold: float,
        n_splits_inner: int,
     ) -> Dict[str, str]:
    global PROJ
    PROJ = Path(proj)

    substrate_kind = str(substrate_kind).strip()
    emb_tag = str(emb_tag).strip()
    emb_fp = Path(emb_fp)
    substrate_fp = Path(substrate_fp)

    # required console header
    print(f"[Track B] SUBSTRATE_REP={substrate_kind} | substrate_fp={substrate_fp} | SUBREP_TAG={substrate_kind}")

    out = {}
    for spec in eval_specs:
        run_dir = run_frozen_eval(
            universe_tag=spec["universe_tag"],
            split_name=spec["split_name"],
            split_json_fp=Path(spec["split_json_fp"]),
            cv_group_col=spec.get("cv_group_col"),
            emb_tag=emb_tag,
            emb_fp=emb_fp,
            substrate_kind=substrate_kind,
            substrate_fp=substrate_fp,
            scaffold_map_fp=spec.get("scaffold_map_fp"),
            do_oof_threshold=do_oof_threshold,
            report_binary_metrics=report_binary_metrics,
            default_threshold=default_threshold,
            n_splits_inner=n_splits_inner,
            force=force,
        )
        out[spec["split_name"]] = str(run_dir)

    print("\n[Track B] DONE. Run dirs:")
    print(json.dumps(out, indent=2))
    return out

def _assert_eval_dir_ok(run_dir: Path, split_name: str):
    run_dir = Path(run_dir)
    ok, missing = _trackB_eval_complete(run_dir, split_name)
    assert ok, f"[accept] {split_name}: bundle incomplete: missing={missing} | run_dir={run_dir}"
    ed = _cached_eval_dir(run_dir, split_name)
    assert (ed / "headline.json").exists(), f"[accept] missing headline.json: {ed/'headline.json'}"
    assert (ed / "preds.csv").exists(), f"[accept] missing preds.csv: {ed/'preds.csv'}"

def _normalize_ext_tags_by_universe(d: dict[str, list[str]] | None) -> dict[str, list[str]]:
    return {str(k): [str(x) for x in v] for k, v in (d or {}).items()}

def _pair_key(df: pd.DataFrame) -> pd.Series:
    """
    Stable key for a pair. Priority:
      (enz_idx, sub_idx)  >  pair_id  >  (enzyme, acceptor)
    """
    cols = set(df.columns)

    if {"enz_idx", "sub_idx"}.issubset(cols):
        e = pd.to_numeric(df["enz_idx"], errors="coerce").astype("Int64")
        s = pd.to_numeric(df["sub_idx"], errors="coerce").astype("Int64")
        return e.astype(str) + "__" + s.astype(str)

    if "pair_id" in cols:
        return df["pair_id"].astype(str)

    if {"enzyme", "acceptor"}.issubset(cols):
        enz = df["enzyme"].astype(str).str.strip()
        acc = df["acceptor"].astype(str).str.strip()
        return enz + "__" + acc

    raise AssertionError(
        "Cannot form pair key. Need (enz_idx,sub_idx) or pair_id or (enzyme,acceptor). "
        f"Columns={sorted(cols)}"
    )

def _enzyme_key(df: pd.DataFrame):
    return df["enzyme"].astype(str).str.strip() if "enzyme" in df.columns else None

def _audit_overlap(pairs_u: pd.DataFrame, df_ext: pd.DataFrame) -> dict:
    k_tr = _pair_key(pairs_u)
    k_ex = _pair_key(df_ext)

    tr_idx = pd.Index(k_tr.dropna().unique())
    ex_idx = pd.Index(k_ex.dropna().unique())
    ov_pairs = tr_idx.intersection(ex_idx)

    audit = dict(
        n_train_pairs=int(len(tr_idx)),
        n_ext_pairs=int(len(ex_idx)),
        n_pair_overlap=int(len(ov_pairs)),
        frac_ext_pairs_overlapping=float(len(ov_pairs) / max(1, len(ex_idx))),
    )

    enz_tr = _enzyme_key(pairs_u)
    enz_ex = _enzyme_key(df_ext)
    if (enz_tr is not None) and (enz_ex is not None):
        et = pd.Index(enz_tr.dropna().unique())
        ee = pd.Index(enz_ex.dropna().unique())
        ov_enz = et.intersection(ee)
        audit.update(dict(
            n_train_enzymes=int(len(et)),
            n_ext_enzymes=int(len(ee)),
            n_enzyme_overlap=int(len(ov_enz)),
            frac_ext_enzymes_overlapping=float(len(ov_enz) / max(1, len(ee))),
        ))
    else:
        audit.update(dict(
            n_train_enzymes=None,
            n_ext_enzymes=None,
            n_enzyme_overlap=None,
            frac_ext_enzymes_overlapping=None,
        ))

    audit["_overlap_pair_keys"] = set(map(str, ov_pairs.tolist()))
    return audit

def _cm_rates_from_weighted_counts(tn, fp, fn, tp):
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1  = (2*ppv*tpr/(ppv+tpr)) if (ppv+tpr) else 0.0
    bal = 0.5*(tpr+tnr)
    denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    mcc = ((tp*tn - fp*fn)/math.sqrt(denom)) if denom > 0 else 0.0
    return dict(precision=ppv, recall=tpr, f1=f1, balanced_accuracy=bal, mcc=mcc, tnr=tnr)

def _threshold_report(y, p, w, t):
    yhat = (p >= t).astype(int)
    cm_w = confusion_matrix(y, yhat, labels=[0,1], sample_weight=w)
    tn, fp, fn, tp = map(float, cm_w.ravel())
    return dict(
        threshold=float(t),
        counts_w=dict(tn=tn, fp=fp, fn=fn, tp=tp),
        rates=_cm_rates_from_weighted_counts(tn, fp, fn, tp),
        cm_w=cm_w,
        yhat=yhat,
    )

def _weighted_pr_f1_sweep_fallback(y, p, w):
    prec, rec, thr = precision_recall_curve(y, p, sample_weight=w)
    prec_t, rec_t = prec[:-1], rec[:-1]
    f1 = (2.0 * prec_t * rec_t) / (prec_t + rec_t + 1e-12)
    best_i = int(np.argmax(f1))
    return float(thr[best_i]), float(f1[best_i]), prec_t, rec_t, thr, f1

def _flatten_row(context, evald, prefer_key="t_train_oof"):
    row = dict(
        universe=context["universe"],
        ext_dataset=context["ext_dataset"],
        emb_tag=context["emb_tag"],
        model_fp=context["model_fp"],
        best_params_source=context.get("best_params_source"),
        threshold_policy=context.get("threshold_policy"),
        n=evald.get("n", np.nan),
        weight_sum=evald.get("weight_sum", np.nan),
        pos_rate_weighted=evald.get("pos_rate_weighted", np.nan),
        auroc_weighted=evald.get("auroc_weighted", np.nan),
        ap_weighted=evald.get("ap_weighted", np.nan),
        log_loss_weighted=evald.get("log_loss_weighted", np.nan),
        brier_weighted=evald.get("brier_weighted", np.nan),
    )
    thr = evald.get("thresholded") or {}
    key = prefer_key if prefer_key in thr else ("t0p5" if "t0p5" in thr else (next(iter(thr.keys())) if thr else None))
    row["threshold_key"] = key
    if key is None:
        row.update(dict(threshold=np.nan, f1=np.nan, precision=np.nan, recall=np.nan, mcc=np.nan))
    else:
        row["threshold"] = thr[key].get("threshold", np.nan)
        rates = thr[key].get("rates", {}) or {}
        row["f1"] = rates.get("f1", np.nan)
        row["precision"] = rates.get("precision", np.nan)
        row["recall"] = rates.get("recall", np.nan)
        row["mcc"] = rates.get("mcc", np.nan)
    return row

def _fmt_line(evald, prefer_key="t_train_oof"):
    auroc = evald.get("auroc_weighted", float("nan"))
    ap    = evald.get("ap_weighted", float("nan"))
    thr = evald.get("thresholded") or {}
    key = prefer_key if prefer_key in thr else ("t0p5" if "t0p5" in thr else (next(iter(thr.keys())) if thr else None))
    if key is None:
        return f"AUROC={auroc:.3f} | AP={ap:.3f}"
    t = thr[key].get("threshold", float("nan"))
    f1 = (thr[key].get("rates", {}) or {}).get("f1", float("nan"))
    return f"AUROC={auroc:.3f} | AP={ap:.3f} | {key}={t:.3f} | F1={f1:.3f}"

def _sf(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def _fmt(x, nd=3):
    x = _sf(x)
    return "nan" if not np.isfinite(x) else f"{x:.{nd}f}"

def _fmt_delta(x, base, nd=3):
    x = _sf(x); base = _sf(base)
    if not (np.isfinite(x) and np.isfinite(base)):
        return "nan"
    d = x - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.{nd}f}"

def _print_sanity(U, ext, sanity_summary: dict):
    if (not PRINT_SANITY) or (not isinstance(sanity_summary, dict)):
        return

    base_ap  = _sf((sanity_summary.get("baseline") or {}).get("ap", np.nan))
    base_auc = _sf((sanity_summary.get("baseline") or {}).get("auroc", np.nan))

    ab = sanity_summary.get("ablations") or {}
    pe = sanity_summary.get("permutations") or {}
    su = sanity_summary.get("seen_unseen_2x2") or {}

    ap_enz  = _sf(((ab.get("enzyme_only") or {}).get("ap", np.nan)))
    ap_sub  = _sf(((ab.get("substrate_only") or {}).get("ap", np.nan)))
    ap_penz = _sf(((pe.get("permute_enz") or {}).get("ap", np.nan)))
    ap_psub = _sf(((pe.get("permute_sub") or {}).get("ap", np.nan)))

    def _su_get(name):
        v = su.get(name)
        if isinstance(v, dict):
            return int(v.get("n", 0)), _sf(v.get("ap", np.nan))
        return int(v) if v is not None else 0, float("nan")

    n_ss, ap_ss = _su_get("E_seen__S_seen")
    n_su, ap_su = _su_get("E_seen__S_unseen")
    n_us, ap_us = _su_get("E_unseen__S_seen")
    n_uu, ap_uu = _su_get("E_unseen__S_unseen")

    if SANITY_PRINT_STYLE == "line":
        msg = (
            f"[Sanity] {U} → ext_{ext} | "
            f"AP(full)={_fmt(base_ap)} AUROC={_fmt(base_auc)} | "
            f"AP(enz)={_fmt(ap_enz)}(Δ{_fmt_delta(ap_enz, base_ap)}) "
            f"AP(sub)={_fmt(ap_sub)}(Δ{_fmt_delta(ap_sub, base_ap)}) | "
            f"AP(permE)={_fmt(ap_penz)}(Δ{_fmt_delta(ap_penz, base_ap)}) "
            f"AP(permS)={_fmt(ap_psub)}(Δ{_fmt_delta(ap_psub, base_ap)}) | "
            f"2x2 n=[SS:{n_ss}, SU:{n_su}, US:{n_us}, UU:{n_uu}]"
        )
        if SANITY_PRINT_SHOW_2x2_AP:
            msg += f" AP=[SS:{_fmt(ap_ss)}, SU:{_fmt(ap_su)}, US:{_fmt(ap_us)}, UU:{_fmt(ap_uu)}]"
        print(msg)
        return

    print(f"[Sanity] {U} → ext_{ext}")
    print(f"  baseline : AP={_fmt(base_ap)} | AUROC={_fmt(base_auc)}")
    if ab:
        print(f"  ablation : enzyme_only AP={_fmt(ap_enz)} (Δ{_fmt_delta(ap_enz, base_ap)}) | "
              f"substrate_only AP={_fmt(ap_sub)} (Δ{_fmt_delta(ap_sub, base_ap)})")
    if pe:
        print(f"  permute  : permute_enz AP={_fmt(ap_penz)} (Δ{_fmt_delta(ap_penz, base_ap)}) | "
              f"permute_sub AP={_fmt(ap_psub)} (Δ{_fmt_delta(ap_psub, base_ap)})")
    if su:
        print(f"  2x2 n    : SS={n_ss} | SU={n_su} | US={n_us} | UU={n_uu}")
        if SANITY_PRINT_SHOW_2x2_AP:
            print(f"  2x2 AP   : SS={_fmt(ap_ss)} | SU={_fmt(ap_su)} | US={_fmt(ap_us)} | UU={_fmt(ap_uu)}")

def replay_trackB_external_run(*, run_dir: Path, universe_tags: list[str], ext_tags: list[str], ext_tags_by_universe: dict, prefer_key: str) -> None:
    run_dir = Path(run_dir)
    track_root = run_dir / "trackB_external"
    assert track_root.exists(), f"Cached mode requires track_root exists: {track_root}"

    ok_complete, missing = _trackB_external_run_complete(
        run_dir,
        universe_tags=universe_tags,
        ext_tags=ext_tags,
        ext_tags_by_universe=ext_tags_by_universe,
    )
    assert ok_complete, f"Cached external Track B run is incomplete: run_dir={run_dir} | missing={missing}"

    best_params_source = str(FROZEN_BP_FP) if globals().get("FROZEN_BP_FP", None) else "FROZEN_PARAMS"

    for U in universe_tags:
        # Print universe header (load pairs only to recover fp name + rows; not training)
        pairs_u = _load_pairs_universe(U)
        pairs_fp = Path(pairs_u.attrs.get("_pairs_fp", "unknown"))
        print(f"\n[Track B] Universe={U} | pairs={pairs_fp.name} | rows={len(pairs_u):,}")
        print(f"[Track B] Using FROZEN_PARAMS from Track A: best_params_source={best_params_source}")

        # If train-oof threshold exists (and the current config requests it), replay that line too.
        if REPORT_BINARY_METRICS and DO_TRAIN_OOF_THRESHOLD:
            tj = track_root / U / "train_oof" / "threshold_train_oof.json"
            if tj.exists():
                try:
                    obj = json.loads(tj.read_text())
                    t = float(obj.get("t_train_oof", np.nan))
                    if np.isfinite(t):
                        print(f"[Track B] Universe={U}: frozen t_train_oof={t:.3f} (maxF1 on TRAIN-OOF)")
                except Exception:
                    pass

        ext_tags_this_U = ext_tags_by_universe.get(U, ext_tags)
        print(f"[Track B] Universe={U}: evaluating ext sets = {ext_tags_this_U}")

        for ext in ext_tags_this_U:
            out_dir = track_root / U / f"ext_{ext}"
            if not out_dir.exists():
                print(f"[warn] Missing cached ext output dir: {out_dir} (skipping)")
                continue

            preds_fp = out_dir / "preds.csv"
            assert preds_fp.exists(), f"Cached ext dir missing preds.csv: {preds_fp}"

            # Overlap one-liner (if available)
            audit_fp = out_dir / "overlap_audit.json"
            if audit_fp.exists():
                try:
                    audit = json.loads(audit_fp.read_text())
                    msg = (f"[Overlap] {U} vs ext_{ext}: "
                           f"pair_overlap={audit.get('n_pair_overlap')}/{audit.get('n_ext_pairs')} "
                           f"({float(audit.get('frac_ext_pairs_overlapping', 0.0)):.3%} of ext pairs)")
                    if audit.get("n_enzyme_overlap") is not None:
                        msg += (f" | enzyme_overlap={audit.get('n_enzyme_overlap')}/{audit.get('n_ext_enzymes')} "
                                f"({float(audit.get('frac_ext_enzymes_overlapping', 0.0)):.3%} of ext enzymes)")
                    print("".join(msg))
                except Exception:
                    pass

            # Determine label presence + eval dict
            evald = None
            labels_present = None

            summ_fp = out_dir / "summary.json"
            bund_fp = out_dir / "bundle.json"
            if summ_fp.exists():
                try:
                    summ = json.loads(summ_fp.read_text())
                    if isinstance(summ, dict) and ("eval" in summ):
                        labels_present = (summ.get("eval") is not None)
                        if labels_present and isinstance(summ.get("eval"), dict):
                            evald = summ.get("eval")
                except Exception:
                    pass
            if labels_present is None and bund_fp.exists():
                try:
                    bund = json.loads(bund_fp.read_text())
                    if isinstance(bund, dict) and ("eval" in bund):
                        labels_present = (bund.get("eval") is not None)
                        if labels_present and isinstance(bund.get("eval"), dict):
                            evald = bund.get("eval")
                except Exception:
                    pass
            if labels_present is None:
                # fallback: check preds.csv columns
                try:
                    df0 = pd.read_csv(preds_fp, nrows=5)
                    labels_present = ("y_true" in df0.columns)
                except Exception:
                    labels_present = False

            if not labels_present:
                print(f"[Track B] {U} → ext_{ext}: labels missing → wrote preds only (ranking use-case).")
                counts_fp = out_dir / "sanity_seen_unseen_counts.json"
                if counts_fp.exists():
                    try:
                        quad_counts = json.loads(counts_fp.read_text())
                        print(f"[Sanity] {U} → ext_{ext} (labels-missing) seen/unseen counts:", quad_counts)
                    except Exception:
                        pass
                continue

            # labeled: prefer eval.json, else eval from summary/bundle
            ev_fp = out_dir / "eval.json"
            if ev_fp.exists():
                try:
                    evald = json.loads(ev_fp.read_text())
                except Exception:
                    pass
            if not isinstance(evald, dict):
                # final fallback: if summary/bundle had eval dict already, keep it; else warn
                if not isinstance(evald, dict):
                    print(f"[warn] Missing cached eval for {U} → ext_{ext} (skipping metrics print)")
                    continue

            print(f"[Track B] {U} → ext_{ext}: " + _fmt_line(evald, prefer_key=prefer_key))

            # Optional sanity replay
            ss_fp = out_dir / "sanity" / "sanity_summary.json"
            if ss_fp.exists():
                try:
                    ss = json.loads(ss_fp.read_text())
                    _print_sanity(U, ext, ss)
                except Exception:
                    pass

    # end universes/ext loops
    sum_fp = run_dir / "trackB_external_summary.csv"
    if sum_fp.exists():
        print("\n[Track B] Wrote summary CSV:", sum_fp)
    sanity_fp = run_dir / "trackB_external_sanity_index.csv"
    if sanity_fp.exists():
        print("[Track B] Wrote sanity index CSV:", sanity_fp)

    print("\n[Track B] DONE")
    print("  RUN_DIR =", run_dir)

def _compute_thresholded(y, p, w, *,
                         do_binary_metrics: bool,
                         default_threshold: float,
                         t_train_oof: float | None,
                         do_oracle: bool,
                         plots_dir: Path | None,
                         title_prefix: str):
    """
    Returns: (thresholded_dict_or_None, reps_for_plots_dict, threshold_policy_str)
    Oracle (if enabled) is explicitly labeled.
    """
    thresholded = None
    reps_for_plots = {}
    threshold_policy = "none"

    if not do_binary_metrics:
        return None, {}, threshold_policy

    thresholded = {}
    threshold_policy = "fixed0p5"

    rep0 = _threshold_report(y, p, w, float(default_threshold))
    thresholded["t0p5"] = dict(threshold=rep0["threshold"], counts_w=rep0["counts_w"], rates=rep0["rates"])
    reps_for_plots["t0p5"] = rep0

    if (t_train_oof is not None) and np.isfinite(t_train_oof):
        rep1 = _threshold_report(y, p, w, float(t_train_oof))
        thresholded["t_train_oof"] = dict(threshold=rep1["threshold"], counts_w=rep1["counts_w"], rates=rep1["rates"])
        reps_for_plots["t_train_oof"] = rep1
        threshold_policy = "fixed0p5+train_oof"

    if do_oracle:
        t_or, f1_or, prec_t, rec_t, thr_grid, f1_grid = _weighted_pr_f1_sweep(y, p, w)
        rep_or = _threshold_report(y, p, w, float(t_or))
        thresholded["t_oracle_ext_maxf1"] = dict(
            threshold=rep_or["threshold"],
            counts_w=rep_or["counts_w"],
            rates=rep_or["rates"],
            f1_max=float(f1_or),
        )
        reps_for_plots["t_oracle_ext_maxf1"] = rep_or
        threshold_policy = threshold_policy + "+oracle_ext_maxf1"

        if plots_dir is not None:
            plt.figure()
            plt.plot(thr_grid, f1_grid)
            plt.axvline(t_or, linestyle="--")
            plt.xlabel("Threshold"); plt.ylabel("F1 (weighted via PR curve)")
            plt.title(f"{title_prefix}: F1 vs threshold | oracle t*={t_or:.3f}")
            plt.tight_layout()
            plt.savefig(plots_dir / "f1_vs_threshold__oracle.png", dpi=160)
            plt.close()

    return thresholded, reps_for_plots, threshold_policy

def _permute_probs_full_model(clf, df_ext: pd.DataFrame, embs: np.ndarray, fps: np.ndarray, which: str, seed: int):
    rng = np.random.default_rng(int(seed))
    enz_idx = df_ext["enz_idx"].to_numpy(dtype=int)
    sub_idx = df_ext["sub_idx"].to_numpy(dtype=int)
    perm = rng.permutation(len(df_ext))
    if which == "enz":
        enz_idx = enz_idx[perm]
    elif which == "sub":
        sub_idx = sub_idx[perm]
    else:
        raise ValueError("which must be 'enz' or 'sub'")
    Xp = np.hstack([embs[enz_idx], fps[sub_idx]])
    return clf.predict_proba(Xp)[:, 1]

def _seen_unseen_masks(df_train: pd.DataFrame, df_ext: pd.DataFrame):
    train_enz = set(df_train["enz_idx"].astype(int).tolist())
    train_sub = set(df_train["sub_idx"].astype(int).tolist())
    e_seen = df_ext["enz_idx"].astype(int).isin(train_enz).to_numpy()
    s_seen = df_ext["sub_idx"].astype(int).isin(train_sub).to_numpy()
    return e_seen, s_seen

def run_trackB_external_benchmarking(*,
        proj: Path,
        emb_tag: str,
        emb_fp: Path,
        substrate_kind: str,     # "morgan" or "molencoder"
        substrate_fp: Path,      # MODEL substrate features
        universe_tags: list[str],
        ext_tags: list[str],
        ext_tags_by_universe: dict[str, list[str]],
        force: bool,
        seed: int,
        n_splits_inner: int,
        include_substrate_kind_in_runid: bool = True,
     ) -> Path:

    # ensure helpers that rely on global PROJ work
    global PROJ
    PROJ = Path(proj)

    emb_tag = str(emb_tag).strip()
    substrate_kind = str(substrate_kind).strip()
    emb_fp = Path(emb_fp)
    substrate_fp = Path(substrate_fp)

    assert emb_fp.exists(), f"Missing emb_fp: {emb_fp}"
    assert substrate_fp.exists(), f"Missing substrate_fp: {substrate_fp}"

    sanity_flags = dict(
        ablations=bool(SANITY_DO_ABLATIONS),
        permutations=bool(SANITY_DO_PERMUTATIONS),
        seen_unseen_2x2=bool(SANITY_DO_SEEN_UNSEEN_2x2),
        sanity_n_estimators_cap=SANITY_N_ESTIMATORS_CAP,
        sanity_print=bool(PRINT_SANITY),
        sanity_print_style=str(SANITY_PRINT_STYLE),
        sanity_print_show_2x2_ap=bool(SANITY_PRINT_SHOW_2x2_AP),
    )

    cfg_hash = _trackB_external_cfg_hash(
        emb_tag=emb_tag,
        emb_fp=emb_fp,
        substrate_kind=substrate_kind,
        substrate_fp=substrate_fp,
        universe_tags=universe_tags,
        ext_tags=ext_tags,
        ext_tags_by_universe=ext_tags_by_universe,
        report_binary_metrics=REPORT_BINARY_METRICS,
        default_threshold=DEFAULT_THRESHOLD,
        do_train_oof_threshold=DO_TRAIN_OOF_THRESHOLD,
        do_ext_oracle_maxf1=DO_EXT_ORACLE_MAXF1,
        do_overlap_audit=DO_OVERLAP_AUDIT,
        filter_overlap_from_ext=FILTER_OVERLAP_FROM_EXT,
        do_trackb_sanity=DO_TRACKB_SANITY,
        sanity_flags=sanity_flags,
    )

    # Explicit feature loading (CRITICAL: do NOT call _load_features())
    embs = np.load(emb_fp)
    subs = np.load(substrate_fp)

    # Run directory (scoped)
    if include_substrate_kind_in_runid:
        RUN_ID  = f"trackB__external__{emb_tag}__{substrate_kind}__{_now_tag()}"
    else:
        RUN_ID  = f"trackB__external__{emb_tag}__{_now_tag()}"
    RUN_DIR = _ensure_dir(PROJ / "metrics" / "runs" / RUN_ID)

    if (RUN_DIR.exists() and any(RUN_DIR.iterdir()) and (not force)):
        print(f"[skip] RUN_DIR already populated (FORCE=False): {RUN_DIR}")
        globals()["RUN_ID"] = RUN_ID
        globals()["RUN_DIR"] = RUN_DIR
        return RUN_DIR

    summary_rows = []
    sanity_rows  = []  # optional compact sanity summaries

    best_params = dict(FROZEN_PARAMS)
    best_params_source = str(FROZEN_BP_FP) if FROZEN_BP_FP else "FROZEN_PARAMS"

    for U in universe_tags:
        # 2a) Load training universe and fit on ALL rows
        pairs_u = _load_pairs_universe(U)
        pairs_fp = Path(pairs_u.attrs.get("_pairs_fp", "unknown"))
        print(f"\n[Track B] Universe={U} | pairs={pairs_fp.name} | rows={len(pairs_u):,}")

        # Console continuity: one-line "best params" provenance per universe
        print(f"[Track B] Using FROZEN_PARAMS from Track A: best_params_source={best_params_source}")

        label_col, w_col, y_u, w_u = _get_label_and_weight(pairs_u)
        X_u = _build_X(pairs_u, embs, subs)

        clf = _fit_xgb(X_u, y_u, w_u, best_params, seed=int(seed))

        # save model
        model_dir = _ensure_dir(RUN_DIR / "trackB_external" / U / "model")
        model_fp = model_dir / "model.json"
        clf.get_booster().save_model(str(model_fp))

        # -----------------------------
        # Optional: frozen threshold from TRAIN UNIVERSE only (OOF)
        # -----------------------------
        t_train_oof = None
        f1_train_oof = None
        if REPORT_BINARY_METRICS and DO_TRAIN_OOF_THRESHOLD:
            cv = StratifiedGroupKFold(n_splits=int(n_splits_inner), shuffle=True, random_state=int(seed))
            p_oof_u = np.full(len(y_u), np.nan, dtype=float)

            if "cluster_id_80" in pairs_u.columns and pairs_u["cluster_id_80"].notna().any():
                groups_u = pairs_u["cluster_id_80"].fillna(-1).astype(int).to_numpy()
            elif "enzyme" in pairs_u.columns:
                groups_u = pairs_u["enzyme"].astype(str).to_numpy()
            else:
                groups_u = np.arange(len(pairs_u))

            for fold, (tr_idx, va_idx) in enumerate(cv.split(X_u, y_u, groups=groups_u), 1):
                clf_fold = _fit_xgb(X_u[tr_idx], y_u[tr_idx], w_u[tr_idx], best_params, seed=int(seed) + 100 + fold)
                p_oof_u[va_idx] = clf_fold.predict_proba(X_u[va_idx])[:, 1]

            if not np.isfinite(p_oof_u).all():
                raise AssertionError("Universe OOF prediction contains NaN/inf (unexpected).")

            t_train_oof, f1_train_oof, prec_t, rec_t, thr_grid, f1_grid = _weighted_pr_f1_sweep(y_u, p_oof_u, w_u)
            print(f"[Track B] Universe={U}: frozen t_train_oof={t_train_oof:.3f} (maxF1 on TRAIN-OOF)")

            oof_dir = _ensure_dir(RUN_DIR / "trackB_external" / U / "train_oof")
            np.save(oof_dir / "y_oof.npy", y_u.astype(int))
            np.save(oof_dir / "p_oof.npy", p_oof_u.astype(float))
            np.save(oof_dir / "w_oof.npy", w_u.astype(float))
            pd.DataFrame({"threshold": thr_grid, "precision": prec_t, "recall": rec_t, "f1": f1_grid}).to_csv(
                oof_dir / "threshold_sweep_f1.csv", index=False
            )
            (oof_dir / "threshold_train_oof.json").write_text(json.dumps({
                "universe": U,
                "emb_tag": emb_tag,
                "criterion": "maximize_weighted_f1_on_train_oof",
                "t_train_oof": float(t_train_oof),
                "f1_oof_max": float(f1_train_oof),
                "n_train": int(len(y_u)),
                "stamp": _now_tag(),
            }, indent=2))

        # -----------------------------
        # Track B sanity: train ablation models ONCE per universe
        # -----------------------------
        clf_enz = None
        clf_sub = None
        if DO_TRACKB_SANITY and SANITY_DO_ABLATIONS:
            # CRITICAL: use FROZEN_PARAMS for ALL fits (no per-sanity params)
            X_u_enz = _build_X_mode(pairs_u, embs, subs, mode="enzyme_only")
            X_u_sub = _build_X_mode(pairs_u, embs, subs, mode="substrate_only")

            clf_enz = _fit_xgb(X_u_enz, y_u, w_u, best_params, seed=int(seed) + 11)
            clf_sub = _fit_xgb(X_u_sub, y_u, w_u, best_params, seed=int(seed) + 22)

            sdir = _ensure_dir(RUN_DIR / "trackB_external" / U / "sanity_models")
            clf_enz.get_booster().save_model(str(sdir / "model__enzyme_only.json"))
            clf_sub.get_booster().save_model(str(sdir / "model__substrate_only.json"))
            (sdir / "params_used.json").write_text(json.dumps(best_params, indent=2))

        # 2b) Evaluate on each external dataset (universe-specific)
        ext_tags_this_U = ext_tags_by_universe.get(U, ext_tags)
        print(f"[Track B] Universe={U}: evaluating ext sets = {ext_tags_this_U}")

        train_ref = pairs_u[["enz_idx","sub_idx"]].copy() if {"enz_idx","sub_idx"}.issubset(pairs_u.columns) else None

        for ext in ext_tags_this_U:
            ext_fp = PROJ / "data" / f"ext_{ext}.parquet"
            if not ext_fp.exists():
                print(f"[warn] Missing external dataset: {ext_fp.name} (skipping)")
                continue

            df_ext = pd.read_parquet(ext_fp).reset_index(drop=True)

            out_dir = _ensure_dir(RUN_DIR / "trackB_external" / U / f"ext_{ext}")
            plots   = _ensure_dir(out_dir / "plots")

            # -----------------------------
            # Overlap audit (and OPTIONAL filtering) — filter BEFORE building X/p
            # -----------------------------
            audit = None
            if DO_OVERLAP_AUDIT:
                audit = _audit_overlap(pairs_u, df_ext)

                audit_json = {k: v for k, v in audit.items() if k != "_overlap_pair_keys"}
                audit_json.update(dict(
                    universe=U,
                    ext_dataset=f"ext_{ext}",
                    ext_fp=str(ext_fp),
                    stamp=_now_tag(),
                ))
                (out_dir / "overlap_audit.json").write_text(json.dumps(audit_json, indent=2))

                msg = (f"[Overlap] {U} vs ext_{ext}: "
                      f"pair_overlap={audit['n_pair_overlap']}/{audit['n_ext_pairs']} "
                      f"({audit['frac_ext_pairs_overlapping']:.3%} of ext pairs)")
                if audit.get("n_enzyme_overlap") is not None:
                    msg += (f" | enzyme_overlap={audit['n_enzyme_overlap']}/{audit['n_ext_enzymes']} "
                            f"({audit['frac_ext_enzymes_overlapping']:.3%} of ext enzymes)")
                print(msg)

                if FILTER_OVERLAP_FROM_EXT and audit["n_pair_overlap"] > 0:
                    k_ex = _pair_key(df_ext).astype(str)
                    before = len(df_ext)
                    df_ext = df_ext.loc[~k_ex.isin(audit["_overlap_pair_keys"])].reset_index(drop=True)
                    after = len(df_ext)
                    print(f"[Overlap] FILTER_OVERLAP_FROM_EXT: dropped {before-after} ext rows (now n={after:,}).")

            # allow label-less external sets (ranking only)
            label_col_ext = next((c for c in ["reaction","y","label","class"] if c in df_ext.columns), None)
            w_col_ext = next((c for c in ["weight","sample_weight","w"] if c in df_ext.columns), None)

            # Build X + predict (AFTER filtering)
            X_ext = _build_X(df_ext, embs, subs)
            p_ext = clf.predict_proba(X_ext)[:, 1]

            # keep backwards compat for old downstream cells (and add rep-specific globals)
            if ext == "gasp":
                globals()[f"p_ext_gasp_{substrate_kind}"] = p_ext
                globals()[f"X_ext_gasp_{substrate_kind}"] = X_ext
                globals()[f"clf_{substrate_kind}"] = clf
                # best-effort: keep old unsuffixed names pointing to most recent run
                globals()["p_ext_gasp"] = p_ext
                globals()["X_ext_gasp"] = X_ext
                globals()["clf"] = clf

            y_ext = None
            w_ext = None
            if label_col_ext is not None:
                y_ext = df_ext[label_col_ext].astype(int).to_numpy()
                if w_col_ext is not None:
                    w_ext = df_ext[w_col_ext].fillna(1.0).to_numpy(dtype=float)
                else:
                    w_ext = np.ones(len(df_ext), dtype=float)

            thresholded = None
            threshold_policy = "none"
            if y_ext is not None:
                thresholded, reps_for_plots, threshold_policy = _compute_thresholded(
                    y_ext, p_ext, w_ext,
                    do_binary_metrics=REPORT_BINARY_METRICS,
                    default_threshold=float(DEFAULT_THRESHOLD),
                    t_train_oof=(float(t_train_oof) if (t_train_oof is not None) else None),
                    do_oracle=bool(DO_EXT_ORACLE_MAXF1),
                    plots_dir=plots,
                    title_prefix=f"{U} → ext_{ext}",
                )

            context = dict(
                track="B_external_benchmarking",
                universe=U,
                ext_dataset=f"ext_{ext}",
                ext_fp=str(ext_fp),
                emb_tag=emb_tag,
                model_fp=str(model_fp),
                best_params_source=best_params_source,
                threshold_policy=threshold_policy,
                split=f"{U}__ext_{ext}",
                title=f"{U} → ext_{ext}",
                reported_thresholds=(list(thresholded.keys()) if isinstance(thresholded, dict) else None),
            )

            evald = _write_pred_eval_bundle(
                out_dir=out_dir,
                context=context,
                df=df_ext,
                p=p_ext,
                y=y_ext,
                w=w_ext,
                thresholded=thresholded,
                threshold_policy=threshold_policy,
                make_plots=True,
            )

            # NEW: per-ext smoke check (fail loudly)
            ok, missing = _smoke_check_ext_dir(out_dir, has_labels=(y_ext is not None))
            assert ok, f"[smoke] {U} → ext_{ext}: missing={missing} | out_dir={out_dir}"

            if y_ext is None:
                print(f"[Track B] {U} → ext_{ext}: labels missing → wrote preds only (ranking use-case).")
                if DO_TRACKB_SANITY and SANITY_DO_SEEN_UNSEEN_2x2 and train_ref is not None and {"enz_idx","sub_idx"}.issubset(df_ext.columns):
                    e_seen, s_seen = _seen_unseen_masks(train_ref, df_ext)
                    quad_counts = {
                        "E_seen__S_seen": int((e_seen & s_seen).sum()),
                        "E_seen__S_unseen": int((e_seen & ~s_seen).sum()),
                        "E_unseen__S_seen": int((~e_seen & s_seen).sum()),
                        "E_unseen__S_unseen": int((~e_seen & ~s_seen).sum()),
                    }
                    (out_dir / "sanity_seen_unseen_counts.json").write_text(json.dumps(quad_counts, indent=2))
                    print(f"[Sanity] {U} → ext_{ext} (labels-missing) seen/unseen counts:", quad_counts)
                continue

            prefer_key = "t_train_oof" if (REPORT_BINARY_METRICS and DO_TRAIN_OOF_THRESHOLD and (t_train_oof is not None)) else "t0p5"
            print(f"[Track B] {U} → ext_{ext}: " + _fmt_line(evald, prefer_key=prefer_key))
            summary_rows.append(_flatten_row(context, evald, prefer_key=prefer_key))

            # -----------------------------
            # SANITY CHECKS per (U, ext) + PRINTS (ADAPTED)
            # -----------------------------
            if DO_TRACKB_SANITY and ({"enz_idx","sub_idx"}.issubset(df_ext.columns)):
                sanity_root = _ensure_dir(out_dir / "sanity")
                sanity_summary = {
                    "universe": U,
                    "ext_dataset": f"ext_{ext}",
                    "baseline": {"auroc": float(evald["auroc_weighted"]), "ap": float(evald["ap_weighted"])},
                    "ablations": None,
                    "permutations": None,
                    "seen_unseen_2x2": None,
                    "stamp": _now_tag(),
                }

                # (1) Ablations: enzyme-only / substrate-only (trained once per universe)
                if SANITY_DO_ABLATIONS and (clf_enz is not None) and (clf_sub is not None):
                    X_ext_enz = _build_X_mode(df_ext, embs, subs, mode="enzyme_only")
                    X_ext_sub = _build_X_mode(df_ext, embs, subs, mode="substrate_only")
                    p_enz = clf_enz.predict_proba(X_ext_enz)[:, 1]
                    p_sub = clf_sub.predict_proba(X_ext_sub)[:, 1]

                    thr_sanity, _, policy_sanity = _compute_thresholded(
                        y_ext, p_enz, w_ext,
                        do_binary_metrics=REPORT_BINARY_METRICS,
                        default_threshold=float(DEFAULT_THRESHOLD),
                        t_train_oof=(float(t_train_oof) if (t_train_oof is not None) else None),
                        do_oracle=False,
                        plots_dir=None,
                        title_prefix="",
                    )
                    ev_enz = _write_pred_eval_bundle(
                        out_dir=sanity_root / "ablations" / "enzyme_only",
                        context=dict(context, sanity="ablations", variant="enzyme_only", split=f"{U}__ext_{ext}__enzyme_only", title=f"{U} → ext_{ext} enzyme_only"),
                        df=df_ext, p=p_enz, y=y_ext, w=w_ext,
                        thresholded=thr_sanity, threshold_policy=policy_sanity,
                        make_plots=False,
                    )

                    thr_sanity, _, policy_sanity = _compute_thresholded(
                        y_ext, p_sub, w_ext,
                        do_binary_metrics=REPORT_BINARY_METRICS,
                        default_threshold=float(DEFAULT_THRESHOLD),
                        t_train_oof=(float(t_train_oof) if (t_train_oof is not None) else None),
                        do_oracle=False,
                        plots_dir=None,
                        title_prefix="",
                    )
                    ev_sub = _write_pred_eval_bundle(
                        out_dir=sanity_root / "ablations" / "substrate_only",
                        context=dict(context, sanity="ablations", variant="substrate_only", split=f"{U}__ext_{ext}__substrate_only", title=f"{U} → ext_{ext} substrate_only"),
                        df=df_ext, p=p_sub, y=y_ext, w=w_ext,
                        thresholded=thr_sanity, threshold_policy=policy_sanity,
                        make_plots=False,
                    )

                    sanity_summary["ablations"] = {
                        "enzyme_only": {"auroc": float(ev_enz["auroc_weighted"]), "ap": float(ev_enz["ap_weighted"])},
                        "substrate_only": {"auroc": float(ev_sub["auroc_weighted"]), "ap": float(ev_sub["ap_weighted"])},
                        "delta_ap_vs_full": {
                            "enzyme_only": float(ev_enz["ap_weighted"] - evald["ap_weighted"]),
                            "substrate_only": float(ev_sub["ap_weighted"] - evald["ap_weighted"]),
                        }
                    }

                # (2) Permutations (use full model only)
                if SANITY_DO_PERMUTATIONS:
                    p_perm_enz = _permute_probs_full_model(clf, df_ext, embs, subs, which="enz", seed=int(seed) + 9001)
                    p_perm_sub = _permute_probs_full_model(clf, df_ext, embs, subs, which="sub", seed=int(seed) + 9002)

                    thr_sanity, _, policy_sanity = _compute_thresholded(
                        y_ext, p_perm_enz, w_ext,
                        do_binary_metrics=REPORT_BINARY_METRICS,
                        default_threshold=float(DEFAULT_THRESHOLD),
                        t_train_oof=(float(t_train_oof) if (t_train_oof is not None) else None),
                        do_oracle=False,
                        plots_dir=None,
                        title_prefix="",
                    )
                    ev_penz = _write_pred_eval_bundle(
                        out_dir=sanity_root / "permutations" / "permute_enz",
                        context=dict(context, sanity="permutations", variant="permute_enz", split=f"{U}__ext_{ext}__permute_enz", title=f"{U} → ext_{ext} permute_enz"),
                        df=df_ext, p=p_perm_enz, y=y_ext, w=w_ext,
                        thresholded=thr_sanity, threshold_policy=policy_sanity,
                        make_plots=False,
                    )

                    thr_sanity, _, policy_sanity = _compute_thresholded(
                        y_ext, p_perm_sub, w_ext,
                        do_binary_metrics=REPORT_BINARY_METRICS,
                        default_threshold=float(DEFAULT_THRESHOLD),
                        t_train_oof=(float(t_train_oof) if (t_train_oof is not None) else None),
                        do_oracle=False,
                        plots_dir=None,
                        title_prefix="",
                    )
                    ev_psub = _write_pred_eval_bundle(
                        out_dir=sanity_root / "permutations" / "permute_sub",
                        context=dict(context, sanity="permutations", variant="permute_sub", split=f"{U}__ext_{ext}__permute_sub", title=f"{U} → ext_{ext} permute_sub"),
                        df=df_ext, p=p_perm_sub, y=y_ext, w=w_ext,
                        thresholded=thr_sanity, threshold_policy=policy_sanity,
                        make_plots=False,
                    )

                    sanity_summary["permutations"] = {
                        "permute_enz": {"auroc": float(ev_penz["auroc_weighted"]), "ap": float(ev_penz["ap_weighted"])},
                        "permute_sub": {"auroc": float(ev_psub["auroc_weighted"]), "ap": float(ev_psub["ap_weighted"])},
                        "delta_ap_vs_full": {
                            "permute_enz": float(ev_penz["ap_weighted"] - evald["ap_weighted"]),
                            "permute_sub": float(ev_psub["ap_weighted"] - evald["ap_weighted"]),
                        }
                    }

                # (3) Seen/Unseen 2x2 breakdown (baseline full probs) — store AP/AUROC per quadrant
                if SANITY_DO_SEEN_UNSEEN_2x2 and (train_ref is not None):
                    e_seen, s_seen = _seen_unseen_masks(train_ref, df_ext)
                    quad = {
                        "E_seen__S_seen": (e_seen & s_seen),
                        "E_seen__S_unseen": (e_seen & ~s_seen),
                        "E_unseen__S_seen": (~e_seen & s_seen),
                        "E_unseen__S_unseen": (~e_seen & ~s_seen),
                    }

                    sanity_summary["seen_unseen_2x2"] = {}

                    for name, m in quad.items():
                        n_q = int(m.sum())
                        if n_q == 0:
                            sanity_summary["seen_unseen_2x2"][name] = {"n": 0, "ap": float("nan"), "auroc": float("nan")}
                            continue

                        df_q = df_ext.loc[m].reset_index(drop=True)
                        y_q  = y_ext[m]
                        w_q  = w_ext[m]
                        p_q  = p_ext[m]

                        thr_q, _, policy_q = _compute_thresholded(
                            y_q, p_q, w_q,
                            do_binary_metrics=REPORT_BINARY_METRICS,
                            default_threshold=float(DEFAULT_THRESHOLD),
                            t_train_oof=(float(t_train_oof) if (t_train_oof is not None) else None),
                            do_oracle=False,
                            plots_dir=None,
                            title_prefix="",
                        )
                        ev_q = _write_pred_eval_bundle(
                            out_dir=sanity_root / "seen_unseen_2x2" / name,
                            context=dict(context, sanity="seen_unseen_2x2", variant=name, split=f"{U}__ext_{ext}__{name}", title=f"{U} → ext_{ext} {name}"),
                            df=df_q, p=p_q, y=y_q, w=w_q,
                            thresholded=thr_q, threshold_policy=policy_q,
                            make_plots=False,
                        )

                        sanity_summary["seen_unseen_2x2"][name] = {
                            "n": n_q,
                            "ap": float(ev_q["ap_weighted"]),
                            "auroc": float(ev_q["auroc_weighted"]),
                        }

                (sanity_root / "sanity_summary.json").write_text(json.dumps(sanity_summary, indent=2))

                # Print sanity checks immediately (Track A-style)
                _print_sanity(U, ext, sanity_summary)

                sanity_rows.append({
                    "universe": U,
                    "ext_dataset": f"ext_{ext}",
                    "ap_full": float(evald["ap_weighted"]),
                    "auroc_full": float(evald["auroc_weighted"]),
                    "sanity_fp": str(sanity_root / "sanity_summary.json")
                })

    # write global table (flat, CSV-friendly)
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        (RUN_DIR / "trackB_external_summary.csv").write_text(df_sum.to_csv(index=False))
        print("\n[Track B] Wrote summary CSV:", RUN_DIR / "trackB_external_summary.csv")

    # optional: compact sanity index
    if sanity_rows:
        df_s = pd.DataFrame(sanity_rows)
        (RUN_DIR / "trackB_external_sanity_index.csv").write_text(df_s.to_csv(index=False))
        print("[Track B] Wrote sanity index CSV:", RUN_DIR / "trackB_external_sanity_index.csv")

    # run manifest (with frozen provenance)
    (RUN_DIR / "run_manifest.json").write_text(json.dumps({
        "run_id": RUN_ID,
        "track": "B_external_benchmarking",
        "emb_tag": emb_tag,
        "emb_fp": str(emb_fp),
        "substrate_kind": substrate_kind,
        "substrate_fp": str(substrate_fp),
        "cfg_hash": cfg_hash,

        "frozen_best_params_source_fp": FROZEN_BP_FP,
        "frozen_hpo_source": FROZEN_HPO_SOURCE,
        "best_params_source": best_params_source,
        "frozen_params": FROZEN_PARAMS,

        "universes": list(universe_tags),
        "ext_tags": list(ext_tags),
        "ext_tags_by_universe": {k: list(v) for k, v in (ext_tags_by_universe or {}).items()},

        "report_binary_metrics": bool(REPORT_BINARY_METRICS),
        "default_threshold": float(DEFAULT_THRESHOLD),
        "did_train_oof_threshold": bool(DO_TRAIN_OOF_THRESHOLD),
        "did_ext_oracle_maxf1": bool(DO_EXT_ORACLE_MAXF1),
        "did_trackB_sanity": bool(DO_TRACKB_SANITY),
        "did_overlap_audit": bool(DO_OVERLAP_AUDIT),
        "did_filter_overlap_from_ext": bool(FILTER_OVERLAP_FROM_EXT),
        "sanity_flags": sanity_flags,
        "stamp": _now_tag(),
    }, indent=2))

    print("\n[Track B] DONE")
    print("  RUN_DIR =", RUN_DIR)

    # Best-effort globals for downstream convenience
    globals()["RUN_ID"] = RUN_ID
    globals()["RUN_DIR"] = RUN_DIR

    return RUN_DIR

def run_or_replay_trackB_external(*,
        proj: Path,
        run_id: str,
        emb_tag: str,
        emb_fp: Path,
        substrate_kind: str,
        substrate_fp: Path,
        universe_tags: list[str],
        ext_tags: list[str],
        ext_tags_by_universe: dict[str, list[str]],
        force: bool,
        seed: int,
        n_splits_inner: int,
        prefer_key: str,
    ) -> Path:

    proj = Path(proj)
    emb_tag = str(emb_tag).strip()
    substrate_kind = str(substrate_kind).strip()
    emb_fp = Path(emb_fp)
    substrate_fp = Path(substrate_fp)

    sanity_flags = dict(
        ablations=bool(SANITY_DO_ABLATIONS),
        permutations=bool(SANITY_DO_PERMUTATIONS),
        seen_unseen_2x2=bool(SANITY_DO_SEEN_UNSEEN_2x2),
        sanity_n_estimators_cap=SANITY_N_ESTIMATORS_CAP,
        sanity_print=bool(PRINT_SANITY),
        sanity_print_style=str(SANITY_PRINT_STYLE),
        sanity_print_show_2x2_ap=bool(SANITY_PRINT_SHOW_2x2_AP),
    )

    cfg_hash = _trackB_external_cfg_hash(
        emb_tag=emb_tag,
        emb_fp=emb_fp,
        substrate_kind=substrate_kind,
        substrate_fp=substrate_fp,
        universe_tags=universe_tags,
        ext_tags=ext_tags,
        ext_tags_by_universe=ext_tags_by_universe,
        report_binary_metrics=REPORT_BINARY_METRICS,
        default_threshold=DEFAULT_THRESHOLD,
        do_train_oof_threshold=DO_TRAIN_OOF_THRESHOLD,
        do_ext_oracle_maxf1=DO_EXT_ORACLE_MAXF1,
        do_overlap_audit=DO_OVERLAP_AUDIT,
        filter_overlap_from_ext=FILTER_OVERLAP_FROM_EXT,
        do_trackb_sanity=DO_TRACKB_SANITY,
        sanity_flags=sanity_flags,
    )

    run_dir_existing = None
    if (not force):
        try:
            run_dir_existing = _resolve_trackB_external_run_dir(
                proj,
                run_id=run_id,
                emb_tag=emb_tag,
                emb_fp=emb_fp,
                substrate_kind=substrate_kind,
                substrate_fp=substrate_fp,
                universe_tags=universe_tags,
                ext_tags=ext_tags,
                ext_tags_by_universe=ext_tags_by_universe,
                report_binary_metrics=REPORT_BINARY_METRICS,
                default_threshold=DEFAULT_THRESHOLD,
                do_train_oof_threshold=DO_TRAIN_OOF_THRESHOLD,
                do_ext_oracle_maxf1=DO_EXT_ORACLE_MAXF1,
                do_trackb_sanity=DO_TRACKB_SANITY,
                sanity_flags=sanity_flags,
                cfg_hash=cfg_hash,
            )
        except FileNotFoundError:
            run_dir_existing = None

    if (run_dir_existing is not None) and (not force):
        run_dir_existing = Path(run_dir_existing)
        assert (run_dir_existing / "trackB_external").exists(), f"Cached mode requires trackB_external exists: {run_dir_existing / 'trackB_external'}"
        replay_trackB_external_run(
            run_dir=run_dir_existing,
            universe_tags=universe_tags,
            ext_tags=ext_tags,
            ext_tags_by_universe=ext_tags_by_universe,
            prefer_key=prefer_key,
        )
        # best-effort globals for downstream convenience
        globals()["RUN_ID"] = run_dir_existing.name
        globals()["RUN_DIR"] = run_dir_existing
        return run_dir_existing

    # Compute path (unchanged semantics): create a new timestamped run dir and write artifacts
    return run_trackB_external_benchmarking(
        proj=proj,
        emb_tag=emb_tag,
        emb_fp=Path(emb_fp),
        substrate_kind=substrate_kind,
        substrate_fp=Path(substrate_fp),
        universe_tags=universe_tags,
        ext_tags=ext_tags,
        ext_tags_by_universe=ext_tags_by_universe,
        force=force,
        seed=seed,
        n_splits_inner=n_splits_inner,
    )

def _write_trackc_json(fp: Path, obj):
    fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(obj, indent=2))

def trackc_cfg_for_split(split_name: str | None, *, external: bool = False, base_cfg: dict | None = None):
    cfg = copy.deepcopy(base_cfg if base_cfg is not None else TRACKC_CFG)
    if (not external) and TRACKC_ENABLE_A3_TRUE_DOUBLECOLD_REFIT and str(split_name).startswith("A3"):
        cfg.update(TRACKC_A3_PATCH_OVERRIDES)
    return cfg

def _stable_json(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"))

def _sha12(x):
    return hashlib.sha1(_stable_json(x).encode("utf-8")).hexdigest()[:12]

def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def trackc_attach_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if ("cluster_id_80" not in df.columns) and CLUSTERMAP_CSV.exists():
        cmap = pd.read_csv(CLUSTERMAP_CSV)
        cmap["enzyme"] = cmap["enzyme"].astype(str).str.strip()
        if "cluster_id_80" in cmap.columns:
            df = df.merge(cmap[["enzyme", "cluster_id_80"]], on="enzyme", how="left")
    if ("sub_group" not in df.columns) and SCAF_FP.exists():
        smap = pd.read_csv(SCAF_FP, usecols=["sub_idx", "sub_group"])
        smap["sub_idx"] = pd.to_numeric(smap["sub_idx"], errors="raise").astype(int)
        smap["sub_group"] = pd.to_numeric(smap["sub_group"], errors="coerce")
        df = df.merge(smap, on="sub_idx", how="left")
    return df

def _safe_weighted_ap(y, p, w):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float); w = np.asarray(w).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p, sample_weight=w))

def _weighted_bce_with_logits(logits, y, w):
    y = y.float(); w = w.float()
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * w).sum() / w.sum().clamp_min(1e-12)

class PairTokenDataset(torch.utils.data.Dataset):
    def __init__(self, df_part: pd.DataFrame):
        need({"enz_idx", "sub_idx"}.issubset(df_part.columns), "df_part missing enz_idx/sub_idx")
        self.df = df_part.reset_index(drop=True).copy()
        _, _, y, w = _get_label_and_weight(self.df)
        self.y = y.astype(np.float32, copy=False)
        self.w = w.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[int(i)]
        enz_idx = int(row["enz_idx"]); sub_idx = int(row["sub_idx"])
        enz_fp = enz_path_map.get(enz_idx, None); sub_fp = sub_path_map.get(sub_idx, None)
        need(enz_fp is not None, f"Missing enz_idx in ESM token index: {enz_idx}")
        need(sub_fp is not None, f"Missing sub_idx in Mol token index: {sub_idx}")
        meta = {c: row[c] for c in ["pair_id", "enzyme", "enz_idx", "sub_idx", "source", "organism"] if c in row.index}
        return {
            "enz_tok": _load_token_file(enz_fp).astype(np.float32, copy=False),
            "sub_tok": _load_token_file(sub_fp).astype(np.float32, copy=False),
            "y": float(self.y[i]),
            "w": float(self.w[i]),
            "meta": meta,
        }

def _pad_token_batch(arr_list):
    B = len(arr_list)
    lens = [int(a.shape[0]) for a in arr_list]
    d = int(arr_list[0].shape[1])
    L = max(lens)
    x = torch.zeros((B, L, d), dtype=torch.float32)
    m = torch.zeros((B, L), dtype=torch.bool)
    for i, a in enumerate(arr_list):
        n = int(a.shape[0]); x[i, :n] = torch.from_numpy(a); m[i, :n] = True
    return x, m

def trackc_collate(batch):
    enz_tok, enz_mask = _pad_token_batch([b["enz_tok"] for b in batch])
    sub_tok, sub_mask = _pad_token_batch([b["sub_tok"] for b in batch])
    y = torch.tensor([b["y"] for b in batch], dtype=torch.float32)
    w = torch.tensor([b["w"] for b in batch], dtype=torch.float32)
    meta = [b["meta"] for b in batch]
    return {"enz_tok": enz_tok, "enz_mask": enz_mask, "sub_tok": sub_tok, "sub_mask": sub_mask, "y": y, "w": w, "meta": meta}

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.q_ln = nn.LayerNorm(d_model)
        self.kv_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, q, q_mask, kv, kv_mask):
        out = q
        valid_rows = kv_mask.any(dim=1)
        if bool(valid_rows.any()):
            qv = self.q_ln(q[valid_rows]); kvv = self.kv_ln(kv[valid_rows])
            attn_out, _ = self.attn(qv, kvv, kvv, key_padding_mask=(~kv_mask[valid_rows]), need_weights=False)
            out = out.clone()
            out[valid_rows] = out[valid_rows] + self.drop(attn_out)
        out = out + self.ffn(self.ffn_ln(out))
        out = out * q_mask.unsqueeze(-1).to(out.dtype)
        return out

class TrackCModel(nn.Module):
    def __init__(self, d_prot_in: int, d_mol_in: int, cfg: dict):
        super().__init__()
        d = int(cfg["d_model"])
        self.p_drop_enzyme = float(cfg["p_drop_enzyme"])
        self.prot_proj = nn.Sequential(nn.LayerNorm(d_prot_in), nn.Linear(d_prot_in, d))
        self.sub_proj = nn.Sequential(nn.LayerNorm(d_mol_in), nn.Linear(d_mol_in, d))
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(d_model=d, n_heads=int(cfg["n_heads"]), dropout=float(cfg["dropout"]))
            for _ in range(int(cfg["n_xattn_blocks"]))
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, int(cfg["classifier_hidden"])),
            nn.GELU(),
            nn.Dropout(float(cfg["dropout"])),
            nn.Linear(int(cfg["classifier_hidden"]), 1),
        )

    def _masked_mean(self, x, mask):
        w = mask.unsqueeze(-1).to(x.dtype)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)

    def forward(self, enz_tok, enz_mask, sub_tok, sub_mask):
        enz = self.prot_proj(enz_tok)
        sub = self.sub_proj(sub_tok)

        if self.training and self.p_drop_enzyme > 0.0:
            drop_rows = (torch.rand(enz.shape[0], device=enz.device) < self.p_drop_enzyme)
            if bool(drop_rows.any()):
                enz = enz.clone(); enz_mask = enz_mask.clone()
                enz[drop_rows] = 0.0
                enz_mask[drop_rows] = False

        x = sub
        for blk in self.blocks:
            x = blk(x, sub_mask, enz, enz_mask)

        pooled = self._masked_mean(x, sub_mask)
        return self.classifier(pooled).squeeze(-1)

def trackc_make_loader(df_part: pd.DataFrame, batch_size: int, shuffle: bool):
    ds = PairTokenDataset(df_part)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(TRACKC_CFG["num_workers"]),
        collate_fn=trackc_collate,
        pin_memory=(device == "cuda"),
    )

def trackc_predict_model(model: nn.Module, loader):
    model.eval()
    ys, ws, ps, meta_rows = [], [], [], []
    for batch in loader:
        enz_tok = batch["enz_tok"].to(device, non_blocking=True)
        enz_mask = batch["enz_mask"].to(device, non_blocking=True)
        sub_tok = batch["sub_tok"].to(device, non_blocking=True)
        sub_mask = batch["sub_mask"].to(device, non_blocking=True)
        logits = model(enz_tok, enz_mask, sub_tok, sub_mask)
        prob = torch.sigmoid(logits).detach().to("cpu").numpy()
        ys.append(batch["y"].numpy()); ws.append(batch["w"].numpy()); ps.append(prob); meta_rows.extend(batch["meta"])
    y = np.concatenate(ys).astype(int, copy=False)
    w = np.concatenate(ws).astype(float, copy=False)
    p = np.concatenate(ps).astype(float, copy=False)
    return y, w, p, pd.DataFrame(meta_rows)

def _choose_val_split(train_df: pd.DataFrame, split_name: str, seed: int):
    train_df = trackc_attach_groups(train_df)
    _, _, y, _ = _get_label_and_weight(train_df)
    y = y.astype(int)

    if str(split_name).startswith(("A0b", "A1", "A1_external")):
        gcol = "cluster_id_80"
    elif str(split_name).startswith("A2"):
        gcol = "sub_group"
    elif str(split_name).startswith("A3"):
        gcol = "__combo_group__"
        train_df[gcol] = train_df["cluster_id_80"].astype(str) + "__" + train_df["sub_group"].astype(str)
    else:
        gcol = None

    if (gcol is not None) and (gcol in train_df.columns) and train_df[gcol].notna().sum() > 0:
        groups = train_df[gcol].fillna("NA").astype(str).to_numpy()
        try:
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(seed))
            tr_idx, va_idx = next(splitter.split(np.zeros(len(train_df)), y, groups=groups))
        except Exception:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(seed))
            tr_idx, va_idx = next(splitter.split(np.zeros(len(train_df)), y))
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(seed))
        tr_idx, va_idx = next(splitter.split(np.zeros(len(train_df)), y))
    return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[va_idx].reset_index(drop=True)

def _choose_val_split_A3_true_doublecold(train_df: pd.DataFrame, seed: int, cfg: dict):
    df = trackc_attach_groups(train_df).copy()

    need({"cluster_id_80", "sub_group"}.issubset(df.columns), "A3 patch requires cluster_id_80 and sub_group")
    mask_ok = df["cluster_id_80"].notna() & df["sub_group"].notna()
    n_missing = int((~mask_ok).sum())
    if n_missing:
        print(f"[Track C][A3 refit][warn] dropping {n_missing} rows with missing cluster_id_80/sub_group before selector split")
    df = df.loc[mask_ok].reset_index(drop=True)
    need(len(df) > 0, "A3 patch: no rows remain after dropping missing group labels")

    _, _, y, _ = _get_label_and_weight(df)
    y = y.astype(int)

    clusters = np.array(sorted(pd.unique(df["cluster_id_80"].astype(str))))
    scaffs = np.array(sorted(pd.unique(df["sub_group"].astype(str))))
    need(len(clusters) >= 2, "A3 patch requires at least 2 enzyme clusters")
    need(len(scaffs) >= 2, "A3 patch requires at least 2 scaffold groups")

    frac_entities = float(cfg.get("a3_val_entity_holdout_frac", 1.0 / 3.0))
    k_c = max(1, min(len(clusters) - 1, int(round(frac_entities * len(clusters)))))
    k_s = max(1, min(len(scaffs) - 1, int(round(frac_entities * len(scaffs)))))

    target_frac_kept = float(cfg.get("a3_val_target_frac_kept", 0.20))
    n_trials = int(cfg.get("a3_val_n_trials", 256))
    min_train_rows = int(cfg.get("a3_val_min_train_rows", 512))
    min_val_rows = int(cfg.get("a3_val_min_val_rows", 128))
    min_pos = int(cfg.get("a3_val_min_pos", 8))
    min_neg = int(cfg.get("a3_val_min_neg", 8))

    rng = np.random.RandomState(int(seed) + int(cfg.get("a3_val_seed_offset", 1000)))
    clus_ser = df["cluster_id_80"].astype(str)
    scaf_ser = df["sub_group"].astype(str)

    best = None
    best_masks = None

    for _ in range(n_trials):
        val_clusters = set(rng.choice(clusters, size=k_c, replace=False).tolist())
        val_scaffs = set(rng.choice(scaffs, size=k_s, replace=False).tolist())

        c_val = clus_ser.isin(val_clusters).to_numpy()
        s_val = scaf_ser.isin(val_scaffs).to_numpy()

        val_mask = c_val & s_val
        train_mask = (~c_val) & (~s_val)
        drop_mask = ~(val_mask | train_mask)

        n_tr = int(train_mask.sum())
        n_va = int(val_mask.sum())
        n_dr = int(drop_mask.sum())
        if (n_tr < min_train_rows) or (n_va < min_val_rows):
            continue

        y_tr = y[train_mask]
        y_va = y[val_mask]
        pos_tr = int(y_tr.sum())
        neg_tr = int(len(y_tr) - pos_tr)
        pos_va = int(y_va.sum())
        neg_va = int(len(y_va) - pos_va)

        if min(pos_tr, neg_tr) < min_pos or min(pos_va, neg_va) < min_pos:
            continue
        if neg_tr < min_neg or neg_va < min_neg:
            continue

        frac_kept = n_va / max(n_tr + n_va, 1)
        prev_tr = float(y_tr.mean())
        prev_va = float(y_va.mean())
        drop_frac = n_dr / max(len(df), 1)

        score = 4.0 * abs(frac_kept - target_frac_kept) + 1.0 * abs(prev_va - prev_tr) + 0.25 * drop_frac
        cand = {
            "score": float(score),
            "n_train": n_tr,
            "n_val": n_va,
            "n_drop": n_dr,
            "frac_val_of_kept": float(frac_kept),
            "prev_train": float(prev_tr),
            "prev_val": float(prev_va),
            "n_val_clusters": int(len(val_clusters)),
            "n_val_scaffolds": int(len(val_scaffs)),
        }
        if (best is None) or (cand["score"] < best["score"]):
            best = cand
            best_masks = (train_mask.copy(), val_mask.copy(), drop_mask.copy())

    if best_masks is None:
        raise RuntimeError(
            "A3 patch could not find a feasible strict selector split. "
            "Relax a3_val_* minima or increase a3_val_n_trials."
        )

    train_mask, val_mask, drop_mask = best_masks
    print(
        "[Track C][A3 refit] strict selector split | "
        f"n_train={best['n_train']} | n_val={best['n_val']} | n_drop={best['n_drop']} | "
        f"frac_val_of_kept={best['frac_val_of_kept']:.3f} | "
        f"prev_train={best['prev_train']:.3f} | prev_val={best['prev_val']:.3f} | "
        f"n_val_clusters={best['n_val_clusters']} | n_val_scaffolds={best['n_val_scaffolds']}"
    )

    df_tr = df.loc[train_mask].reset_index(drop=True)
    df_va = df.loc[val_mask].reset_index(drop=True)
    meta = dict(best)
    meta["n_total_after_missing_drop"] = int(len(df))
    meta["n_missing_group_rows_dropped"] = int(n_missing)
    return df_tr, df_va, meta

def _trackc_train_fixed_epochs(df_train: pd.DataFrame, n_epochs: int, cfg: dict, split_name: str, log_prefix: str):
    seed_everything(int(cfg["seed"]))
    train_loader = trackc_make_loader(df_train, batch_size=int(cfg["batch_size"]), shuffle=True)

    model = TrackCModel(d_prot_in=ENZ_TOK_DIM, d_mol_in=SUB_TOK_DIM, cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    hist = []
    for epoch in range(1, int(n_epochs) + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            enz_tok = batch["enz_tok"].to(device, non_blocking=True)
            enz_mask = batch["enz_mask"].to(device, non_blocking=True)
            sub_tok = batch["sub_tok"].to(device, non_blocking=True)
            sub_mask = batch["sub_mask"].to(device, non_blocking=True)
            yb = batch["y"].to(device, non_blocking=True)
            wb = batch["w"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(enz_tok, enz_mask, sub_tok, sub_mask)
                loss = _weighted_bce_with_logits(logits, yb, wb)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(train_losses)) if len(train_losses) else np.nan
        hist.append({"epoch": int(epoch), "train_loss": mean_loss})
        print(f"[Track C][{split_name}][{log_prefix}] epoch={epoch:02d} train_loss={mean_loss:.4f}")

    return model, hist

def _trackc_select_epoch_A3(train_df: pd.DataFrame, split_name: str, selector_dir: Path, cfg: dict):
    cfg = copy.deepcopy(cfg)
    seed_everything(int(cfg["seed"]))

    df_tr, df_va, split_meta = _choose_val_split_A3_true_doublecold(train_df, seed=int(cfg["seed"]), cfg=cfg)
    train_loader = trackc_make_loader(df_tr, batch_size=int(cfg["batch_size"]), shuffle=True)
    val_loader = trackc_make_loader(df_va, batch_size=int(cfg["eval_batch_size"]), shuffle=False)

    model = TrackCModel(d_prot_in=ENZ_TOK_DIM, d_mol_in=SUB_TOK_DIM, cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_state = None
    best_ap = -np.inf
    best_epoch = None
    bad_epochs = 0
    hist = []

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            enz_tok = batch["enz_tok"].to(device, non_blocking=True)
            enz_mask = batch["enz_mask"].to(device, non_blocking=True)
            sub_tok = batch["sub_tok"].to(device, non_blocking=True)
            sub_mask = batch["sub_mask"].to(device, non_blocking=True)
            yb = batch["y"].to(device, non_blocking=True)
            wb = batch["w"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(enz_tok, enz_mask, sub_tok, sub_mask)
                loss = _weighted_bce_with_logits(logits, yb, wb)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        y_va, w_va, p_va, _ = trackc_predict_model(model, val_loader)
        ap_va = _safe_weighted_ap(y_va, p_va, w_va)
        if np.isnan(ap_va):
            ap_va = -float(np.mean(train_losses))

        hist.append(dict(epoch=int(epoch), train_loss=float(np.mean(train_losses)), val_ap=float(ap_va)))
        print(f"[Track C][{split_name}][SELECT] epoch={epoch:02d} train_loss={np.mean(train_losses):.4f} val_ap={ap_va:.4f}")

        if ap_va > best_ap:
            best_ap = float(ap_va)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["patience"]):
                print(f"[Track C][{split_name}][SELECT] early stop at epoch={epoch}")
                break

    need(best_state is not None, f"[{split_name}] selector captured no best_state")
    need(best_epoch is not None, f"[{split_name}] selector captured no best_epoch")

    model.load_state_dict(best_state)
    y_va, w_va, p_va, _ = trackc_predict_model(model, val_loader)

    selector_dir = Path(selector_dir)
    selector_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "cfg": cfg, "best_val_ap": best_ap, "best_epoch": best_epoch, "history": hist},
        selector_dir / "selector_model.pt",
    )
    _write_trackc_json(selector_dir / "selector_fit_history.json", hist)
    _write_trackc_json(selector_dir / "selector_split_summary.json", split_meta)

    thresholds = {"t0p5": 0.5}
    if _weighted_pr_f1_sweep is not None and len(np.unique(y_va)) >= 2:
        t_val, f1_val, *_ = _weighted_pr_f1_sweep(y_va, p_va, w_va)
        thresholds = {"t_val_f1": float(t_val), "t0p5": 0.5}
        _write_trackc_json(
            selector_dir / "selector_val_threshold.json",
            {
                "criterion": "maximize_weighted_f1_on_selector_val",
                "t_val_f1": float(t_val),
                "f1_val_max": float(f1_val),
                "best_epoch": int(best_epoch),
                "best_val_ap": float(best_ap),
                "val_scheme": cfg.get("a3_val_scheme", "patched"),
            },
        )

    return best_epoch, thresholds, hist, split_meta

def _trackc_fit_model_A3_patched(train_df: pd.DataFrame, split_name: str, run_dir: Path, cfg: dict):
    run_dir = Path(run_dir)
    selector_dir = run_dir.parent / "selector"
    model_dir = run_dir

    best_epoch, thresholds, selector_hist, split_meta = _trackc_select_epoch_A3(
        train_df=train_df,
        split_name=split_name,
        selector_dir=selector_dir,
        cfg=cfg,
    )
    print(f"[Track C][A3 refit] selected best_epoch={best_epoch}; refitting on ALL non-test train rows (n={len(train_df)})")

    model, refit_hist = _trackc_train_fixed_epochs(
        df_train=train_df,
        n_epochs=int(best_epoch),
        cfg=cfg,
        split_name=split_name,
        log_prefix="REFIT",
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": cfg,
            "refit_epochs": int(best_epoch),
            "history": refit_hist,
        },
        model_dir / "model.pt",
    )
    _write_trackc_json(model_dir / "fit_history.json", refit_hist)
    _write_trackc_json(
        model_dir / "refit_summary.json",
        {
            "refit_epochs": int(best_epoch),
            "n_refit_rows": int(len(train_df)),
            "selector_split_summary": split_meta,
            "selector_dir": str(selector_dir),
        },
    )
    thr_payload = {
        "criterion": "selector_val_threshold",
        "best_epoch": int(best_epoch),
        "a3_fit_scheme": cfg.get("a3_fit_scheme"),
        "a3_val_scheme": cfg.get("a3_val_scheme"),
    }
    thr_payload.update({k: float(v) for k, v in thresholds.items()})
    _write_trackc_json(model_dir / "val_threshold.json", thr_payload)

    return model, thresholds, refit_hist

def trackc_fit_model(train_df: pd.DataFrame, split_name: str, run_dir: Path, cfg: dict):
    cfg = trackc_cfg_for_split(split_name=split_name, external=False, base_cfg=cfg)
    seed_everything(int(cfg["seed"]))
    run_dir = Path(run_dir)

    if str(split_name).startswith("A3") and str(cfg.get("a3_fit_scheme", "")) == "strict_selector_plus_full_non_test_refit_v1":
        return _trackc_fit_model_A3_patched(
            train_df=train_df,
            split_name=split_name,
            run_dir=run_dir,
            cfg=cfg,
        )

    df_tr, df_va = _choose_val_split(train_df, split_name=split_name, seed=int(cfg["seed"]))
    train_loader = trackc_make_loader(df_tr, batch_size=int(cfg["batch_size"]), shuffle=True)
    val_loader = trackc_make_loader(df_va, batch_size=int(cfg["eval_batch_size"]), shuffle=False)

    model = TrackCModel(d_prot_in=ENZ_TOK_DIM, d_mol_in=SUB_TOK_DIM, cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_state = None
    best_ap = -np.inf
    bad_epochs = 0
    hist = []

    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            enz_tok = batch["enz_tok"].to(device, non_blocking=True)
            enz_mask = batch["enz_mask"].to(device, non_blocking=True)
            sub_tok = batch["sub_tok"].to(device, non_blocking=True)
            sub_mask = batch["sub_mask"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            w = batch["w"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(enz_tok, enz_mask, sub_tok, sub_mask)
                loss = _weighted_bce_with_logits(logits, y, w)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        y_va, w_va, p_va, _ = trackc_predict_model(model, val_loader)
        ap_va = _safe_weighted_ap(y_va, p_va, w_va)
        if np.isnan(ap_va):
            ap_va = -float(np.mean(train_losses))

        hist.append(dict(epoch=int(epoch), train_loss=float(np.mean(train_losses)), val_ap=float(ap_va)))
        print(f"[Track C][{split_name}] epoch={epoch:02d} train_loss={np.mean(train_losses):.4f} val_ap={ap_va:.4f}")

        if ap_va > best_ap:
            best_ap = float(ap_va)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["patience"]):
                print(f"[Track C][{split_name}] early stop at epoch={epoch}")
                break

    need(best_state is not None, f"[{split_name}] no best_state captured")
    model.load_state_dict(best_state)

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "cfg": cfg, "best_val_ap": best_ap, "history": hist},
        run_dir / "model.pt",
    )
    _write_trackc_json(run_dir / "fit_history.json", hist)

    thresholds = {"t0p5": 0.5}
    if _weighted_pr_f1_sweep is not None and len(np.unique(y_va)) >= 2:
        t_val, f1_val, *_ = _weighted_pr_f1_sweep(y_va, p_va, w_va)
        thresholds = {"t_val_f1": float(t_val), "t0p5": 0.5}
        _write_trackc_json(
            run_dir / "val_threshold.json",
            {
                "criterion": "maximize_weighted_f1_on_val",
                "t_val_f1": float(t_val),
                "f1_val_max": float(f1_val),
            },
        )

    return model, thresholds, hist

def trackc_cfg_hash(extra: dict, cfg: dict | None = None):
    base = copy.deepcopy(cfg if cfg is not None else TRACKC_CFG)
    base.update(extra)
    return _sha12(base)

def _trackc_cfg_for_target(*, universe_tag: str, split_name: str | None, external: bool):
    return globals()["trackc_cfg_for_split"](
        split_name=split_name,
        external=external,
        base_cfg=TRACKC_CFG,
    )

def _retire_legacy_trackc_a3_runs(active_run_dir: Path):
    import shutil

    runs_root = Path(PROJ) / "metrics" / "runs"
    retire_root = runs_root / "_retired_trackc_a3"
    retire_root.mkdir(parents=True, exist_ok=True)

    active_run_dir = Path(active_run_dir).resolve()
    patt = "trackC__trainpool__A3_doubleCold_cluster80xscafGroup__*__moltokxattn__cfg-*"
    for rd in sorted(runs_root.glob(patt)):
        if rd.resolve() == active_run_dir:
            continue
        dst = retire_root / rd.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(rd), str(dst))
        print(f"[Track C][A3 retire] moved legacy A3 run: {rd.name}")

def _trackc_run_id(*, universe_tag: str, split_name: str | None, external: bool):
    cfg_eff = _trackc_cfg_for_target(
        universe_tag=universe_tag,
        split_name=split_name,
        external=external,
    )
    cfg_sig = globals()["trackc_cfg_hash"](
        {
            "universe_tag": universe_tag,
            "split_name": split_name,
            "external": bool(external),
            "emb_tag": cfg_eff["emb_tag"],
            "token_cache_enz": str(ESMTOK_INDEX_FP),
            "token_cache_sub": str(MOLTOK_INDEX_FP),
        },
        cfg=cfg_eff,
    )
    if external:
        return f"trackC__external__{cfg_eff['emb_tag']}__moltokxattn__cfg-{cfg_sig}"
    return f"trackC__{universe_tag}__{split_name}__{cfg_eff['emb_tag']}__moltokxattn__cfg-{cfg_sig}"

def _trackc_eval_complete(eval_dir: Path):
    eval_dir = Path(eval_dir)
    if TRACKC_bundle_smoke_check is None:
        return eval_dir.exists() and (eval_dir / "headline.json").exists() and (eval_dir / "preds.csv").exists(), []
    with contextlib.redirect_stdout(_io.StringIO()):
        ok, missing = TRACKC_bundle_smoke_check(eval_dir)
    return bool(ok), list(missing)

def _write_trackc_manifest(fp: Path, obj: dict):
    fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(obj, indent=2))

def _run_trackc_single_split(*, universe_tag: str, split_name: str, split_json_fp: Path, force: bool = False):
    pairs = _load_pairs_universe(universe_tag)
    pairs = globals()["trackc_attach_groups"](pairs)

    split_obj = _read_split_obj(Path(split_json_fp))
    train_ids, test_ids = _resolve_train_test_ids_from_split_obj(pairs, split_obj, Path(split_json_fp))
    need(len(train_ids) > 0 and len(test_ids) > 0, f"[{split_name}] empty train/test")
    need(len(np.intersect1d(train_ids, test_ids)) == 0, f"[{split_name}] train/test overlap")

    cfg_eff = _trackc_cfg_for_target(universe_tag=universe_tag, split_name=split_name, external=False)
    run_id = _trackc_run_id(universe_tag=universe_tag, split_name=split_name, external=False)
    run_dir = PROJ / "metrics" / "runs" / run_id

    if split_name.startswith("A1"):
        eval_prefix = "trackC_internal/test"
        model_dir = run_dir / "trackC_internal/model"
    else:
        eval_prefix = f"trackC/{split_name}/test"
        model_dir = run_dir / f"trackC/{split_name}/model"
    eval_dir = run_dir / eval_prefix

    if eval_dir.exists() and (not force):
        ok, missing = _trackc_eval_complete(eval_dir)
        if ok:
            print(f"[Track C][skip] complete bundle exists: {eval_dir}")
            if (
                TRACKC_RETIRE_LEGACY_A3_RUNS
                and str(split_name).startswith("A3")
                and str(cfg_eff.get("a3_fit_scheme", "")).startswith("strict_selector")
            ):
                _retire_legacy_trackc_a3_runs(run_dir)
            return run_dir

    train_df = pairs.iloc[train_ids].reset_index(drop=True).copy()
    test_df = pairs.iloc[test_ids].reset_index(drop=True).copy()

    model, thresholds, hist = globals()["trackc_fit_model"](
        train_df=train_df,
        split_name=split_name,
        run_dir=model_dir,
        cfg=cfg_eff,
    )

    test_loader = globals()["trackc_make_loader"](
        test_df,
        batch_size=int(cfg_eff["eval_batch_size"]),
        shuffle=False,
    )
    y_test, w_test, p_test, _ = globals()["trackc_predict_model"](model, test_loader)

    TRACKC_eval_and_write(
        run_dir=run_dir,
        split_name=split_name,
        df_part=test_df,
        y=y_test,
        w=w_test,
        p=p_test,
        thresholds=thresholds,
        prefix=eval_prefix,
    )

    ok, missing = _trackc_eval_complete(eval_dir)
    need(ok, f"[Track C][{split_name}] wrote incomplete bundle: {missing}")

    track_name = (
        "C_token_xattn_internal_A3_valpatch_refit"
        if str(split_name).startswith("A3") and str(cfg_eff.get("a3_fit_scheme", "")).startswith("strict_selector")
        else "C_token_xattn_internal"
    )

    _write_trackc_manifest(
        run_dir / "run_manifest.json",
        {
            "track": track_name,
            "run_id": run_id,
            "universe": universe_tag,
            "split_name": split_name,
            "split_json_fp": str(split_json_fp),
            "token_cache_enz": str(ESMTOK_INDEX_FP),
            "token_cache_sub": str(MOLTOK_INDEX_FP),
            "cfg": cfg_eff,
            "thresholds": thresholds,
        },
    )

    if (
        TRACKC_RETIRE_LEGACY_A3_RUNS
        and str(split_name).startswith("A3")
        and str(cfg_eff.get("a3_fit_scheme", "")).startswith("strict_selector")
    ):
        _retire_legacy_trackc_a3_runs(run_dir)

    print(f"[Track C] DONE split={split_name} | run_dir={run_dir}")
    return run_dir

def run_trackc_internal_suite(*, force: bool = False):
    a1_json_cands = [
        PROJ / "splits" / "trainpool_A1_enzyme80.json",
        PROJ / "splits" / "trainpool_enzyme80_split.json",
    ]
    a1_json = next((p for p in a1_json_cands if Path(p).exists()), a1_json_cands[-1])
    specs = [dict(universe_tag="trainpool", split_name="A1_enzyme80", split_json_fp=a1_json)]
    if "EVAL_SPECS" in globals():
        for s in EVAL_SPECS:
            specs.append(dict(
                universe_tag=str(s["universe_tag"]),
                split_name=str(s["split_name"]),
                split_json_fp=Path(s["split_json_fp"]),
            ))
    else:
        specs.extend([
            dict(universe_tag="trainpool", split_name="A0_randomRow", split_json_fp=PROJ / "splits" / "trainpool_A0_randomRow.json"),
            dict(universe_tag="trainpool", split_name="A0b_randomEnzCluster80", split_json_fp=PROJ / "splits" / "trainpool_A0b_randomEnzCluster80.json"),
            dict(universe_tag="trainpool", split_name="A2_scaffoldOOD", split_json_fp=PROJ / "splits" / "trainpool_A2_scaffoldOOD.json"),
            dict(universe_tag="trainpool", split_name="A3_doubleCold_cluster80xscafGroup", split_json_fp=PROJ / "splits" / "trainpool_A3_doubleCold_cluster80xscafGroup.json"),
        ])

    out = {}
    for spec in specs:
        need(Path(spec["split_json_fp"]).exists(), f"Missing split json: {spec['split_json_fp']}")
        rd = _run_trackc_single_split(
            universe_tag=spec["universe_tag"],
            split_name=spec["split_name"],
            split_json_fp=Path(spec["split_json_fp"]),
            force=force,
        )
        out[spec["split_name"]] = str(rd)
    print(json.dumps(out, indent=2))
    return out

def _fit_trackc_on_universe(*, universe_tag: str, force: bool = False):
    cfg_eff = _trackc_cfg_for_target(universe_tag=universe_tag, split_name=None, external=True)
    run_id = _trackc_run_id(universe_tag=universe_tag, split_name=None, external=True)
    run_dir = PROJ / "metrics" / "runs" / run_id
    model_dir = run_dir / "trackC_external" / universe_tag / "model"

    model_fp = model_dir / "model.pt"
    if model_fp.exists() and (not force):
        ckpt = torch.load(model_fp, map_location="cpu")
        model = TrackCModel(d_prot_in=ENZ_TOK_DIM, d_mol_in=SUB_TOK_DIM, cfg=ckpt["cfg"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        thresholds = {"t0p5": 0.5}
        thr_fp = model_dir / "val_threshold.json"
        if thr_fp.exists():
            obj = json.loads(thr_fp.read_text())
            if "t_val_f1" in obj:
                thresholds = {"t_val_f1": float(obj["t_val_f1"]), "t0p5": 0.5}
        return run_dir, model, thresholds

    pairs_u = _load_pairs_universe(universe_tag)
    pairs_u = globals()["trackc_attach_groups"](pairs_u)

    model, thresholds, hist = globals()["trackc_fit_model"](
        train_df=pairs_u,
        split_name="A1_external",
        run_dir=model_dir,
        cfg=cfg_eff,
    )

    _write_trackc_manifest(
        run_dir / "trackC_external" / universe_tag / "fit_manifest.json",
        {
            "track": "C_token_xattn_external_fit",
            "run_id": run_id,
            "universe": universe_tag,
            "cfg": cfg_eff,
            "token_cache_enz": str(ESMTOK_INDEX_FP),
            "token_cache_sub": str(MOLTOK_INDEX_FP),
            "thresholds": thresholds,
        },
    )
    return run_dir, model, thresholds

def run_trackc_external_suite(*, force: bool = False):
    universe_tags = list(globals().get("UNIVERSE_TAGS", ["trainpool", "multiplex", "mx_plus_gtpredict_pub", "gtpredict_pub"]))
    ext_tags = list(globals().get("EXT_TAGS", ["gasp", "avena", "lycium"]))
    ext_tags_by_universe = dict(globals().get("EXT_TAGS_BY_UNIVERSE", {
        "gtpredict_pub": ["gasp"],
        "mx_plus_gtpredict_pub": ["gasp"],
    }))

    run_dirs = {}
    for U in universe_tags:
        cfg_eff = _trackc_cfg_for_target(universe_tag=U, split_name=None, external=True)
        run_dir, model, thresholds = _fit_trackc_on_universe(universe_tag=U, force=force)
        ext_list = list(ext_tags_by_universe.get(U, ext_tags))

        for ext in ext_list:
            ext_fp = PROJ / "data" / f"ext_{ext}.parquet"
            if not ext_fp.exists():
                print(f"[Track C][warn] missing external dataset: {ext_fp}")
                continue

            df_ext = pd.read_parquet(ext_fp).reset_index(drop=True).copy()
            eval_dir = run_dir / "trackC_external" / U / f"ext_{ext}"
            if eval_dir.exists() and (not force):
                ok, missing = _trackc_eval_complete(eval_dir)
                if ok:
                    print(f"[Track C][skip] complete external bundle exists: {eval_dir}")
                    continue

            test_loader = globals()["trackc_make_loader"](
                df_ext,
                batch_size=int(cfg_eff["eval_batch_size"]),
                shuffle=False,
            )
            y_ext, w_ext, p_ext, _ = globals()["trackc_predict_model"](model, test_loader)

            TRACKC_eval_and_write(
                run_dir=run_dir,
                split_name=f"{U}__ext_{ext}",
                df_part=df_ext,
                y=y_ext,
                w=w_ext,
                p=p_ext,
                thresholds=thresholds,
                prefix=f"trackC_external/{U}/ext_{ext}",
            )

            ok, missing = _trackc_eval_complete(eval_dir)
            need(ok, f"[Track C][external {U}->{ext}] wrote incomplete bundle: {missing}")
            print(f"[Track C] external done: {U} -> ext_{ext}")

        run_dirs[U] = str(run_dir)

    _write_trackc_manifest(PROJ / "metrics" / "runs" / "trackC_external_runs_summary.json", run_dirs)
    print(json.dumps(run_dirs, indent=2))
    return run_dirs

# =============================================================================
# Phase 4C: MMVAE helpers extracted from Section 7 of the notebook.
#
# These functions and classes implement the multimodal supervised variational
# autoencoder (MMVAE) and associated training wrappers used in Phase 4C.
# They preserve the original notebook semantics but should be bound to
# notebook-level globals (e.g., VAE_CFG, apply_mode_profile, need) after
# import using the binder pattern described in the notebook.
# =============================================================================

# Default profile for MMVAE canonical configuration. This should match the
# notebook constant defined in Section 7.1.
MMVAE_CANONICAL_PROFILE = "decoupled_vae"

def mmvae_canonical_cfg(cfg: dict | None = None) -> dict:
    """
    Apply the canonical MMVAE profile to the provided VAE configuration.

    Parameters
    ----------
    cfg : dict, optional
        Base configuration dictionary to copy and update. If None, falls back to
        VAE_CFG from the notebook environment.

    Returns
    -------
    dict
        Configuration dictionary with 'mode_profile' set to the canonical
        profile and processed via apply_mode_profile.
    """
    # Copy the base VAE configuration; default to notebook's VAE_CFG.
    cfg = dict(globals().get("VAE_CFG") if cfg is None else cfg)
    cfg["mode_profile"] = MMVAE_CANONICAL_PROFILE
    # apply_mode_profile must be provided by the notebook environment.
    return globals()["apply_mode_profile"](cfg)

def mmvae_profile_cfg(profile: str, cfg: dict | None = None) -> dict:
    """
    Apply an arbitrary MMVAE profile to the provided VAE configuration.

    Parameters
    ----------
    profile : str
        Name of the mode profile to apply.
    cfg : dict, optional
        Base configuration dictionary to copy and update. If None, falls back to
        VAE_CFG from the notebook environment.

    Returns
    -------
    dict
        Configuration dictionary with 'mode_profile' set to the provided
        profile and processed via apply_mode_profile.
    """
    cfg = dict(globals().get("VAE_CFG") if cfg is None else cfg)
    cfg["mode_profile"] = str(profile).strip()
    return globals()["apply_mode_profile"](cfg)

def mmvae_load_model_bundle(run_dir: Path, *, device: str | None = None):
    """
    Load a saved MMVAE model bundle from an existing run directory.

    This utility reads the model configuration, the scaler, and the trained
    model weights from the run_dir and constructs a MultimodalSupervisedVAE
    instance with the appropriate dimensions.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory containing 'cfg.json' and the 'model' subdirectory.
    device : str, optional
        Device identifier (e.g. 'cpu' or 'cuda') on which to load the model.
        If None, defaults to DEVICE.

    Returns
    -------
    (model, scal, cfg) : tuple
        model : MultimodalSupervisedVAE
            The loaded and ready-to-evaluate model.
        scal : dict
            Dictionary containing scaler parameters and mode info.
        cfg : dict
            The configuration dictionary loaded from cfg.json.
    """
    run_dir = Path(run_dir)
    cfg_fp = run_dir / "cfg.json"
    # need() comes from the notebook environment; assert the presence of cfg.json.
    globals()["need"](cfg_fp.exists(), f"Missing cfg.json in run_dir: {cfg_fp}")
    cfg = json.loads(cfg_fp.read_text())

    # Ensure the MMVAE model class is available.
    globals()["need"]("MultimodalSupervisedVAE" in globals(), "MultimodalSupervisedVAE missing; run cell 7.2 before loading cached models.")
    device = device or DEVICE

    npz = np.load(run_dir / "model" / "scaler.npz", allow_pickle=True)
    scal = {k: npz[k] for k in npz.files}

    d_enz = int(np.asarray(scal.get("d_enz")).item())
    d_fp  = int(np.asarray(scal.get("d_fp")).item())

    model = MultimodalSupervisedVAE(
        d_enz=d_enz,
        d_fp=d_fp,
        z_dim=int(cfg["z_dim"]),
        h_dim=int(cfg["h_dim"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
    )
    state = torch.load(run_dir / "model" / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Normalize the scaler dictionary for consistency.
    scal["d_enz"] = d_enz
    scal["d_fp"]  = d_fp
    scal["enz_mu"] = np.asarray(scal.get("enz_mu"), dtype=np.float32)
    scal["enz_sd"] = np.asarray(scal.get("enz_sd"), dtype=np.float32)
    scal["mode"] = str(np.asarray(scal.get("mode")).item()) if "mode" in scal else "full"

    return model, scal, cfg

def _mmvae_mlp(d_in: int, h_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    """
    Build a simple multilayer perceptron used in MMVAE encoders and decoders.

    Parameters
    ----------
    d_in : int
        Input dimension.
    h_dim : int
        Hidden layer dimension.
    n_layers : int
        Number of hidden layers (must be >= 1).
    dropout : float
        Dropout probability.

    Returns
    -------
    torch.nn.Sequential
        A sequential MLP with ReLU activations and dropout.
    """
    if int(n_layers) < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    layers = []
    d = int(d_in)
    for _ in range(int(n_layers)):
        layers += [nn.Linear(d, int(h_dim)), nn.ReLU(), nn.Dropout(float(dropout))]
        d = int(h_dim)
    return nn.Sequential(*layers)

class MultimodalSupervisedVAE(nn.Module):
    """
    Minimal Section 7 dual‑tower supervised variational autoencoder.

    This implementation extends the single‑tower VAE by adding separate
    encoder and decoder towers for enzyme and substrate modalities, a shared
    latent representation, and a classifier head. Its forward method
    returns a dictionary matching the output contract of the Section 6 VAE.
    """
    def __init__(self, d_enz: int, d_fp: int, z_dim: int, h_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.d_enz = int(d_enz)
        self.d_fp  = int(d_fp)
        self.h_dim = int(h_dim)

        if (self.d_enz + self.d_fp) <= 0:
            raise ValueError("d_enz + d_fp must be > 0")

        # Encoders
        self.enc_enz = _mmvae_mlp(self.d_enz, h_dim, n_layers, dropout) if self.d_enz > 0 else None
        self.enc_fp  = _mmvae_mlp(self.d_fp,  h_dim, n_layers, dropout) if self.d_fp  > 0 else None

        self.has_both_modalities = bool(self.d_enz > 0 and self.d_fp > 0)
        self.fuse = (
            nn.Sequential(
                nn.Linear(2 * int(h_dim), int(h_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            )
            if self.has_both_modalities else None
        )

        self.mu = nn.Linear(int(h_dim), int(z_dim))
        self.logvar = nn.Linear(int(h_dim), int(z_dim))

        # Decoders (separate towers)
        self.dec_enz_tower = _mmvae_mlp(int(z_dim), h_dim, n_layers, dropout) if self.d_enz > 0 else None
        self.dec_fp_tower  = _mmvae_mlp(int(z_dim), h_dim, n_layers, dropout) if self.d_fp  > 0 else None

        self.dec_enz       = nn.Linear(int(h_dim), self.d_enz) if self.d_enz > 0 else None
        self.dec_fp_logits = nn.Linear(int(h_dim), self.d_fp)  if self.d_fp  > 0 else None

        # Classifier head
        self.cls = nn.Sequential(
            nn.Linear(int(z_dim), int(h_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(h_dim), 1),
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _split_modalities(self, x: torch.Tensor):
        if self.d_enz > 0 and self.d_fp > 0:
            x_enz = x[:, :self.d_enz]
            x_fp  = x[:, self.d_enz:self.d_enz + self.d_fp]
            return x_enz, x_fp
        if self.d_enz > 0:
            return x, None
        if self.d_fp > 0:
            return None, x
        raise RuntimeError("Both modalities absent.")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x_enz, x_fp = self._split_modalities(x)

        h_enz = self.enc_enz(x_enz) if (self.enc_enz is not None and x_enz is not None) else None
        h_fp  = self.enc_fp(x_fp)   if (self.enc_fp  is not None and x_fp  is not None) else None

        if h_enz is not None and h_fp is not None:
            return self.fuse(torch.cat([h_enz, h_fp], dim=1))
        if h_enz is not None:
            return h_enz
        if h_fp is not None:
            return h_fp
        raise RuntimeError("No encoder output produced.")

    def forward(self, x, *, sample_z: bool = True, cls_use_mu: bool = False):
        h = self._encode(x)
        mu, logvar = self.mu(h), self.logvar(h)

        z_sample = self.reparam(mu, logvar)
        z_dec = z_sample if sample_z else mu
        z_cls = mu if cls_use_mu else z_dec

        enz_hat = None
        fp_logits = None

        if self.dec_enz_tower is not None:
            enz_hat = self.dec_enz(self.dec_enz_tower(z_dec))
        if self.dec_fp_tower is not None:
            fp_logits = self.dec_fp_logits(self.dec_fp_tower(z_dec))

        y_logit = self.cls(z_cls).squeeze(1)

        return dict(
            mu=mu,
            logvar=logvar,
            z=z_dec,
            enz_hat=enz_hat,
            fp_logits=fp_logits,
            y_logit=y_logit,
        )

def train_multimodal_vae(X_train, y_train, w_train, cfg: dict, mode: str = "full", *, groups=None):
    """
    Train the multimodal supervised VAE on a single split and return the best model.

    This wrapper mirrors the notebook implementation with several important
    modifications:
      - Validation uses deterministic latent variables (sample_z=False) to
        avoid stochastic variability in AP/AUROC metrics.
      - Supports optional group-aware train/val splits when a `groups` array
        is provided (see _make_train_val_split).
      - The cfg key "train_sample_z" controls whether training samples latent
        variables; defaults to True.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix of shape (n_samples, n_features).
    y_train : array-like
        Binary target labels of shape (n_samples,).
    w_train : array-like
        Sample weights of shape (n_samples,).
    cfg : dict
        Configuration dictionary containing hyperparameters.
    mode : str, optional
        One of {"full", "enzyme_only", "substrate_only"} controlling how
        dimensions are partitioned for enzyme and substrate.
    groups : array-like, optional
        Group labels used for stratified group k-fold splitting.

    Returns
    -------
    model : MultimodalSupervisedVAE
        Trained model loaded to DEVICE and containing the best validation state.
    scal : dict
        Scaler dictionary with enzyme mean/std and dimension metadata.
    log_df : pandas.DataFrame
        DataFrame summarizing training and validation metrics per epoch.
    best_epoch : int
        Epoch number at which the best validation AP was observed.
    """
    # Disable deterministic algorithms for speed (matching notebook behavior).
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    set_seed(int(cfg["seed"]))

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    w_train = np.asarray(w_train, dtype=np.float32).reshape(-1)

    if len(X_train) != len(y_train) or len(X_train) != len(w_train):
        raise ValueError(f"Shape mismatch: X={len(X_train)}, y={len(y_train)}, w={len(w_train)}")

    d_in = int(X_train.shape[1])
    d_enz, d_fp = infer_dims(d_in, cfg, mode)

    # Train/val split (optionally group-aware)
    n = int(len(X_train))
    tr_idx, val_idx = _make_train_val_split(n, float(cfg["val_frac"]), int(cfg["seed"]), groups=groups)

    # Fit scaler on TRAIN split only; prep train/val
    enz_mu, enz_sd = fit_scaler(X_train[tr_idx], d_enz)
    Xtr = prep_X(X_train[tr_idx], d_enz, d_fp, enz_mu, enz_sd)
    Xva = prep_X(X_train[val_idx], d_enz, d_fp, enz_mu, enz_sd)

    ytr, wtr = y_train[tr_idx], w_train[tr_idx]
    yva, wva = y_train[val_idx], w_train[val_idx]

    ds_tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(wtr))
    ds_va = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva), torch.from_numpy(wva))
    dl_tr = DataLoader(ds_tr, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=int(cfg["batch_size"]), shuffle=False, drop_last=False)

    model = MultimodalSupervisedVAE(
        d_enz=int(d_enz),
        d_fp=int(d_fp),
        z_dim=int(cfg["z_dim"]),
        h_dim=int(cfg["h_dim"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["use_amp"]) and DEVICE == "cuda")

    best_ap = -1.0
    best_state = None
    best_epoch = 0
    bad = 0
    log_rows = []

    max_epochs = int(cfg["max_epochs"])
    warm = int(cfg["kl_warmup_epochs"])
    beta_max = float(cfg["beta_kl"])
    patience = int(cfg["patience"])
    train_sample_z = bool(cfg.get("train_sample_z", True))

    for epoch in range(1, max_epochs + 1):
        model.train()

        beta = beta_max * min(1.0, epoch / max(1, warm))

        losses = []
        for xb, yb, wb in dl_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            wb = wb.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=bool(cfg["use_amp"]) and DEVICE == "cuda"):
                out = model(
                    xb,
                    sample_z=train_sample_z,
                    cls_use_mu=bool(cfg.get("cls_use_mu", False)),
                )
                loss, cls_loss, rec_loss, kl_loss = compute_loss(
                    out, xb, yb, wb, cfg, d_enz, d_fp, beta=beta
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu().item()))

        # Validation (deterministic, use mu)
        model.eval()
        yhat, ytrue, wts = [], [], []
        with torch.no_grad():
            for xb, yb, wb in dl_va:
                xb = xb.to(DEVICE, non_blocking=True)
                out = model(xb, sample_z=False)  # deterministic validation
                p = torch.sigmoid(out["y_logit"]).detach().cpu().numpy()
                yhat.append(p)
                ytrue.append(yb.numpy())
                wts.append(wb.numpy())

        yhat = np.concatenate(yhat).reshape(-1)
        ytrue = np.concatenate(ytrue).reshape(-1)
        wts = np.concatenate(wts).reshape(-1)

        ap = float("nan")
        au = float("nan")
        if len(np.unique(ytrue)) > 1:
            ap = float(average_precision_score(ytrue, yhat, sample_weight=wts))
            au = float(roc_auc_score(ytrue, yhat, sample_weight=wts))

        log_rows.append(
            dict(epoch=int(epoch), loss=float(np.mean(losses)), val_ap=float(ap), val_auroc=float(au),
                 beta=float(beta), train_sample_z=bool(train_sample_z))
        )

        if np.isfinite(ap) and (ap > best_ap + 1e-6):
            best_ap = float(ap)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = int(epoch)
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    if best_state is None:
        raise AssertionError("Training failed: no best_state captured (val AP never finite/improving).")

    model.load_state_dict(best_state)

    scal = dict(
        enz_mu=np.asarray(enz_mu, dtype=np.float32),
        enz_sd=np.asarray(enz_sd, dtype=np.float32),
        d_enz=int(d_enz),
        d_fp=int(d_fp),
        mode=str(mode),
    )

    return model, scal, pd.DataFrame(log_rows), best_epoch

def retrain_mmvae_full_train(
    X_train,
    y_train,
    w_train,
    cfg: dict,
    best_epoch: int,
    scal: dict | None = None,
    mode: str = "full",
):
    """
    Retrain the MMVAE on 100% of the training data for exactly best_epoch epochs.

    This wrapper is identical to the notebook implementation except that it
    respects cfg["train_sample_z"] when determining whether to sample latent
    variables during retraining.
    """
    set_seed(int(cfg["seed"]))

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    w_train = np.asarray(w_train, dtype=np.float32).reshape(-1)

    if len(X_train) != len(y_train) or len(X_train) != len(w_train):
        raise ValueError(f"Shape mismatch: X={len(X_train)}, y={len(y_train)}, w={len(w_train)}")

    d_in = int(X_train.shape[1])
    d_enz, d_fp = infer_dims(d_in, cfg, mode)

    # Use provided scaler or compute on full train (enzyme dims only)
    if scal is None:
        enz_mu, enz_sd = fit_scaler(X_train, d_enz)
    else:
        enz_mu = np.asarray(scal.get("enz_mu"), dtype=np.float32)
        enz_sd = np.asarray(scal.get("enz_sd"), dtype=np.float32)

    X_prep = prep_X(X_train, d_enz, d_fp, enz_mu, enz_sd)

    ds = TensorDataset(torch.from_numpy(X_prep), torch.from_numpy(y_train), torch.from_numpy(w_train))
    dl = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=False)

    model = MultimodalSupervisedVAE(
        d_enz=int(d_enz),
        d_fp=int(d_fp),
        z_dim=int(cfg["z_dim"]),
        h_dim=int(cfg["h_dim"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["wd"]))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["use_amp"]) and DEVICE == "cuda")

    warm = int(cfg["kl_warmup_epochs"])
    beta_max = float(cfg["beta_kl"])
    train_sample_z = bool(cfg.get("train_sample_z", True))

    print(
        f"[RETRAIN] mode={mode} | d_in={d_in} (d_enz={d_enz}, d_fp={d_fp}) | "
        f"epochs={best_epoch} | samples={len(X_train)} | train_sample_z={train_sample_z}"
    )

    for epoch in range(1, int(best_epoch) + 1):
        model.train()
        beta = beta_max * min(1.0, epoch / max(1, warm))

        for xb, yb, wb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            wb = wb.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=bool(cfg["use_amp"]) and DEVICE == "cuda"):
                out = model(
                    xb,
                    sample_z=train_sample_z,
                    cls_use_mu=bool(cfg.get("cls_use_mu", False)),
                )
                loss, _, _, _ = compute_loss(out, xb, yb, wb, cfg, d_enz, d_fp, beta=beta)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    scal_out = dict(
        enz_mu=np.asarray(enz_mu, dtype=np.float32),
        enz_sd=np.asarray(enz_sd, dtype=np.float32),
        d_enz=int(d_enz),
        d_fp=int(d_fp),
        mode=str(mode),
    )

    return model, scal_out
