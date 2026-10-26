import argparse
import math
import os
import random
import time
import hashlib
from copy import deepcopy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from model import vision_transformer as vits
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from data.data_utils import StripDomainLabelDataset

from util.general_utils import init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model.utils import DINOHead, model_statistics
from config import *  # Dataset, pretrained-weight and experiment paths.

try:
    import psutil
except Exception:
    psutil = None


# =========================================================
#                 Determinism + utilities
# =========================================================

def robust_cholesky_spd(A: torch.Tensor, eps: float = 1e-6,
                        jitter_mult: float = 10.0, max_tries: int = 8):
    """
    Kept for compatibility / debugging (not used in FD path).
    """
    A = 0.5 * (A + A.T)
    d = A.shape[0]
    eye = torch.eye(d, device=A.device, dtype=A.dtype)
    diag = torch.diag(A)
    min_diag = diag.min()
    if min_diag <= 0:
        A = A + (-min_diag + eps) * eye
    diag_mean = torch.diag(A).mean()
    jitter = eps * diag_mean.clamp_min(1.0)
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(A + jitter * eye, check_errors=False)
        if int(info.item()) == 0:
            return L, float(jitter.item())
        jitter = jitter * jitter_mult
    L, info = torch.linalg.cholesky_ex(A + jitter * eye, check_errors=False)
    if int(info.item()) != 0:
        raise torch.linalg.LinAlgError(
            f"robust_cholesky_spd failed: info={int(info.item())}, last_jitter={float(jitter.item())}"
        )
    return L, float(jitter.item())


def loader_fingerprint(loader, take_batches=2):
    h = hashlib.sha1()
    it = iter(loader)
    for _ in range(take_batches):
        batch = next(it)
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = torch.as_tensor(x).detach().cpu().float()
        h.update(x[:8].contiguous().numpy().tobytes())
    return h.hexdigest()


def make_deterministic(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# =========================================================
#                 Memory tracking (from memory.py)
# =========================================================

def _bytes_to_mib(x: float) -> float:
    return float(x) / (1024.0 ** 2)


class MemoryTracker:
    """
    Track CUDA memory (allocated/reserved) + optional CPU RSS.
    Records min/max/avg over snapshots, plus CUDA peak stats.

    NOTE: snapshot() calls torch.cuda.synchronize() for accuracy; keep mem_every moderate.
    """
    def __init__(self, device="cuda:0", track_cpu=True):
        self.device = torch.device(device) if torch.cuda.is_available() else None
        self.track_cpu = track_cpu and (psutil is not None)
        self.proc = psutil.Process(os.getpid()) if self.track_cpu else None

        self.t = []
        self.cuda_alloc = []
        self.cuda_res = []
        self.cpu_rss = []

    def reset_cuda_peak(self):
        if self.device is None:
            return
        torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self):
        self.t.append(time.time())
        if self.device is not None:
            torch.cuda.synchronize(self.device)
            self.cuda_alloc.append(torch.cuda.memory_allocated(self.device))
            self.cuda_res.append(torch.cuda.memory_reserved(self.device))
        if self.proc is not None:
            self.cpu_rss.append(self.proc.memory_info().rss)

    def _stat(self, xs):
        if len(xs) == 0:
            return None
        return {"min": min(xs), "max": max(xs), "avg": sum(xs) / len(xs)}

    def report(self, prefix="", logger=None):
        def _fmt(stat):
            if stat is None:
                return "N/A"
            return f"min={_bytes_to_mib(stat['min']):.2f} MiB, avg={_bytes_to_mib(stat['avg']):.2f} MiB, max={_bytes_to_mib(stat['max']):.2f} MiB"

        alloc_stat = self._stat(self.cuda_alloc)
        res_stat = self._stat(self.cuda_res)
        cpu_stat = self._stat(self.cpu_rss)

        peak_alloc = peak_res = None
        if self.device is not None:
            peak_alloc = torch.cuda.max_memory_allocated(self.device)
            peak_res = torch.cuda.max_memory_reserved(self.device)

        lines = []
        lines.append(f"{prefix}[Mem] CUDA allocated: {_fmt(alloc_stat)}")
        lines.append(f"{prefix}[Mem] CUDA reserved : {_fmt(res_stat)}")
        if peak_alloc is not None:
            lines.append(f"{prefix}[Mem] CUDA peak allocated: {_bytes_to_mib(peak_alloc):.2f} MiB")
            lines.append(f"{prefix}[Mem] CUDA peak reserved : {_bytes_to_mib(peak_res):.2f} MiB")
        if cpu_stat is not None:
            lines.append(f"{prefix}[Mem] CPU RSS: {_fmt(cpu_stat)}")

        msg = "\n".join(lines)
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)


# =========================================================
#                 Student-t utilities
# =========================================================

def t_logpdf_diag(y, mu, Sigma_diag, nu, eps=1e-6):
    d = y.numel()
    s = Sigma_diag.clamp_min(eps)
    diff = y - mu
    mahal = (diff * diff / s).sum()
    logdet = torch.log(s).sum()
    const = (
        torch.lgamma(0.5 * (nu + d))
        - torch.lgamma(0.5 * nu)
        - 0.5 * (d * torch.log(nu * torch.tensor(math.pi, device=y.device)) + logdet)
    )
    return const - 0.5 * (nu + d) * torch.log1p(mahal / nu)


def t_logpdf_lowrank_iso(y: torch.Tensor,
                         mu: torch.Tensor,
                         nu: torch.Tensor,
                         sigma2: torch.Tensor,
                         U: torch.Tensor,
                         lam: torch.Tensor,
                         logdet: Optional[torch.Tensor] = None,
                         eps: float = 1e-6) -> torch.Tensor:
    """
    Student-t logpdf with Sigma = sigma2 I + U diag(lam) U^T, U^T U = I, lam>=0.
    Computes mahal via Woodbury and logdet via determinant lemma.
    """
    d = y.numel()
    x = y - mu
    sigma2 = sigma2.clamp_min(eps)

    # z = U^T x  (r,)
    # If U is all-zeros (early cluster), z will be 0, formula degenerates to isotropic.
    z = U.transpose(0, 1) @ x  # (r,)

    denom = (sigma2 + lam).clamp_min(eps)  # (r,)
    w = lam / denom                         # (r,)
    norm2 = (x * x).sum()
    mahal = (norm2 - (w * (z * z)).sum()) / sigma2

    if logdet is None:
        r = lam.numel()
        logdet = (d - r) * torch.log(sigma2) + torch.log(denom).sum()

    const = (
        torch.lgamma(0.5 * (nu + d))
        - torch.lgamma(0.5 * nu)
        - 0.5 * (d * torch.log(nu * torch.tensor(math.pi, device=y.device)) + logdet)
    )
    return const - 0.5 * (nu + d) * torch.log1p(mahal / nu)


# =========================================================
#               Frequent Directions (FD) sketch
# =========================================================

class FrequentDirections:
    """
    FD sketch for stream of vectors x in R^d.
    Maintains B in R^{ell x d} such that B^T B approximates sum x x^T.

    Storage: O(ell*d) float32.
    Update:
      - Insert x as a new row; when full, compress via SVD with shrinkage.
    Additionally tracks:
      - energy = sum ||x||^2  (true trace of scatter)
      - count = number of inserted vectors
    """
    def __init__(self, d: int, ell: int, device: torch.device, eps: float = 1e-8):
        self.d = int(d)
        self.ell = int(ell)
        self.device = device
        self.eps = float(eps)

        self.B = torch.zeros(self.ell, self.d, device=self.device, dtype=torch.float32)
        self.fill = 0
        self.energy = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.count = 0

    @torch.no_grad()
    def insert(self, x: torch.Tensor):
        x = x.to(self.device, dtype=torch.float32)
        self.energy += (x * x).sum()
        self.count += 1

        if self.fill < self.ell:
            self.B[self.fill].copy_(x)
            self.fill += 1
            return

        self._compress()
        # write new vector into last row
        self.B[-1].copy_(x)
        self.fill = self.ell

    @torch.no_grad()
    def _compress(self):
        # SVD of (ell,d), ell small. Use float32.
        U, S, Vh = torch.linalg.svd(self.B, full_matrices=False)  # S: (ell,)
        delta = (S[-1] ** 2).clamp_min(self.eps)
        S2 = (S * S - delta).clamp_min(0.0)
        S_shrink = torch.sqrt(S2)
        self.B = torch.diag(S_shrink) @ Vh  # (ell,d)

    @torch.no_grad()
    def top_eigs(self, r: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (U_r, eta_r) where:
          - U_r: (d,r) orthonormal (right singular vectors)
          - eta_r: (r,) eigenvalues of B^T B (approx scatter eigenvalues)
        """
        r = int(r)
        if r <= 0:
            return torch.zeros(self.d, 0, device=self.device, dtype=torch.float32), torch.zeros(0, device=self.device, dtype=torch.float32)

        # SVD(B): B = U S V^T => B^T B = V diag(S^2) V^T
        U_b, S, Vh = torch.linalg.svd(self.B, full_matrices=False)
        V = Vh.transpose(0, 1)  # (d, ell)
        r_eff = min(r, V.shape[1])
        U_r = V[:, :r_eff].contiguous()
        eta_r = (S[:r_eff] ** 2).contiguous()
        if r_eff < r:
            padU = torch.zeros(self.d, r - r_eff, device=self.device, dtype=torch.float32)
            padE = torch.zeros(r - r_eff, device=self.device, dtype=torch.float32)
            U_r = torch.cat([U_r, padU], dim=1)
            eta_r = torch.cat([eta_r, padE], dim=0)
        return U_r, eta_r

    @torch.no_grad()
    def lowrank_iso_scatter(self, r: int, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Approximate scatter S = sum x x^T as:
          S ≈ sigma2 * I + U diag(lam) U^T
        with:
          - sigma2 = tail_mean (scatter-level) scalar
          - U: (d,r)
          - lam: (r,) nonnegative "excess" over sigma2
        """
        d = self.d
        r = int(r)
        eps = float(eps)
        if r <= 0:
            sigma2 = (self.energy / max(d, 1)).clamp_min(eps)
            U = torch.zeros(d, 0, device=self.device, dtype=torch.float32)
            lam = torch.zeros(0, device=self.device, dtype=torch.float32)
            return sigma2, U, lam

        U_r, eta_r = self.top_eigs(r)

        # sigma2 from true energy - captured energy
        tail = (self.energy - eta_r.sum()).clamp_min(0.0)
        denom = max(d - r, 1)
        sigma2 = (tail / denom).clamp_min(eps)  # scatter-level isotropic
        lam = (eta_r - sigma2).clamp_min(0.0)   # excess
        return sigma2, U_r, lam


# =========================================================
#                 Low-rank+iso combination helper
# =========================================================

@torch.no_grad()
def decompose_psd_lowrank_iso_full(A: torch.Tensor, r: int, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose PSD matrix A (d,d) into:
      A ≈ sigma2 I + U diag(lam) U^T
    using full eigen decomposition ONCE (d=768 is acceptable).
    Returns:
      sigma2 (scalar), U (d,r), lam (r,) "excess" over sigma2
    """
    A = 0.5 * (A + A.T)
    d = A.shape[0]
    r = int(r)
    if r <= 0:
        sigma2 = (torch.trace(A) / max(d, 1)).clamp_min(eps)
        U = torch.zeros(d, 0, device=A.device, dtype=torch.float32)
        lam = torch.zeros(0, device=A.device, dtype=torch.float32)
        return sigma2, U, lam

    # eigh returns ascending
    evals, evecs = torch.linalg.eigh(A.to(torch.float32))
    evals = evals.flip(0)
    evecs = evecs[:, torch.arange(d - 1, -1, -1, device=A.device)]

    r_eff = min(r, d)
    head = evals[:r_eff]
    if r_eff < d:
        sigma2 = evals[r_eff:].mean().clamp_min(eps)
    else:
        sigma2 = torch.tensor(eps, device=A.device, dtype=torch.float32)

    lam = (head - sigma2).clamp_min(0.0)
    U = evecs[:, :r_eff].contiguous()
    if r_eff < r:
        padU = torch.zeros(d, r - r_eff, device=A.device, dtype=torch.float32)
        padL = torch.zeros(r - r_eff, device=A.device, dtype=torch.float32)
        U = torch.cat([U, padU], dim=1)
        lam = torch.cat([lam, padL], dim=0)
    return sigma2, U, lam


@torch.no_grad()
def combine_lowrank_iso(
    d: int,
    r: int,
    sigma_iso: torch.Tensor,
    comps: List[Tuple[torch.Tensor, torch.Tensor]],  # [(U_i(d,ri), lam_i(ri,)), ...] lam is "excess"
    rank1: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (v(d,), gamma(scalar))
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combine:
      Psi ≈ sigma_iso I + sum_i U_i diag(lam_i) U_i^T + gamma * v v^T (optional)
    Then compress into rank-r + isotropic:
      Psi ≈ sigma2 I + U diag(lam) U^T
    via small-subspace eigendecomposition.

    Returns:
      sigma2 (scalar), U(d,r), lam(r,) (excess over sigma2)
    """
    device = sigma_iso.device
    r = int(r)
    d = int(d)
    eps = float(eps)

    if r <= 0:
        sigma2 = sigma_iso.clamp_min(eps)
        U = torch.zeros(d, 0, device=device, dtype=torch.float32)
        lam = torch.zeros(0, device=device, dtype=torch.float32)
        return sigma2, U, lam

    cols = []
    for (Ui, lami) in comps:
        if Ui is None or lami is None:
            continue
        if Ui.numel() == 0 or lami.numel() == 0:
            continue
        # even if some lami are 0, keep Ui; QR handles it
        cols.append(Ui.to(torch.float32))

    if rank1 is not None:
        v, gamma = rank1
        if v is not None and gamma is not None:
            v = v.to(torch.float32)
            if v.numel() == d and float(gamma.item()) > 0:
                cols.append(v.view(d, 1))

    if len(cols) == 0:
        sigma2 = sigma_iso.clamp_min(eps)
        U = torch.zeros(d, r, device=device, dtype=torch.float32)
        lam = torch.zeros(r, device=device, dtype=torch.float32)
        return sigma2, U, lam

    A = torch.cat(cols, dim=1)  # (d, m0)
    # QR to orthonormal basis
    Q, _ = torch.linalg.qr(A, mode="reduced")  # (d,m)
    m = Q.shape[1]

    # Build small matrix M = Q^T [sum lowrank terms] Q  (size m x m)
    M = torch.zeros(m, m, device=device, dtype=torch.float32)

    # add each lowrank component
    for (Ui, lami) in comps:
        if Ui is None or lami is None:
            continue
        Ui = Ui.to(torch.float32)
        lami = lami.to(torch.float32)
        if Ui.numel() == 0 or lami.numel() == 0:
            continue
        P = Q.transpose(0, 1) @ Ui  # (m, ri)
        # M += P diag(lam) P^T
        M = M + P @ (lami.view(-1, 1) * P.transpose(0, 1))

    # add rank-1
    if rank1 is not None:
        v, gamma = rank1
        if v is not None and gamma is not None and float(gamma.item()) > 0:
            pv = Q.transpose(0, 1) @ v.to(torch.float32)  # (m,)
            M = M + gamma.to(torch.float32) * torch.ger(pv, pv)

    # eigendecompose small M
    evals, evecs = torch.linalg.eigh(M)  # ascending
    evals = evals.flip(0)
    evecs = evecs[:, torch.arange(m - 1, -1, -1, device=device)]

    r_eff = min(r, m)
    head = evals[:r_eff]
    tail_sum = evals[r_eff:].sum() if r_eff < m else torch.tensor(0.0, device=device, dtype=torch.float32)
    tail_avg = (tail_sum / max(d - r_eff, 1)).clamp_min(0.0)

    # final isotropic and excess
    sigma2 = (sigma_iso + tail_avg).clamp_min(eps)
    lam = (head - tail_avg).clamp_min(0.0)

    U = (Q @ evecs[:, :r_eff]).contiguous()  # (d, r_eff)
    if r_eff < r:
        padU = torch.zeros(d, r - r_eff, device=device, dtype=torch.float32)
        padL = torch.zeros(r - r_eff, device=device, dtype=torch.float32)
        U = torch.cat([U, padU], dim=1)
        lam = torch.cat([lam, padL], dim=0)

    return sigma2, U, lam


# =========================================================
#                 Shared prior container
# =========================================================

class SharedNIWPrior:
    """
    A device-resident shared NIW prior.

    FD mode requirements:
      - store Psi0 in full fp32 (precision)
      - also store a lowrank+iso decomposition of Psi0:
          Psi0 ≈ psi0_sigma2 I + psi0_U diag(psi0_lam) psi0_U^T
    """
    def __init__(self, mu0, kappa0, Psi0, nu0, cov_mode="full"):
        self.mu0 = mu0.to(torch.float32)  # (d,)
        self.kappa0 = kappa0 if torch.is_tensor(kappa0) else torch.tensor(kappa0, device=mu0.device, dtype=torch.float32)
        self.nu0 = nu0 if torch.is_tensor(nu0) else torch.tensor(nu0, device=mu0.device, dtype=torch.float32)

        self.cov_mode = str(cov_mode).lower()
        self.d = int(self.mu0.numel())

        if self.cov_mode == "diag":
            if Psi0.ndim == 2:
                Psi0 = torch.diag(Psi0)
            self.Psi0 = Psi0.to(torch.float32)  # (d,)
        else:
            # FD: keep full fp32 Psi0 for once-off decomposition; later only use decomposed form
            self.Psi0 = Psi0.to(torch.float32)  # (d,d)

        # FD decomposition placeholders (used only when cov_mode=="full")
        self.psi0_sigma2: Optional[torch.Tensor] = None
        self.psi0_U: Optional[torch.Tensor] = None
        self.psi0_lam: Optional[torch.Tensor] = None
        self.psi0_trace: Optional[torch.Tensor] = None

    @classmethod
    def from_prior_dict(cls, prior: dict, device="cuda:0", cov_mode="full"):
        device = torch.device(device)
        cov_mode = str(cov_mode).lower()

        mu0 = prior["mu0"].to(device, dtype=torch.float32)
        kappa0 = torch.tensor(prior["kappa0"], device=device, dtype=torch.float32)
        nu0 = torch.tensor(prior["nu0"], device=device, dtype=torch.float32)

        Psi0 = prior["Psi0"].to(device)
        return cls(mu0=mu0, kappa0=kappa0, Psi0=Psi0, nu0=nu0, cov_mode=cov_mode)

    @torch.no_grad()
    def ensure_psi0_fd_decomp(self, r: int, eps: float = 1e-8):
        """
        Compute (once) Psi0 ≈ sigma2 I + U diag(lam) U^T.
        """
        if self.cov_mode != "full":
            return
        if self.psi0_sigma2 is not None:
            return
        sigma2, U, lam = decompose_psd_lowrank_iso_full(self.Psi0, r=r, eps=eps)
        self.psi0_sigma2 = sigma2
        self.psi0_U = U
        self.psi0_lam = lam
        self.psi0_trace = torch.trace(self.Psi0).to(torch.float32)


# =========================================================
#      Cluster: FD sketch + cached predictive lowrank+iso
# =========================================================

class Cluster:
    """
    Per-cluster persistent tensors:
      - n (scalar fp32)
      - m (d fp32)
      - FD sketch: B (ell,d fp32), energy (scalar), count
      - cached predictive params (FD approx, full mode):
          mu (d,)
          nu_pred (scalar)
          sigma2_pred (scalar)
          lam_pred (r,)
          U_pred (d,r)
          logdet_pred (scalar)
    """
    def __init__(self,
                 d: int,
                 prior: SharedNIWPrior,
                 cov_mode: str = "full",
                 device: str = "cuda:0",
                 fd_rank: int = 16,
                 fd_ell: int = 32,
                 fd_refresh_every: int = 1,
                 fd_min_n: int = 8,
                 eps: float = 1e-6):
        self.prior = prior
        self.cov_mode = str(cov_mode).lower()
        self.device = torch.device(device)
        assert self.cov_mode == prior.cov_mode, "Cluster cov_mode must match SharedNIWPrior cov_mode."

        self.d = int(d)
        self.r = int(fd_rank)
        self.fd_refresh_every = int(max(fd_refresh_every, 1))
        self.fd_min_n = int(max(fd_min_n, 0))
        self.eps = float(eps)

        self.n = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.m = torch.zeros(self.d, device=self.device, dtype=torch.float32)

        # diag-mode exact scatter (vector)
        self.S_diag = torch.zeros(self.d, device=self.device, dtype=torch.float32) if self.cov_mode == "diag" else None

        # full-mode FD sketch
        self.fd = FrequentDirections(d=self.d, ell=int(fd_ell), device=self.device, eps=1e-8) if self.cov_mode == "full" else None
        self._refresh_counter = 0

        # cached predictive
        self.mu = self.prior.mu0.clone()
        self.nu_pred = torch.tensor(1.0, device=self.device, dtype=torch.float32)

        # FD cached predictive lowrank+iso
        self.sigma2_pred = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        self.U_pred = torch.zeros(self.d, self.r, device=self.device, dtype=torch.float32)
        self.lam_pred = torch.zeros(self.r, device=self.device, dtype=torch.float32)
        self.logdet_pred = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        self.refresh_cache(force=True)

    @torch.no_grad()
    def update_with_point(self, y: torch.Tensor):
        """
        Update cluster sufficient stats.
        - Mean update via Welford.
        - Scatter update:
            diag mode: exact Welford diag scatter increment.
            full mode: insert residual vector into FD sketch (approx scatter).
        Then refresh predictive cache.
        """
        y = y.to(self.device, dtype=torch.float32)

        n_new = self.n + 1.0
        delta = y - self.m
        m_new = self.m + delta / n_new
        delta2 = y - m_new

        if self.cov_mode == "diag":
            # exact diag Welford
            self.S_diag = self.S_diag + (delta * delta2)
        else:
            # FD uses approximate centered residual; use updated mean
            x = delta * torch.sqrt(self.n / n_new)
            self.fd.insert(x)

        self.n = n_new
        self.m = m_new

        self._refresh_counter += 1
        if self._refresh_counter >= self.fd_refresh_every:
            self.refresh_cache(force=True)
            self._refresh_counter = 0
        else:
            # still update mu/nu_pred cheaply (cov cache can be slightly stale)
            self.refresh_cache(force=False)

    @torch.no_grad()
    def refresh_cache(self, force: bool = True):
        """
        Recompute predictive Student-t parameters.
        - mu and nu_pred always updated (cheap).
        - cov cache updated if force==True (or diag mode which is cheap anyway).
        """
        mu0 = self.prior.mu0
        kappa0 = self.prior.kappa0
        nu0 = self.prior.nu0

        n = self.n
        m = self.m

        kappa = kappa0 + n
        mu = (kappa0 * mu0 + n * m) / kappa
        nu = nu0 + n

        d = self.d
        nu_pred = nu - d + 1.0
        c_pred = (kappa + 1.0) / (kappa * nu_pred)

        self.mu = mu
        self.nu_pred = nu_pred

        if self.cov_mode == "diag":
            # exact diag NIW
            Psi0_diag = self.prior.Psi0.to(torch.float32)  # (d,)
            diff = (m - mu0)
            coef = (kappa0 * n / kappa)
            Psi = Psi0_diag + self.S_diag.to(torch.float32) + coef * (diff * diff)
            Sigma_diag = c_pred * Psi
            logdet = torch.log(Sigma_diag.clamp_min(self.eps)).sum()
            # set as "rank-0 lowrank iso" for uniform interface
            self.sigma2_pred = torch.tensor(1.0, device=self.device, dtype=torch.float32)  # unused
            self.U_pred.zero_()
            self.lam_pred.zero_()
            self.logdet_pred = logdet
            return

        # full (FD) mode
        if (not force):
            return

        # ensure prior decomp is ready
        self.prior.ensure_psi0_fd_decomp(r=self.r, eps=1e-8)
        psi0_sigma2 = self.prior.psi0_sigma2
        psi0_U = self.prior.psi0_U
        psi0_lam = self.prior.psi0_lam

        # cluster scatter decomp from FD
        # if too few points, don't trust FD: use "no data scatter" (sigma_s=0, lam_s=0)
        if int(self.n.item()) >= self.fd_min_n and self.fd is not None:
            s_sigma2, s_U, s_lam = self.fd.lowrank_iso_scatter(r=self.r, eps=1e-8)
        else:
            s_sigma2 = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            s_U = torch.zeros(self.d, self.r, device=self.device, dtype=torch.float32)
            s_lam = torch.zeros(self.r, device=self.device, dtype=torch.float32)

        # rank-1 NIW mean correction term: coef * diff diff^T
        diff = (m - mu0).to(torch.float32)
        coef = (kappa0 * n / kappa).to(torch.float32)  # scalar
        diff_norm2 = (diff * diff).sum()
        if float(diff_norm2.item()) > 0.0 and float(coef.item()) > 0.0:
            v = diff / torch.sqrt(diff_norm2.clamp_min(1e-12))
            gamma = (coef * diff_norm2).clamp_min(0.0)
            rank1 = (v, gamma)
        else:
            rank1 = None

        # Psi ≈ (psi0_sigma2 + s_sigma2) I + lowrank parts + rank1
        sigma_iso = (psi0_sigma2 + s_sigma2).to(torch.float32)

        sigma2_psi, U_psi, lam_psi = combine_lowrank_iso(
            d=self.d,
            r=self.r,
            sigma_iso=sigma_iso,
            comps=[(psi0_U, psi0_lam), (s_U, s_lam)],
            rank1=rank1,
            eps=1e-8
        )

        # predictive Sigma = c_pred * Psi
        sigma2_pred = (c_pred * sigma2_psi).clamp_min(self.eps)
        lam_pred = (c_pred * lam_psi).clamp_min(0.0)
        U_pred = U_psi  # already orthonormal

        # cache logdet for speed
        denom = (sigma2_pred + lam_pred).clamp_min(self.eps)
        logdet = (self.d - self.r) * torch.log(sigma2_pred) + torch.log(denom).sum()

        self.sigma2_pred = sigma2_pred
        self.lam_pred = lam_pred
        self.U_pred = U_pred
        self.logdet_pred = logdet

    @torch.no_grad()
    def predictive_logpdf(self, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        log p(y | D_k) under predictive Student-t.
        full: FD lowrank+iso (Woodbury)
        diag: exact diag Student-t
        """
        y = y.to(self.device, dtype=torch.float32)
        if self.cov_mode == "diag":
            # rebuild Sigma_diag (exact) O(d)
            mu0 = self.prior.mu0
            kappa0 = self.prior.kappa0
            Psi0 = self.prior.Psi0.to(torch.float32)
            nu0 = self.prior.nu0

            n = self.n
            m = self.m
            kappa = kappa0 + n
            nu = nu0 + n

            d = y.numel()
            nu_pred = nu - d + 1.0
            c_pred = (kappa + 1.0) / (kappa * nu_pred)
            diff = (m - mu0)
            coef = (kappa0 * n / kappa)

            Psi = Psi0 + self.S_diag.to(torch.float32) + coef * (diff * diff)
            Sigma_diag = c_pred * Psi
            return t_logpdf_diag(y, self.mu, Sigma_diag, self.nu_pred, eps=eps)

        # full FD: use cached sigma2/U/lam/logdet
        return t_logpdf_lowrank_iso(
            y=y,
            mu=self.mu,
            nu=self.nu_pred,
            sigma2=self.sigma2_pred,
            U=self.U_pred,
            lam=self.lam_pred,
            logdet=self.logdet_pred,
            eps=eps
        )


@torch.no_grad()
def spawn_new_cluster(y: torch.Tensor, prior_dev: SharedNIWPrior, args, device="cuda:0") -> Cluster:
    cov_mode = str(getattr(args, "cov_mode", "full")).lower()
    d = int(y.numel())
    cl = Cluster(
        d=d,
        prior=prior_dev,
        cov_mode=cov_mode,
        device=str(device),
        fd_rank=int(getattr(args, "fd_rank", 16)),
        fd_ell=int(getattr(args, "fd_ell", 32)),
        fd_refresh_every=int(getattr(args, "fd_refresh_every", 1)),
        fd_min_n=int(getattr(args, "fd_min_n", 8)),
        eps=float(getattr(args, "eps", 1e-6)),
    )
    cl.update_with_point(y)
    return cl


# =========================================================
#   Efficient best-old selection (chunked)
# =========================================================

@torch.no_grad()
def best_old_cluster_logscore(
    y: torch.Tensor,
    clusters: List[Cluster],
    beta: float,
    temperature_inv: float,
    cov_mode: str,
    eps: float,
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, int]:
    """
    Return (best_log_score, best_idx) over old clusters.
    Uses chunking to control temporary memory.
    """
    if len(clusters) == 0:
        return torch.tensor(-1e9, device=y.device, dtype=torch.float32), -1

    device = clusters[0].device
    y = y.to(device, dtype=torch.float32)

    best_val = None
    best_idx = -1

    K = len(clusters)
    d = y.numel()

    for s in range(0, K, chunk_size):
        sub = clusters[s:s + chunk_size]
        c = len(sub)

        n = torch.stack([cl.n for cl in sub], dim=0).to(torch.float32)  # (c,)
        logn = torch.log(n.clamp_min(1.0))

        if cov_mode == "diag":
            logp = torch.stack([cl.predictive_logpdf(y, eps=eps) for cl in sub], dim=0)  # (c,)
        else:
            # vectorized lowrank+iso logpdf over chunk
            mu = torch.stack([cl.mu for cl in sub], dim=0)                  # (c,d)
            nu = torch.stack([cl.nu_pred for cl in sub], dim=0)             # (c,)
            sigma2 = torch.stack([cl.sigma2_pred for cl in sub], dim=0)     # (c,)
            lam = torch.stack([cl.lam_pred for cl in sub], dim=0)           # (c,r)
            U = torch.stack([cl.U_pred for cl in sub], dim=0)               # (c,d,r)
            logdet = torch.stack([cl.logdet_pred for cl in sub], dim=0)     # (c,)

            diff = y.unsqueeze(0) - mu                                      # (c,d)
            # z = U^T diff : (c,r)
            z = torch.einsum("cdr,cd->cr", U, diff)

            denom = (sigma2.unsqueeze(-1) + lam).clamp_min(eps)             # (c,r)
            w = lam / denom                                                 # (c,r)
            norm2 = (diff * diff).sum(dim=1)                                # (c,)
            mahal = (norm2 - (w * (z * z)).sum(dim=1)) / sigma2.clamp_min(eps)

            const = (
                torch.lgamma(0.5 * (nu + d))
                - torch.lgamma(0.5 * nu)
                - 0.5 * (d * torch.log(nu * torch.tensor(math.pi, device=device)) + logdet)
            )
            logp = const - 0.5 * (nu + d) * torch.log1p(mahal / nu)         # (c,)

        scores = beta * logn + temperature_inv * logp
        chunk_best_val, chunk_best_pos = torch.max(scores, dim=0)
        chunk_best_idx = s + int(chunk_best_pos.item())

        if (best_val is None) or (chunk_best_val > best_val):
            best_val = chunk_best_val
            best_idx = chunk_best_idx

    return best_val, best_idx


# =========================================================
#               BDP_init (Empirical Bayes)
#               (kept structurally similar; no arg deletions)
# =========================================================

@torch.no_grad()
def BDP_init(model, data_loader, args):
    device = torch.device("cuda:0")
    model = model.to(device).eval()

    mem = MemoryTracker(device=str(device), track_cpu=True)
    mem.reset_cuda_peak()
    mem.snapshot()

    cov_mode = str(getattr(args, "cov_mode", "full")).lower()
    return_full_matrix = bool(getattr(args, "return_full_matrix", True))
    standardize = bool(getattr(args, "standardize", False))

    n0 = float(getattr(args, "n0", 10.0))
    kappa0_init = getattr(args, "kappa0_init", None)
    kappa0_strategy = getattr(args, "kappa0_strategy", None)
    shrink_lambda = float(getattr(args, "shrink_lambda", 0.0))

    d = int(args.feat_dim)

    # NOTE: keep packed args for compatibility, but FD full mode ignores them for precision.
    use_packed = bool(getattr(args, "symm_packed", 1)) and (cov_mode == "full")
    cov_store_dtype = str(getattr(args, "cov_store_dtype", "fp32"))

    if cov_mode == "full":
        if getattr(args, "logger", None) is not None:
            args.logger.info(f"[FD] full-cov uses FD sketch in fp32. symm_packed={use_packed} and cov_store_dtype={cov_store_dtype} are kept for compatibility but NOT used in FD path.")
        else:
            print(f"[FD] full-cov uses FD sketch in fp32. symm_packed={use_packed} and cov_store_dtype={cov_store_dtype} kept but NOT used.")

    # 1) optional global standardization (two-pass)
    sum_y = torch.zeros(d, device=device)
    sum_y2 = torch.zeros(d, device=device)
    N_total = 0

    if standardize:
        for batch in data_loader:
            images, class_labels, uq_idxs = batch
            images = images.to(device, non_blocking=True)
            feats, _ = model(images)
            feats = feats.detach()
            sum_y += feats.sum(dim=0)
            sum_y2 += (feats ** 2).sum(dim=0)
            N_total += feats.shape[0]
        mean_all = sum_y / max(N_total, 1)
        var_all = (sum_y2 / max(N_total, 1)) - mean_all ** 2
        std_all = torch.sqrt(var_all.clamp_min(1e-12))
    else:
        mean_all = None
        std_all = None

    # 2) class-wise stats for EB prior
    # We keep class mean and scatter to compute Sigma_within & Sigma_between.
    # This is done only once; memory is manageable (C is small).
    class_n = defaultdict(int)
    class_mu = dict()
    class_S = dict()

    def _init_class_buffers():
        if cov_mode == "full":
            return torch.zeros(d, device=device, dtype=torch.float32), torch.zeros(d, d, device=device, dtype=torch.float32)
        else:
            return torch.zeros(d, device=device, dtype=torch.float32), torch.zeros(d, device=device, dtype=torch.float32)

    def _batch_scatter(X):
        xb = X.mean(dim=0)
        diff = X - xb
        if cov_mode == "full":
            Sb = diff.T @ diff  # (d,d)
        else:
            Sb = (diff ** 2).sum(dim=0)  # (d,)
        return xb, Sb, X.shape[0]

    def _merge_welford(n, m, C, nb, mb, Sb):
        if n == 0:
            return nb, mb, Sb
        n_new = n + nb
        delta = (mb - m)

        if cov_mode == "full":
            C_new = C + Sb + torch.ger(delta, delta) * (n * nb / n_new)
        else:
            C_new = C + Sb + (delta * delta) * (n * nb / n_new)

        m_new = m + delta * (nb / n_new)
        return n_new, m_new, C_new

    for batch in data_loader:
        images, class_labels, uq_idxs = batch
        images = images.to(device, non_blocking=True)
        class_labels = class_labels.to(device, non_blocking=True)

        feats, _ = model(images)
        X = feats.detach()
        if standardize:
            X = (X - mean_all) / std_all.clamp_min(1e-12)

        for c in class_labels.unique():
            c = int(c.item())
            idx = (class_labels == c)
            Xc = X[idx]
            if Xc.numel() == 0:
                continue
            if c not in class_mu:
                m_c, S_c = _init_class_buffers()
                class_mu[c] = m_c.clone()
                class_S[c] = S_c.clone()
                class_n[c] = 0

            mb, Sb, nb = _batch_scatter(Xc)
            n_old, m_old, S_old = class_n[c], class_mu[c], class_S[c]
            n_new, m_new, S_new = _merge_welford(n_old, m_old, S_old, nb, mb, Sb)
            class_n[c], class_mu[c], class_S[c] = n_new, m_new, S_new

    classes = sorted(class_n.keys())
    C = len(classes)
    assert C > 0, "No labeled classes found in data_loader."
    N = sum(class_n[c] for c in classes)

    # 3) global mean
    mu_global = torch.zeros(d, device=device, dtype=torch.float32)
    for c in classes:
        mu_global += class_mu[c] * float(class_n[c])
    mu_global = mu_global / max(N, 1)

    # 4) within / between covariance
    if cov_mode == "full":
        Sw_sum = torch.zeros(d, d, device=device, dtype=torch.float32)
        for c in classes:
            Sw_sum += class_S[c].to(torch.float32)
        Sigma_within = Sw_sum / max(N - C, 1)

        Sb_sum = torch.zeros(d, d, device=device, dtype=torch.float32)
        for c in classes:
            delta = (class_mu[c] - mu_global)
            weight = class_n[c] if kappa0_strategy is None else 1.0
            Sb_sum += weight * torch.ger(delta, delta)
        Sigma_between = Sb_sum / max(C - 1, 1)
    else:
        Sw_sum = torch.zeros(d, device=device, dtype=torch.float32)
        for c in classes:
            Sw_sum += class_S[c].to(torch.float32)
        Sigma_within = Sw_sum / max(N - C, 1)

        Sb_sum = torch.zeros(d, device=device, dtype=torch.float32)
        for c in classes:
            delta = (class_mu[c] - mu_global)
            weight = class_n[c] if kappa0_strategy is None else 1.0
            Sb_sum += weight * delta * delta
        Sigma_between = Sb_sum / max(C - 1, 1)

    if shrink_lambda > 0.0 and cov_mode == "full":
        diagW = torch.diag(torch.diag(Sigma_within))
        Sigma_within = (1.0 - shrink_lambda) * Sigma_within + shrink_lambda * diagW

    # 5) NIW hyperparams
    mu0 = mu_global.clone()
    nu0 = float(d + 1 + max(n0, 1.0))
    Psi0 = Sigma_within * (nu0 - d - 1.0)

    if kappa0_init is not None:
        kappa0 = float(kappa0_init)
    else:
        if kappa0_strategy is None:
            avg_n = N / float(C)
            mean_inv_n = 1.0 / max(avg_n, 1.0)
        else:
            mean_inv_n = sum(1.0 / class_n[c] for c in classes) / float(C)
        if cov_mode == "full":
            tr_within = torch.trace(Sigma_within).item()
            tr_between = torch.trace(Sigma_between).item()
        else:
            tr_within = Sigma_within.sum().item()
            tr_between = Sigma_between.sum().item()
        denom = max(tr_between - mean_inv_n * tr_within, 1e-8)
        kappa0 = float(tr_within / denom)

    # 6) move to CPU for prior dict
    mu0_cpu = mu0.detach().cpu()
    Sigma_within_cpu = Sigma_within.detach().cpu()
    Sigma_between_cpu = Sigma_between.detach().cpu()

    Psi0_cpu = Psi0.detach().cpu()
    if cov_mode == "full":
        if not return_full_matrix:
            Psi0_cpu = torch.diag(Psi0_cpu)
    else:
        if return_full_matrix:
            Psi0_cpu = torch.diag(Psi0_cpu)

    prior = {
        "mu0": mu0_cpu,
        "kappa0": float(kappa0),
        "Psi0": Psi0_cpu,
        "nu0": float(nu0),

        "Sigma_within": Sigma_within_cpu,
        "Sigma_between": Sigma_between_cpu,
        "global_mean": mu0_cpu.clone(),

        "class_means": {int(c): class_mu[c].detach().cpu() for c in classes},
        "class_counts": {int(c): int(class_n[c]) for c in classes},
        "class_scatter": {int(c): class_S[c].detach().cpu() for c in classes},

        "cov_mode": cov_mode,
        "return_full_matrix": bool(return_full_matrix),
        "standardize": bool(standardize),
        "mean_all": mean_all,
        "std_all": std_all,
        "shrink_lambda": float(shrink_lambda),
    }

    # 7) Build known clusters (FD full mode)
    alloc_before = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    res_before = torch.cuda.memory_reserved(device) if torch.cuda.is_available() else 0

    prior_dev = SharedNIWPrior.from_prior_dict(prior, device=device, cov_mode=cov_mode)

    # precompute Psi0 FD decomp once if full
    if cov_mode == "full":
        prior_dev.ensure_psi0_fd_decomp(r=int(getattr(args, "fd_rank", 16)), eps=1e-8)

    known_clusters: List[Cluster] = []
    for cid in classes:
        cl = Cluster(
            d=d,
            prior=prior_dev,
            cov_mode=cov_mode,
            device=str(device),
            fd_rank=int(getattr(args, "fd_rank", 16)),
            fd_ell=int(getattr(args, "fd_ell", 32)),
            fd_refresh_every=int(getattr(args, "fd_refresh_every", 1)),
            fd_min_n=int(getattr(args, "fd_min_n", 8)),
            eps=float(getattr(args, "eps", 1e-6)),
        )
        # initialize n and m
        cl.n = torch.tensor(float(class_n[cid]), device=device, dtype=torch.float32)
        cl.m = class_mu[cid].to(device, dtype=torch.float32)

        if cov_mode == "diag":
            # diag scatter (vector) from class_S (already diag in this mode)
            cl.S_diag = class_S[cid].to(device, dtype=torch.float32)
        else:
            # seed FD with approximate scatter energy using eigen-rows from class scatter (one-off per known class)
            # This avoids needing to replay data again just to fill FD.
            # We do a small top-ell eig approximation of class scatter by full eigh only for known classes (C small).
            # Set fd_init_seed=0 to skip seeding and initialize an empty FD sketch.
            fd_init_seed = int(getattr(args, "fd_init_seed", 1))
            if fd_init_seed == 1:
                S_full = class_S[cid].to(device, dtype=torch.float32)
                # take top ell directions for seeding B so that B^T B roughly matches scatter
                ell = int(getattr(args, "fd_ell", 32))
                # full eigh on 768x768 per known class (C small); uses fp32 for stability
                evals, evecs = torch.linalg.eigh(0.5 * (S_full + S_full.T))
                evals = evals.flip(0)
                evecs = evecs[:, torch.arange(d - 1, -1, -1, device=device)]
                ell_eff = min(ell, d)
                # Build B rows: sqrt(eigs) * v^T
                B_seed = torch.zeros(ell, d, device=device, dtype=torch.float32)
                for t in range(ell_eff):
                    if float(evals[t].item()) <= 0:
                        break
                    B_seed[t] = torch.sqrt(evals[t].clamp_min(0.0)) * evecs[:, t]
                cl.fd.B.copy_(B_seed)
                cl.fd.fill = ell
                cl.fd.energy = torch.trace(S_full).to(torch.float32)
                cl.fd.count = int(max(int(cl.n.item()) - 1, 0))  # rough; only used for diagnostics
            else:
                # no seeding; FD empty
                cl.fd.B.zero_()
                cl.fd.fill = 0
                cl.fd.energy = torch.tensor(0.0, device=device, dtype=torch.float32)
                cl.fd.count = 0

        cl.refresh_cache(force=True)
        known_clusters.append(cl)

    alloc_after = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    res_after = torch.cuda.memory_reserved(device) if torch.cuda.is_available() else 0

    if getattr(args, "logger", None) is not None:
        args.logger.info(
            f"[BDP_init] known_clusters incremental allocated = {_bytes_to_mib(alloc_after-alloc_before):.2f} MiB, "
            f"reserved = {_bytes_to_mib(res_after-res_before):.2f} MiB"
        )
    else:
        print(
            f"[BDP_init] known_clusters incremental allocated = {_bytes_to_mib(alloc_after-alloc_before):.2f} MiB, "
            f"reserved = {_bytes_to_mib(res_after-res_before):.2f} MiB"
        )

    mem.snapshot()
    mem.report(prefix="[BDP_init] ", logger=getattr(args, "logger", None))
    return prior, known_clusters


# =========================================================
#                         TEST
# =========================================================

@torch.no_grad()
def test(model, test_loader_unlabelled, prior, clusters: List[Cluster], args):
    model.eval()
    device = torch.device("cuda:0")
    model = model.to(device)
    y_set = set(range(len(args.train_classes)))
    
    profile_fwd = int(getattr(args, "profile_forward_time", 1)) == 1
    fwd_include_h2d = int(getattr(args, "profile_forward_include_h2d", 0)) == 1
    fwd_warmup = int(getattr(args, "profile_forward_warmup_batches", 5))

    fwd_sum_ms = 0.0
    fwd_num_images = 0
    fwd_num_batches = 0
    fwd_max_batch_ms = -1.0
    fwd_max_batch_idx = -1

    cov_mode = str(getattr(args, "cov_mode", "full")).lower()
    eps = float(getattr(args, "eps", 1e-6))
    alpha = float(getattr(args, "alpha", 0.1))
    beta = float(getattr(args, "beta", 1.0))
    temperature = float(getattr(args, "temperature", 1.0))
    temperature_inv = 1.0 / max(temperature, 1e-12)

    chunk_size = int(getattr(args, "score_chunk", 32))
    mem_every = int(getattr(args, "mem_every", 10))
    profile_time = int(getattr(args, "profile_max_sample_time", 1)) == 1
    include_forward = int(getattr(args, "profile_include_forward", 1)) == 1
    use_cuda_timing = profile_time and torch.cuda.is_available()

    max_sample_ms = -1.0
    max_meta = {
        "global_idx": -1,
        "batch_idx": -1,
        "i_in_batch": -1,
        "num_clusters": -1,
        "pred_id": -1,
        "true_label": -1,
        "include_forward": int(include_forward),
    }
    global_sample_idx = 0

    mem = MemoryTracker(device=str(device), track_cpu=True)
    mem.reset_cuda_peak()
    mem.snapshot()

    alloc_before = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    res_before = torch.cuda.memory_reserved(device) if torch.cuda.is_available() else 0

    known_classes = list(range(len(args.train_classes)))

    # Ensure we have a shared device prior for spawning new clusters
    if len(clusters) > 0 and hasattr(clusters[0], "prior"):
        prior_dev = clusters[0].prior
    else:
        prior_dev = SharedNIWPrior.from_prior_dict(prior, device=device, cov_mode=cov_mode)

    # For FD full mode: ensure Psi0 decomp exists
    if cov_mode == "full":
        prior_dev.ensure_psi0_fd_decomp(r=int(getattr(args, "fd_rank", 16)), eps=1e-8)

    # Precompute "new cluster" predictive distribution from prior (fixed during test)
    mu0 = prior_dev.mu0
    kappa0 = prior_dev.kappa0
    nu0 = prior_dev.nu0
    d = mu0.numel()
    nu_new = nu0 - d + 1.0
    scalar_new = (kappa0 + 1.0) / (kappa0 * nu_new)

    if cov_mode == "full":
        # Sigma_new ≈ scalar_new * (psi0_sigma2 I + U diag(lam) U^T)
        sigma2_new = (scalar_new * prior_dev.psi0_sigma2).clamp_min(eps)
        lam_new = (scalar_new * prior_dev.psi0_lam).clamp_min(0.0)
        U_new = prior_dev.psi0_U
        denom_new = (sigma2_new + lam_new).clamp_min(eps)
        logdet_new = (d - int(getattr(args, "fd_rank", 16))) * torch.log(sigma2_new) + torch.log(denom_new).sum()
    else:
        Psi0_diag = prior_dev.Psi0.to(torch.float32)
        Sigma_new_diag = scalar_new * Psi0_diag
        sigma2_new = None
        lam_new = None
        U_new = None
        logdet_new = None

    log_alpha = torch.log(torch.tensor(alpha, device=device, dtype=torch.float32))

    preds = []
    targets = []
    masks = []
    domains = []

    for batch_idx, batch in enumerate(tqdm(test_loader_unlabelled)):
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            images, labels, domain_labels, _ = batch
        else:
            images, labels, _ = batch
            domain_labels = None

        do_fwd_profile = profile_fwd and torch.cuda.is_available()

        if do_fwd_profile:
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)

        if do_fwd_profile and fwd_include_h2d:
            fwd_start.record()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if do_fwd_profile and (not fwd_include_h2d):
            fwd_start.record()

        feats, _ = model(images)

        if do_fwd_profile:
            fwd_end.record()

        feats = feats.detach()

        B = feats.shape[0]
        batch_ev_pairs = []  
        for i in range(B):
            if batch_idx == 0 and i == 0 and len(clusters) > 0:
                if getattr(args, "logger", None) is not None:
                    args.logger.info(f"[DBG] cov_mode={cov_mode}, fd_rank={getattr(args,'fd_rank',None)}, fd_ell={getattr(args,'fd_ell',None)}")
                    args.logger.info(f"[DBG] example cluster U_pred shape={tuple(clusters[0].U_pred.shape)}, fd.B shape={tuple(clusters[0].fd.B.shape) if clusters[0].fd is not None else None}")
                else:
                    print(f"[DBG] cov_mode={cov_mode}, fd_rank={getattr(args,'fd_rank',None)}, fd_ell={getattr(args,'fd_ell',None)}")
                    print(f"[DBG] example cluster U_pred shape={tuple(clusters[0].U_pred.shape)}, fd.B shape={tuple(clusters[0].fd.B.shape) if clusters[0].fd is not None else None}")

            if prior.get("standardize", False):
                y = (feats[i] - prior["mean_all"]) / prior["std_all"].clamp_min(1e-12)
            else:
                y = feats[i]
            y = y.to(device, dtype=torch.float32)

            y_label = int(labels[i].item())
            y_set.add(y_label)

            if use_cuda_timing:
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
            best_old_score, best_old_idx = best_old_cluster_logscore(
                y=y,
                clusters=clusters,
                beta=beta,
                temperature_inv=temperature_inv,
                cov_mode=cov_mode,
                eps=eps,
                chunk_size=chunk_size,
            )

            # new score
            if cov_mode == "full":
                log_p_new = t_logpdf_lowrank_iso(
                    y=y, mu=mu0, nu=nu_new,
                    sigma2=sigma2_new, U=U_new, lam=lam_new,
                    logdet=logdet_new, eps=eps
                )
            else:
                log_p_new = t_logpdf_diag(y, mu0, Sigma_new_diag, nu_new, eps=eps)
            log_s_new = log_alpha + log_p_new

            create_new = (best_old_idx < 0) or (log_s_new > best_old_score)
            


            if create_new:
                pred_id = len(clusters)
                if use_cuda_timing:
                    e_ev.record()
                    batch_ev_pairs.append((
                        s_ev, e_ev,
                        batch_idx, i, global_sample_idx,
                        int(pred_id), int(y_label),
                        int(len(clusters))
                    ))
                global_sample_idx += 1
                cl = spawn_new_cluster(y, prior_dev, args, device=str(device))
                clusters.append(cl)
            else:
                pred_id = best_old_idx
                if use_cuda_timing:
                    e_ev.record()
                    batch_ev_pairs.append((
                        s_ev, e_ev,
                        batch_idx, i, global_sample_idx,
                        int(pred_id), int(y_label),
                        int(len(clusters))
                    ))
                global_sample_idx += 1
                cl = clusters[best_old_idx]
                cl.update_with_point(y)
                

            preds.append(pred_id)
            targets.append(y_label)
            masks.append(y_label in known_classes)
            if domain_labels is not None:
                domains.append(int(domain_labels[i].item()))
                
        msg = f"[Stream] batch={batch_idx} | num_clusters={len(clusters)} | num_true_classes_seen={len(y_set)}"
        if getattr(args, "logger", None) is not None:
            args.logger.info(msg)
        else:
            print(msg)

        if (batch_idx % mem_every) == 0:
            mem.snapshot()
        if use_cuda_timing:
            # synchronize once per batch to make all events valid
            if len(batch_ev_pairs) > 0:
                batch_ev_pairs[-1][1].synchronize()
            else:
                fwd_end.synchronize()

            fwd_ms = fwd_start.elapsed_time(fwd_end) if include_forward else 0.0
            fwd_per_sample_ms = fwd_ms / max(B, 1)
            if do_fwd_profile:
                this_fwd_ms = float(fwd_start.elapsed_time(fwd_end))

                if batch_idx >= fwd_warmup:
                    fwd_sum_ms += this_fwd_ms
                    fwd_num_images += int(B)
                    fwd_num_batches += 1
                    if this_fwd_ms > fwd_max_batch_ms:
                        fwd_max_batch_ms = this_fwd_ms
                        fwd_max_batch_idx = int(batch_idx)

            for (s_ev, e_ev, bidx, ii, gidx, pred_id_i, ylab_i, nclu_i) in batch_ev_pairs:
                sample_ms = s_ev.elapsed_time(e_ev) + fwd_per_sample_ms
                if sample_ms > max_sample_ms:
                    max_sample_ms = float(sample_ms)
                    max_meta.update({
                        "global_idx": int(gidx),
                        "batch_idx": int(bidx),
                        "i_in_batch": int(ii),
                        "num_clusters": int(nclu_i),
                        "pred_id": int(pred_id_i),
                        "true_label": int(ylab_i),
                    })

    final_num_clusters = len(clusters)
    final_num_classes = args.mlp_out_dim

    if getattr(args, "logger", None) is not None:
        args.logger.info(f"[test] FINAL num_clusters={final_num_clusters}, FINAL num_classes={final_num_classes}")
    else:
        print(f"[test] FINAL num_clusters={final_num_clusters}, FINAL num_classes={final_num_classes}")

    preds = np.asarray(preds, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    mask = np.asarray(masks, dtype=bool)

    epoch = getattr(args, "epoch", 0)
    save_name = getattr(args, "save_name", "bdpn_test")

    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=mask,
        T=epoch, eval_funcs=getattr(args, "eval_funcs", None),
        save_name=save_name, args=args
    )

    # memory report
    del images, labels, feats
    if 'domain_labels' in locals():
        del domain_labels
    torch.cuda.synchronize(device)

    mem.snapshot()
    alloc_after = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    res_after = torch.cuda.memory_reserved(device) if torch.cuda.is_available() else 0

    if getattr(args, "logger", None) is not None:
        args.logger.info(
            f"[test] novel_clusters incremental allocated = {_bytes_to_mib(alloc_after-alloc_before):.2f} MiB, "
            f"reserved = {_bytes_to_mib(res_after-res_before):.2f} MiB"
        )
    else:
        print(
            f"[test] novel_clusters incremental allocated = {_bytes_to_mib(alloc_after-alloc_before):.2f} MiB, "
            f"reserved = {_bytes_to_mib(res_after-res_before):.2f} MiB"
        )
    mem.report(prefix="[test] ", logger=getattr(args, "logger", None))

    # DomainNet per-domain eval (optional)
    if args.dataset_name == "domainnet" and len(domains) == len(preds):
        domains = np.asarray(domains, dtype=np.int64)
        for dom_val, dom_name in ((0, "Src"), (1, "Tgt")):
            dom_mask = domains == dom_val
            if not np.any(dom_mask):
                continue
            log_accs_from_preds(
                y_true=targets[dom_mask],
                y_pred=preds[dom_mask],
                mask=mask[dom_mask],
                T=epoch,
                eval_funcs=getattr(args, "eval_funcs", None),
                save_name=f"{save_name}_{dom_name}",
                args=args
            )

    if profile_time:
        if use_cuda_timing and max_sample_ms >= 0:
            msg = (
                f"[test] MAX per-sample inference time = {max_sample_ms:.3f} ms "
                f"(GPU events, include_forward={max_meta['include_forward']}) | "
                f"global_idx={max_meta['global_idx']} batch={max_meta['batch_idx']} i={max_meta['i_in_batch']} | "
                f"num_clusters={max_meta['num_clusters']} pred_id={max_meta['pred_id']} true_label={max_meta['true_label']}"
            )
        else:
            # fallback: no CUDA or no samples measured
            msg = "[test] MAX per-sample inference time: CUDA not available or profiling disabled/empty."

        if getattr(args, "logger", None) is not None:
            args.logger.info(msg)
        else:
            print(msg)
            
    if profile_fwd:
        if fwd_num_images > 0:
            avg_fwd_ms_per_image = fwd_sum_ms / max(fwd_num_images, 1)
            avg_fwd_ms_per_batch = fwd_sum_ms / max(fwd_num_batches, 1)
            msg = (
                f"[test] Forward time: avg={avg_fwd_ms_per_image:.3f} ms/img, "
                f"avg_batch={avg_fwd_ms_per_batch:.3f} ms/batch, "
                f"max_batch={fwd_max_batch_ms:.3f} ms (batch={fwd_max_batch_idx}), "
                f"include_h2d={int(fwd_include_h2d)}, warmup_batches={fwd_warmup}, "
                f"count_images={fwd_num_images}"
            )
        else:
            msg = "[test] Forward time: no samples counted (maybe all batches are warmup or profiling disabled)."

        if getattr(args, "logger", None) is not None:
            args.logger.info(msg)
        else:
            print(msg)
    return all_acc, old_acc, new_acc


# =========================================================
#                         ARGS
# =========================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # -------- common (original) --------
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', ])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars',
                        help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19, domainnet')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)

    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--num_labeled_classes', default=80, type=int)
    parser.add_argument('--memax_weight', type=float, default=2.0)

    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float)
    parser.add_argument('--teacher_temp', default=0.04, type=float)
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int)

    parser.add_argument('--pretrain', action='store_false', default=False)
    parser.add_argument('--OCD', action='store_true', default=True)

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--split_mode', default='easy', type=str)

    # -------- BDP / NIW args --------
    parser.add_argument('--cov_mode', default='full', type=str, choices=['diag', 'full'])
    parser.add_argument('--return_full_matrix', action='store_true', default=True)
    parser.add_argument('--standardize', action='store_true', default=False)

    parser.add_argument('--kappa0_init', default=None, type=float)
    parser.add_argument('--kappa0_strategy', action='store_const', const=True, default=None)
    parser.add_argument('--shrink_lambda', default=0.0, type=float)
    parser.add_argument('--eps', default=1e-6, type=float)

    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--tau_new', default=1.0, type=float)
    parser.add_argument('--delta_bf', default=0.1, type=float)
    parser.add_argument('--s_min', default=1, type=int)

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--profile_max_sample_time', default=1, type=int,
                        help='1: compute max per-sample inference time in test (GPU events, sync once per batch); 0: disable')
    parser.add_argument('--profile_include_forward', default=1, type=int,
                        help='1: include forward_ms/B into per-sample time; 0: only score+update time')

    # -------- memory tracking flags --------
    parser.add_argument('--mem_every', default=10, type=int, help='snapshot memory every N batches')

    # packed storage controls (kept; FD full mode ignores them for precision)
    parser.add_argument('--symm_packed', default=1, type=int, help='1: store full-cov matrices as packed upper-tri (kept for compatibility)')
    parser.add_argument('--cov_store_dtype', default='fp32', type=str, choices=['fp32', 'fp16', 'bf16'],
                        help='storage dtype for packed cov/scatter (kept for compatibility)')

    # scoring chunk
    parser.add_argument('--score_chunk', default=32, type=int,
                        help='chunk size for scoring old clusters (avoid huge temps)')

    # -------- optional extra (compatibility) --------
    parser.add_argument('--head_dim', default=128, type=int)
    parser.add_argument('--src_env', type=str, default='real')
    parser.add_argument('--aux_env', type=str, default='clipart')

    # -------- FD-specific (ADDED; does not delete any existing args) --------
    parser.add_argument('--fd_rank', default=16, type=int, help='rank r for lowrank+iso (Sigma ≈ sigma2 I + U diag(lam) U^T)')
    parser.add_argument('--fd_ell', default=32, type=int, help='FD sketch rows ell (typical: 2*r or 3*r)')
    parser.add_argument('--fd_refresh_every', default=1, type=int, help='refresh cov cache every N assigned points per cluster')
    parser.add_argument('--fd_min_n', default=8, type=int, help='minimum cluster n to trust FD scatter (else fallback to prior-only scatter)')
    parser.add_argument('--fd_init_seed', default=1, type=int, help='1: seed known clusters FD sketch from class scatter via eigh; 0: no seeding (faster init, slightly less accurate early)')

    parser.add_argument('--profile_forward_time', default=1, type=int,
                    help='1: profile model forward time (images->feats); 0: disable')
    parser.add_argument('--profile_forward_include_h2d', default=0, type=int,
                        help='1: include images.to(device) H2D time in forward timing; 0: forward only')
    parser.add_argument('--profile_forward_warmup_batches', default=5, type=int,
                        help='number of warmup batches excluded from forward stats')
    return parser


# =========================================================
#                         MAIN
# =========================================================

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    device = torch.device("cuda:0")
    make_deterministic(args.seed)

    # class splits
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    # init logging
    init_experiment(args, runner_name=["simgcd"], exp_id=args.exp_name)
    args.logger.info(f"Using evaluation function {args.eval_funcs[0]}")

    # base model
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    if args.fp16:
        args.logger.info("[WARN] --fp16 is enabled. FD sketches and statistics remain in fp32 for numerical stability.")

    pretrain_path = dino_pretrain_path
    backbone = vits.__dict__["vit_base"]()
    state_dict = torch.load(pretrain_path, map_location="cpu")
    backbone.load_state_dict(state_dict)

    for p in backbone.parameters():
        p.requires_grad = False

    # transforms / datasets
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
        args.dataset_name, train_transform, test_transform, args
    )

    # domainnet handling
    if args.dataset_name == "domainnet":
        if test_dataset is not None:
            test_dataset = StripDomainLabelDataset(test_dataset)
        trainA_unlabelled = datasets.get("trainA_unlabelled")
        trainB_unlabelled = datasets.get("trainB_unlabelled")
        if trainA_unlabelled is not None or trainB_unlabelled is not None:
            parts = []
            if trainA_unlabelled is not None:
                trainA_eval = deepcopy(trainA_unlabelled)
                trainA_eval.transform = test_transform
                parts.append(trainA_eval)
            if trainB_unlabelled is not None:
                trainB_eval = deepcopy(trainB_unlabelled)
                trainB_eval.transform = test_transform
                parts.append(trainB_eval)
            if len(parts) == 1:
                unlabelled_train_examples_test = parts[0]
            elif len(parts) > 1:
                unlabelled_train_examples_test = ConcatDataset(parts)
        else:
            if unlabelled_train_examples_test is not None:
                unlabelled_train_examples_test = StripDomainLabelDataset(unlabelled_train_examples_test)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              drop_last=False, num_workers=0, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, batch_size=args.batch_size, shuffle=False,
                                        drop_last=False, num_workers=0, pin_memory=True)

    # projection head + full model
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes)
    model = nn.Sequential(backbone, projector).to(device)
    for p in model.parameters():
        p.requires_grad = False

    if args.warmup_model_dir is not None:
        args.logger.info(f"Loading weights from {args.warmup_model_dir}")
        sd = torch.load(args.warmup_model_dir, map_location="cpu")["model"]
        model.load_state_dict(sd, strict=True)

    model_statistics(model)

    print("[FP] bdp_loader =", loader_fingerprint(train_loader))
    print("[FP] test_loader_unlabelled =", loader_fingerprint(test_loader_unlabelled))

    # heuristic for n0
    args.n0 = min(int(len(train_dataset) / (2 * args.num_labeled_classes)), 50)

    print(args)

    alloc_before = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    res_before = torch.cuda.memory_reserved(device) if torch.cuda.is_available() else 0

    prior, known_clusters = BDP_init(model, train_loader, args)
    all_acc, old_acc, new_acc = test(model, test_loader_unlabelled, prior, known_clusters, args)

    alloc_after = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
    res_after = torch.cuda.memory_reserved(device) if torch.cuda.is_available() else 0

    if getattr(args, "logger", None) is not None:
        args.logger.info(
            f"[All] All_clusters incremental allocated = {_bytes_to_mib(alloc_after-alloc_before):.2f} MiB, "
            f"reserved = {_bytes_to_mib(res_after-res_before):.2f} MiB"
        )
    else:
        print(
            f"[All] All_clusters incremental allocated = {_bytes_to_mib(alloc_after-alloc_before):.2f} MiB, "
            f"reserved = {_bytes_to_mib(res_after-res_before):.2f} MiB"
        )

    print("Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}".format(all_acc, old_acc, new_acc))
