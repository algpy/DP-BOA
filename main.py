import argparse

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from model import vision_transformer as vits

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from data.data_utils import StripDomainLabelDataset

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model.utils import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups, ETF_Classifier, model_statistics, TeacherHead, DINOHead_
from config import *
from collections import defaultdict
import os

import hashlib
from copy import deepcopy

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

def seed_worker(worker_id):
    worker_seed = args.seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def _unwrap_domainnet_dataset(dataset):
    if isinstance(dataset, StripDomainLabelDataset):
        return dataset.dataset
    return dataset

def _collect_domainnet_labels_domains(dataset):
    dataset = _unwrap_domainnet_dataset(dataset)
    if dataset is None:
        return None, None

    if isinstance(dataset, ConcatDataset):
        labels_list = []
        domains_list = []
        for sub in dataset.datasets:
            sub_labels, sub_domains = _collect_domainnet_labels_domains(sub)
            if sub_labels is None:
                continue
            labels_list.append(sub_labels)
            domains_list.append(sub_domains)
        if not labels_list:
            return None, None
        labels = np.concatenate(labels_list)
        if any(d is None for d in domains_list):
            return labels, None
        domains = np.concatenate(domains_list)
        return labels, domains

    if hasattr(dataset, 'class_targets') and hasattr(dataset, 'domain_targets'):
        return np.array(dataset.class_targets), np.array(dataset.domain_targets)

    if hasattr(dataset, 'samples') and len(dataset.samples) > 0:
        sample = dataset.samples[0]
        if isinstance(sample, (list, tuple)):
            if len(sample) >= 4:
                labels = np.array([s[1] for s in dataset.samples])
                domains = np.array([s[3] for s in dataset.samples])
                return labels, domains
            if len(sample) == 3:
                labels = np.array([s[1] for s in dataset.samples])
                domains = np.array([s[2] for s in dataset.samples])
                return labels, domains
            if len(sample) == 2:
                labels = np.array([s[1] for s in dataset.samples])
                if hasattr(dataset, 'domain_label'):
                    domains = np.full(len(labels), int(dataset.domain_label))
                    return labels, domains
                return labels, None

    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
        if hasattr(dataset, 'domain_label'):
            domains = np.full(len(labels), int(dataset.domain_label))
            return labels, domains
        return labels, None

    return None, None

def _domain_from_key(key):
    if key.startswith('trainA') or 'trainA' in key:
        return 0
    if key.startswith('trainB') or 'trainB' in key:
        return 1
    return None

def _print_domainnet_class_counts(datasets, args):
    counts = {}

    def _init_entry(cls_id):
        if cls_id not in counts:
            counts[cls_id] = {
                'src_labelled': 0,
                'src_unlabelled': 0,
                'tgt_labelled': 0,
                'tgt_unlabelled': 0,
            }

    def _accumulate(labels, domains, labelled_flag):
        if labels is None or domains is None:
            return
        labels = np.asarray(labels)
        domains = np.asarray(domains)
        for domain_val in (0, 1):
            mask = domains == domain_val
            if not np.any(mask):
                continue
            cls_vals, cls_counts = np.unique(labels[mask], return_counts=True)
            for cls_id, cnt in zip(cls_vals, cls_counts):
                _init_entry(int(cls_id))
                if domain_val == 0:
                    key = 'src_labelled' if labelled_flag else 'src_unlabelled'
                else:
                    key = 'tgt_labelled' if labelled_flag else 'tgt_unlabelled'
                counts[int(cls_id)][key] += int(cnt)

    for key, dataset in datasets.items():
        if not key.startswith('train'):
            continue
        if 'val' in key or 'test' in key:
            continue

        if 'unlabelled' in key:
            labelled_flag = False
        elif 'labelled' in key:
            labelled_flag = True
        else:
            continue

        labels, domains = _collect_domainnet_labels_domains(dataset)
        if labels is None:
            continue
        if domains is None:
            domain = _domain_from_key(key)
            if domain is None:
                continue
            domains = np.full(len(labels), int(domain))

        _accumulate(labels, domains, labelled_flag)

    if not counts:
        args.logger.info("DomainNet class counts: no training datasets found.")
        return

    args.logger.info("DomainNet class counts (train only):")
    for cls_id in sorted(counts.keys()):
        entry = counts[cls_id]
        total = entry['src_labelled'] + entry['src_unlabelled'] + entry['tgt_labelled'] + entry['tgt_unlabelled']
        if cls_id in args.train_classes:
            status = 'known'
        elif cls_id in args.unlabeled_classes:
            status = 'unknown'
        else:
            status = 'other'
        args.logger.info(
            f"class {cls_id:03d} ({status}) "
            f"src_L={entry['src_labelled']} src_U={entry['src_unlabelled']} "
            f"tgt_L={entry['tgt_labelled']} tgt_U={entry['tgt_unlabelled']} "
            f"total={total}"
        )

# ----------  Student-t log-pdf  ----------
def t_logpdf(y, mu, Sigma, nu, cov_mode="diag", eps=1e-6):
    """
    y, mu: (d,) torch.float
    Sigma: (d,) for diag  or  (d,d) for full"
    nu: scalar > 0
    return: scalar torch.float (log-density at y)
    """
    d = y.numel()
    if cov_mode == "diag":
        s = Sigma.clamp_min(eps)  # (d,)
        diff = y - mu
        mahal = (diff * diff / s).sum()  # sum((y-mu)^2 / s)
        logdet = torch.log(s).sum()
    else:
        L = torch.linalg.cholesky(Sigma + eps * torch.eye(d, device=Sigma.device))
        diff = y - mu
        z = torch.cholesky_solve(diff.unsqueeze(1), L)  # (d,1)
        mahal = (diff.unsqueeze(0) @ z).squeeze()        
        logdet = 2.0 * torch.log(torch.diag(L)).sum()

    c = torch.lgamma((nu + d) * 0.5) - torch.lgamma(nu * 0.5) \
        - 0.5 * (d * torch.log(torch.tensor(nu * np.pi, device=y.device)) + logdet)
    return c - 0.5 * (nu + d) * torch.log1p(mahal / nu)


# ---------- t from NIW ----------
def predictive_from_NIW(mu, kappa, Psi, nu, cov_mode="diag"):
    # p(y|D) = t_{nu-d+1}( mu,  ((kappa+1)/(kappa*(nu-d+1))) * Psi )
    d = mu.numel()
    nu_pred = nu - d + 1.0
    scale = ((kappa + 1.0) / (kappa * nu_pred)) * Psi
    return nu_pred, scale


# ---------- Welford for (n, mean, scatter) ----------
def welford_update(n, m, S, y, cov_mode="diag"):
    n_new = n + 1.0
    delta = y - m
    m_new = m + delta / n_new
    delta2 = y - m_new
    if cov_mode == "diag":
        S_new = S + delta * delta2
    else:
        S_new = S + torch.outer(delta, delta2)
    return n_new, m_new, S_new


# ---------- posterior distribution of  NIW ----------
def niw_posterior_from_stats(mu0, kappa0, Psi0, nu0, n, m, S, cov_mode="diag"):
    kappa = kappa0 + n
    mu = (kappa0 * mu0 + n * m) / kappa
    nu = nu0 + n
    diff = (m - mu0)
    if cov_mode == "diag":
        Psi = Psi0 + S + (kappa0 * n / kappa) * (diff * diff)
    else:
        Psi = Psi0 + S + (kappa0 * n / kappa) * torch.outer(diff, diff)
    return mu, kappa, Psi, nu


# ---------- Cluster ----------
class Cluster:
    def __init__(self, d, prior, cov_mode="diag", device="cuda"):
        self.cov_mode = cov_mode
        self.device = device

        self.mu0 = prior["mu0"].to(device)
        self.kappa0 = torch.tensor(prior["kappa0"], device=device, dtype=torch.float32)
        self.Psi0 = prior["Psi0"].to(device)
        self.nu0 = torch.tensor(prior["nu0"], device=device, dtype=torch.float32)

        self.n = torch.tensor(0.0, device=device)
        self.m = torch.zeros(d, device=device)
        if cov_mode == "diag":
            self.S = torch.zeros(d, device=device)
        else:
            self.S = torch.zeros(d, d, device=device)

        self.mu = self.mu0.clone()
        self.kappa = self.kappa0.clone()
        self.Psi = self.Psi0.clone()
        self.nu = self.nu0.clone()

    def predictive_logpdf(self, y, eps=1e-6):
        nu_pred, scale = predictive_from_NIW(self.mu, self.kappa, self.Psi, self.nu, self.cov_mode)
        return t_logpdf(y, self.mu, scale, nu_pred, cov_mode=self.cov_mode, eps=eps)

    def update_with_point(self, y):
        self.n, self.m, self.S = welford_update(self.n, self.m, self.S, y, cov_mode=self.cov_mode)
        self.mu, self.kappa, self.Psi, self.nu = niw_posterior_from_stats(
            self.mu0, self.kappa0, self.Psi0, self.nu0,
            self.n, self.m, self.S, cov_mode=self.cov_mode
        )


# ---------- init for new cluster ----------
def spawn_new_cluster(y, prior, cov_mode="diag", device="cuda"):
    d = y.numel()
    cl = Cluster(d, prior, cov_mode=cov_mode, device=device)
    cl.update_with_point(y)
    return cl


@torch.no_grad()
def BDP_init(model, data_loader, args):
    """
    Empirical-Bayes initialization for NIW prior using labeled training data.

    Returns:
        prior = {
            "mu0":  (d,), torch.float (on CPU),
            "kappa0": float,
            "Psi0": (d,d) or (d,) depending on args.cov_mode/return_full_matrix,
            "nu0":  float,
            # (optional diagnostics)
            "Sigma_within": (d,d) or (d,),
            "Sigma_between": (d,d) or (d,),
            "global_mean": (d,),
            "class_means": dict[int -> (d,)],
            "class_counts": dict[int -> int],
        }
    """

    device = torch.device('cuda:0')
    model = model.to(device).eval()

    # --------------------
    # args with defaults
    # --------------------
    cov_mode = getattr(args, "cov_mode", "diag")  # "diag" | "full"
    return_full_matrix = getattr(args, "return_full_matrix", True)  # True -> Psi0 as (d,d)
    # EB strength for covariance: n0 = nu0 - d - 1
    n0 = float(getattr(args, "n0", 10.0))          # typical 5~20
    kappa0_init = getattr(args, "kappa0_init", None)  # if set, we won't fit kappa0
    kappa0_strategy = getattr(args, "kappa0_strategy", None)  # None: original calibration
    kappa0_clip_min = float(getattr(args, "kappa0_clip_min", 1.0))
    kappa0_clip_max = float(getattr(args, "kappa0_clip_max", 10.0))
    # optional shrinkage for Sigma_within toward diagonal (0: none, 1: full diagonal)
    shrink_lambda = float(getattr(args, "shrink_lambda", 0.0))  # 0~1
    # optional standardization in-flight (mean-std over the stream) — default off
    standardize = bool(getattr(args, "standardize", False))

    # --------------------------------
    # 1) First pass stats (optional global standardization)
    # --------------------------------
    # If standardize=True, we compute global mean/std first (single pass).
    d = args.feat_dim
    sum_y = torch.zeros(d, device=device)
    sum_y2 = torch.zeros(d, device=device)
    N_total = 0

    if standardize:
        for batch_idx, batch in enumerate(data_loader):
            images, class_labels, uq_idxs = batch
            images = images.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)

            features, _ = model(images)  # (B,d)
            features = features.detach()
            
            sum_y += features.sum(dim=0)
            sum_y2 += (features ** 2).sum(dim=0)
            N_total += features.shape[0]

        mean_all = sum_y / max(N_total, 1)
        var_all = (sum_y2 / max(N_total, 1)) - mean_all**2
        std_all = torch.sqrt(var_all.clamp_min(1e-12))
    else:
        mean_all = None
        std_all = None

    # --------------------------------
    # 2) Class-wise Welford (single pass, merge-by-batch)
    #    We keep: n_c, mean_c, scatter_c (either full or diag)
    # --------------------------------
    class_n  = defaultdict(int)
    class_mu = dict()   # int -> (d,)
    class_S  = dict()   # int -> (d,d) or (d,)

    def _init_class_buffers(d, cov_mode):
        if cov_mode == "full":
            return torch.zeros(d, device=device), torch.zeros(d, d, device=device)
        else:
            return torch.zeros(d, device=device), torch.zeros(d, device=device)

    def _batch_scatter(X, cov_mode):
        # X: (b,d), returns scatter S_b = sum (x - xbar)(x - xbar)^T (full) or diag version
        xb = X.mean(dim=0)
        diff = X - xb
        if cov_mode == "full":
            Sb = diff.T @ diff
        else:
            Sb = (diff**2).sum(dim=0)
        return xb, Sb, X.shape[0]

    def _merge_welford(n, m, C, nb, mb, Sb):
        # Merge two groups (n,m,C) and (nb,mb,Sb)
        # C, Sb are scatter; m, mb are means (d,)
        # Supports full/diag by broadcasting
        if n == 0:
            return nb, mb, Sb
        n_new = n + nb
        delta = (mb - m)
        if C.ndim == 2:
            # full
            C_new = C + Sb + torch.ger(delta, delta) * (n * nb / n_new)
        else:
            # diag
            C_new = C + Sb + (delta * delta) * (n * nb / n_new)
        m_new = m + delta * (nb / n_new)
        return n_new, m_new, C_new

    # reset loader (some frameworks require re-instantiation; here we assume reusable)
    # Second pass (or first pass if standardize=False): compute class stats
    for batch_idx, batch in enumerate(data_loader):
        images, class_labels, uq_idxs = batch
        images = images.to(device, non_blocking=True)
        class_labels = class_labels.to(device, non_blocking=True)

        features, _ = model(images)  # (B,d)
        X = features.detach()

        if standardize:
            X = (X - mean_all) / std_all.clamp_min(1e-12)

        # group by class within the batch
        for c in class_labels.unique():
            c = c.item()
            idx = (class_labels == c)
            Xc = X[idx]  # (b_c, d)
            if Xc.numel() == 0:
                continue

            if c not in class_mu:
                d = Xc.shape[1]
                m_c, S_c = _init_class_buffers(d, cov_mode)
                class_mu[c] = m_c.clone()
                class_S[c]  = S_c.clone()
                class_n[c]  = 0

            # batch stats for this class
            mb, Sb, nb = _batch_scatter(Xc, cov_mode)

            # merge into class buffers
            n_old = class_n[c]
            m_old = class_mu[c]
            S_old = class_S[c]
            n_new, m_new, S_new = _merge_welford(n_old, m_old, S_old, nb, mb, Sb)

            class_n[c]  = n_new
            class_mu[c] = m_new
            class_S[c]  = S_new

    # --------------------------------
    # 3) Global & pooled stats
    # --------------------------------
    classes = sorted(class_n.keys())
    C = len(classes)
    assert C > 0, "No labeled classes found in data_loader."
    
    N = sum(class_n[c] for c in classes)

    # global mean (weighted by class size)
    mu_global = torch.zeros(d, device=device)
    for c in classes:
        mu_global += class_mu[c] * class_n[c]
    mu_global = mu_global / max(N, 1)

    # pooled within-class covariance: Sigma_within = (sum S_c) / (N - C)
    if cov_mode == "full":
        Sw_sum = torch.zeros(d, d, device=device)
    else:
        Sw_sum = torch.zeros(d, device=device)

    for c in classes:
        Sw_sum += class_S[c]

    denom_within = max(N - C, 1)
    Sigma_within = Sw_sum / denom_within

    # Original calibration weights class means by n_c; corrected calibration uses equal weights.
    if cov_mode == "full":
        Sb_sum = torch.zeros(d, d, device=device)
    else:
        Sb_sum = torch.zeros(d, device=device)

    for c in classes:
        delta = (class_mu[c] - mu_global)
        weight = class_n[c] if kappa0_strategy is None else 1.0
        if cov_mode == "full":
            Sb_sum += weight * torch.ger(delta, delta)
        else:
            Sb_sum += weight * delta * delta

    denom_between = max(C - 1, 1)
    Sigma_between = Sb_sum / denom_between

    # optional shrinkage toward diagonal for Sigma_within
    if shrink_lambda > 0.0:
        if cov_mode == "full":
            diagW = torch.diag(torch.diag(Sigma_within))
            Sigma_within = (1.0 - shrink_lambda) * Sigma_within + shrink_lambda * diagW
        else:
            # already diagonal form
            pass

    # --------------------------------
    # 4) Set NIW hyperparameters: mu0, kappa0, Psi0, nu0
    # --------------------------------
    # mu0
    mu0 = mu_global.clone()

    nu0 = float(d + 1 + max(n0, 1.0))

    # Psi0 chosen so that E[Σ] = Psi0 / (nu0 - d - 1) = Sigma_within  => Psi0 = n0 * Sigma_within
    if cov_mode == "full":
        Psi0 = Sigma_within * (nu0 - d - 1.0)
    else:
        # diagonal as vector
        Psi0 = Sigma_within * (nu0 - d - 1.0)

    # kappa0 via class-mean variance matching (fallback to clip range if unstable)
    if kappa0_init is not None:
        kappa0 = float(kappa0_init)
    else:
        # Original: 1/mean(n_c); corrected: mean(1/n_c).
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
        kappa0_est = tr_within / denom
        # clip to a sensible range
        #kappa0 = float(min(max(kappa0_est, kappa0_clip_min), kappa0_clip_max))
        kappa0 = float(kappa0_est)

    # --------------------------------
    # 5) Move tensors to CPU for serialization / downstream use
    # --------------------------------
    mu0_cpu = mu0.detach().cpu()
    if cov_mode == "full":
        Sigma_within_cpu = Sigma_within.detach().cpu()
        Sigma_between_cpu = Sigma_between.detach().cpu()
        Psi0_cpu = Psi0.detach().cpu()
        if not return_full_matrix:
            # return diagonal only
            Psi0_cpu = torch.diag(Psi0_cpu)
    else:
        # diag vectors
        Sigma_within_cpu = Sigma_within.detach().cpu()
        Sigma_between_cpu = Sigma_between.detach().cpu()
        if return_full_matrix:
            Psi0_cpu = torch.diag(Psi0.detach().cpu())
        else:
            Psi0_cpu = Psi0.detach().cpu()

    class_means_cpu = {int(c): class_mu[c].detach().cpu() for c in classes}
    class_counts = {int(c): int(class_n[c]) for c in classes}
    class_scatters = {int(c): class_S[c].detach().cpu() for c in classes}

    prior = {
        "mu0": mu0_cpu,                 # (d,)
        "kappa0": float(kappa0),        # scalar
        "Psi0": Psi0_cpu,               # (d,d) or (d,)
        "nu0": float(nu0),              # scalar
        # diagnostics
        "Sigma_within": Sigma_within_cpu,
        "Sigma_between": Sigma_between_cpu,
        "global_mean": mu0_cpu.clone(),
        "class_means": class_means_cpu,
        "class_counts": class_counts,
        "class_scatter": class_scatters,
        "cov_mode": cov_mode,
        "return_full_matrix": bool(return_full_matrix),
        "standardize": bool(standardize),
        "std_all": std_all,
        "mean_all": mean_all,
        "shrink_lambda": float(shrink_lambda),
    }
    
    known_clusters = []
    prior_dev = {
        "mu0": prior["mu0"].to(device),
        "kappa0": prior["kappa0"],
        "Psi0": prior["Psi0"].to(device),
        "nu0": prior["nu0"],
    }
    for cid in classes:
        m_c = class_mu[cid].to(device).float()
        n_c = float(class_n[cid])
        S_c = class_S[cid].to(device).float()

        cl = Cluster(d, prior_dev, cov_mode=cov_mode, device=device)
        cl.n = torch.tensor(n_c, device=device)
        cl.m = m_c.clone()
        cl.S = S_c.clone()
        cl.mu, cl.kappa, cl.Psi, cl.nu = niw_posterior_from_stats(
            cl.mu0, cl.kappa0, cl.Psi0, cl.nu0,
            cl.n, cl.m, cl.S, cov_mode=cov_mode
        )
        known_clusters.append(cl)
    return prior, known_clusters


# =========================
#         TEST
# =========================
@torch.no_grad()
def test(model, test_loader_unlabelled, prior, clusters, args):
    """
    online OCD + BDPN inference
    return all_acc, old_acc, new_acc
    """
    model.eval()
    device = torch.device("cuda")
    model = model.to(device)

    # hyperparameters
    alpha = float(getattr(args, "alpha", 0.3))         
    beta = float(getattr(args, "beta", 1.0))            
    temperature = float(getattr(args, "temperature", 1.0))  
    tau_new = float(getattr(args, "tau_new", 0.7))     
    delta_bf = float(getattr(args, "delta_bf", 0.0))   
    s_min = int(getattr(args, "s_min", 1))             
    cov_mode = getattr(args, "cov_mode", "diag")        
    eps = float(getattr(args, "eps", 1e-6))

    # known class set
    known_classes = list(range(len(args.train_classes)))

    # ---- prediciton of new cluster ----
    # p(y|new) = t_{nu0-d+1}(mu0, ((kappa0+1)/(kappa0*(nu0-d+1)))*Psi0 )
    mu0 = prior["mu0"].to(device)
    kappa0 = torch.tensor(prior["kappa0"], device=device, dtype=torch.float32)
    Psi0 = prior["Psi0"].to(device)
    nu0 = torch.tensor(prior["nu0"], device=device, dtype=torch.float32)

    d = mu0.numel()
    nu_new = nu0 - d + 1.0
    Sigma_new = ((kappa0 + 1.0) / (kappa0 * nu_new)) * Psi0  # t-scale for "new"

    # ---- save ----
    preds = []
    targets = []
    masks = []   # True=old, False=new
    domains = []
    y_set = list(range(len(args.train_classes)))

    # ---- OCD ----
    for batch_idx, batch in enumerate(tqdm(test_loader_unlabelled)):
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            images, labels, domain_labels, _ = batch
        else:
            images, labels, _ = batch
            domain_labels = None
        args.logger.info('num of clusters = {}'.format(len(clusters)))
        args.logger.info('num of classes = {}'.format(len(y_set)))
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        feats, _ = model(images)   # (B,d)
        feats = feats.detach()

        B = feats.shape[0]
        for i in range(B):
            if prior['standardize']:
                y = (feats[i] - prior['mean_all']) / prior['std_all'].clamp_min(1e-12)
            else:
                y = feats[i]       
            if labels[i] not in y_set:
                y_set.append(labels[i])       
            y_label = int(labels[i])  

            # ---- log-score of old cluster----
            log_scores = []
            for cl in clusters:
                log_p = cl.predictive_logpdf(y, eps=eps)              # log p(y|D_k)
                # log s_k = β*log(n_k) + (1/T)*log p(y|D_k)
                log_s = beta * torch.log(cl.n.clamp_min(1.0)) + (1.0 / temperature) * log_p
                log_scores.append(log_s)

            #  log-score of new cluster
            log_s_new = torch.log(torch.tensor(alpha, device=device)) + t_logpdf(y, mu0, Sigma_new, nu_new, cov_mode=cov_mode, eps=eps)

            # softmax over [old..., new]
            if len(log_scores) > 0:
                all_logs = torch.stack(log_scores + [log_s_new])
                q = F.softmax(all_logs, dim=0)
                q_new = q[-1].item()
                best_old_idx = int(torch.argmax(torch.stack(log_scores)).item())
                best_old_log = log_scores[best_old_idx]
            else:
                # new cluster
                q_new = 1.0
                best_old_idx = -1
                best_old_log = torch.tensor(-1e9, device=device)

            # Bayes 
            if len(log_scores) > 0:
                if log_s_new > torch.max(torch.stack(log_scores)).item():
                    create_new = True
                else:
                    create_new = False
            else:
                create_new = True

            if create_new:
                cl = spawn_new_cluster(y, prior, cov_mode=cov_mode, device=device)
                clusters.append(cl)
                pred_id = len(clusters) - 1 
            else:
                cl = clusters[best_old_idx]
                cl.update_with_point(y)
                pred_id = best_old_idx

            preds.append(pred_id)
            targets.append(y_label)
            masks.append(y_label in known_classes)
            if domain_labels is not None:
                domains.append(int(domain_labels[i]))


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
    if args.dataset_name == 'domainnet' and len(domains) == len(preds):
        domains = np.asarray(domains, dtype=np.int64)
        for dom_val, dom_name in ((0, 'Src'), (1, 'Tgt')):
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
    return all_acc, old_acc, new_acc



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', ])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
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
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')
    
    parser.add_argument('--pretrain', action='store_false', default=False)
    parser.add_argument('--OCD', action='store_true', default=True)

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--split_mode', default='easy', type=str)
    
    parser.add_argument('--cov_mode', default='full', type=str)
    parser.add_argument('--return_full_matrix', action='store_true', default=True)
    parser.add_argument('--standardize', action='store_false', default=False)
    parser.add_argument('--kappa0_init', default=None, type=float)
    parser.add_argument('--kappa0_strategy', action='store_const', const=True, default=None)
    
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--tau_new', default=1.0, type=float)
    parser.add_argument('--delta_bf', default=0.1, type=float)
    
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--src_env', type=str, default='real')
    parser.add_argument('--aux_env', type=str, default='clipart')
    

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)
    
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    # if args.dataset_name == "cub":
    #     args.num_labeled_classes = len(args.train_classes)
    #     args.num_unlabeled_classes = 231 - len(args.unlabeled_classes)
    # elif args.dataset_name == "aircraft":
    #     args.num_labeled_classes = len(args.train_classes)
    #     args.num_unlabeled_classes = 108 - len(args.unlabeled_classes)
    # elif args.dataset_name == "scars":
    #     args.num_labeled_classes = len(args.train_classes)
    #     args.num_unlabeled_classes = 230 - len(args.unlabeled_classes)
    # else:
    #     args.num_labeled_classes = len(args.train_classes)
    #     args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'], exp_id=args.exp_name)
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    #torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    
    make_deterministic(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    pretrain_path = dino_pretrain_path

    backbone = vits.__dict__['vit_base']()

    state_dict = torch.load(pretrain_path, map_location='cpu')
    backbone.load_state_dict(state_dict)

    # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    # if args.warmup_model_dir is not None:
    #     args.logger.info(f'Loading weights from {args.warmup_model_dir}')
    #     backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
    if args.dataset_name == 'domainnet':
        if test_dataset is not None:
            test_dataset = StripDomainLabelDataset(test_dataset)
        trainA_unlabelled = datasets.get('trainA_unlabelled')
        trainB_unlabelled = datasets.get('trainB_unlabelled')
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
        _print_domainnet_class_counts(datasets, args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=True
    )

    test_loader_unlabelled = DataLoader(
        unlabelled_train_examples_test, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    #teacher_head = TeacherHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, middle_dim=int(args.feat_dim * args.head_dim), nlayer=args.nlayer).to(device)

    # projector = ETF_Classifier(args.feat_dim, args.mlp_out_dim)
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes)
    #projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim)
    model = nn.Sequential(backbone, projector).to(device)
    for m in model.parameters():
        m.requires_grad = False

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        state_dict = torch.load(args.warmup_model_dir, map_location='cpu')["model"]
        model.load_state_dict(state_dict, strict=True)

    model_statistics(model)
    
    print("[FP] bdp_loader =", loader_fingerprint(train_loader))
    print("[FP] test_loader_unlabelled =", loader_fingerprint(test_loader_unlabelled))
    
    args.n0 = min(int(len(train_dataset) / (2 * args.num_labeled_classes)), 50)
    
    print(args)
    prior, known_clusters = BDP_init(model, train_loader, args)
    print('prior = ', prior)
    all_acc, old_acc, new_acc = test(model, test_loader_unlabelled, prior, known_clusters, args)
    print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
