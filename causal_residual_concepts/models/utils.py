"""Some functions that are needed here and there."""

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from json import load
import numpy as np
import argparse
import torch.nn as nn
import torch
from sklearn.metrics import normalized_mutual_info_score, matthews_corrcoef, mutual_info_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import math
from typing import Optional, Sequence, Tuple, Union
from .weight_averaging import EMA
import torch.nn.functional as F
from torch.autograd import Function


DINOV2_EMBED_DIM = 5120

def detect_confounding(ate_bootstrap, ate_naive_bootstrap, alpha=0.05, return_ci=False):
    """
    Detect confounding by testing whether adjusted and naive ATEs differ.

    Args:
        ate_bootstrap: array-like, bootstrap samples of adjusted ATE
        ate_naive_bootstrap: array-like, bootstrap samples of naive ATE
        alpha: significance level for CI (default 0.05 -> 95% CI)
        return_ci: if True, also return (low, high) CI tuple for the difference

    Returns:
        bool or (bool, (low, high)):
            confounded flag (True if CI of difference excludes 0),
            and optionally the CI bounds.
    """
    ate_bootstrap = np.asarray(ate_bootstrap, dtype=float)
    ate_naive_bootstrap = np.asarray(ate_naive_bootstrap, dtype=float)

    if ate_bootstrap.shape != ate_naive_bootstrap.shape:
        raise ValueError("Bootstrap arrays must have the same shape.")

    # Drop NaNs pairwise
    mask = ~np.isnan(ate_bootstrap) & ~np.isnan(ate_naive_bootstrap)
    if mask.sum() == 0:
        raise ValueError("No valid bootstrap samples to compare.")
    
    diff = ate_bootstrap[mask] - ate_naive_bootstrap[mask]

    q_low, q_high = 100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)
    lo, hi = np.percentile(diff, [q_low, q_high])

    confounded = (not (lo <= 0.0 <= hi)) and (abs(diff).mean() > max(0.05, 0.1 * abs(ate_naive_bootstrap[mask]).mean()))
    if return_ci:
        return confounded, (float(lo), float(hi))
    return confounded

# PEHE: sqrt(mean((ite_est - ite_true)^2))
def pehe(ite_est, ite_true):
    return float(np.sqrt(np.mean((ite_est - ite_true)**2))) if ite_est is not None else None


def compute_latent_confounder_metrics(latents, confounders, confounder_names, concept_names=None, bins=10, max_samples=None, include_nmi=False, per_dim=False):
    """
    Compute ROC-AUC, NMI, distance correlation, and MMD between continuous latent vectors and binary confounders.
    
    Args:
        latents: Either:
            - numpy array of shape [N, latent_dim] - continuous latent vectors
            - list of tensors/arrays, one per concept [N, latent_per_concept] each
        confounders: numpy array of shape [N, num_confounders] - binary confounder values
        confounder_names: list of confounder names
        concept_names: list of concept names (required if latents is a list)
        bins: number of bins for NMI discretization
        max_samples: if set, subsample to this many samples for dCor/MMD (reduces O(N²) memory)
        include_nmi: if True, compute normalized mutual information (NMI) scores
        per_dim: if True, compute metrics for each latent dimension separately
        
    Returns:
        dict with ROC-AUC, NMI, distance correlation, and MMD scores for each confounder.
        If latents is a list, returns nested dict: {concept_name: {confounder_name: {...}}}
    """
    
    if isinstance(confounders, torch.Tensor):
        confounders = confounders.detach().cpu().numpy()
    
    # Handle list of latents (one per concept)
    if isinstance(latents, list):
        if concept_names is None:
            concept_names = [f"concept_{i}" for i in range(len(latents))]
        
        results = {}
        for i, (z_chunk, concept_name) in enumerate(zip(latents, concept_names)):
            print(f"\n=== Concept: {concept_name} ===")
            results[concept_name] = _compute_single_latent_metrics(
                z_chunk, confounders, confounder_names, bins, max_samples=max_samples, per_dim=per_dim, include_nmi=include_nmi
            )
        return results
    else:
        # Single latent tensor
        return _compute_single_latent_metrics(latents, confounders, confounder_names, bins, max_samples=max_samples, per_dim=per_dim, include_nmi=include_nmi)


def _distance_correlation(X, Y, max_samples=None):
    """
    Compute distance correlation between X (N, d) and Y (N,).
    Distance correlation measures both linear and nonlinear dependence.
    Range: [0, 1], where 0 = independence, 1 = perfect dependence.
    
    Args:
        max_samples: if set, subsample to this many samples to reduce O(N²) memory usage.
    """
    n = X.shape[0]
    
    # Subsample if requested and needed
    if max_samples is not None and n > max_samples:
        idx = np.random.choice(n, size=max_samples, replace=False)
        X = X[idx]
        Y = Y[idx]
        n = max_samples
    
    # Compute pairwise distance matrices
    # For X (multivariate)
    X_diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    A = np.sqrt((X_diff ** 2).sum(axis=2))
    
    # For Y (univariate, treat as 1D)
    Y = Y.reshape(-1, 1)
    Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
    B = np.sqrt((Y_diff ** 2).sum(axis=2))
    
    # Double center the distance matrices
    A_row_mean = A.mean(axis=1, keepdims=True)
    A_col_mean = A.mean(axis=0, keepdims=True)
    A_mean = A.mean()
    A_centered = A - A_row_mean - A_col_mean + A_mean
    
    B_row_mean = B.mean(axis=1, keepdims=True)
    B_col_mean = B.mean(axis=0, keepdims=True)
    B_mean = B.mean()
    B_centered = B - B_row_mean - B_col_mean + B_mean
    
    # Compute distance covariance and variances
    dCov2 = (A_centered * B_centered).sum() / (n * n)
    dVarX = (A_centered * A_centered).sum() / (n * n)
    dVarY = (B_centered * B_centered).sum() / (n * n)
    
    # Distance correlation
    if dVarX > 0 and dVarY > 0:
        dCor = np.sqrt(dCov2 / np.sqrt(dVarX * dVarY))
    else:
        dCor = 0.0
    
    return float(dCor)


def _mmd_rbf(X, Y, gamma=None, max_samples=None):
    """
    Compute Maximum Mean Discrepancy (MMD) with RBF kernel between 
    latent distributions conditioned on binary confounder Y.
    
    Args:
        X: (N, d) latent vectors
        Y: (N,) binary labels (0 or 1)
        gamma: RBF kernel bandwidth (if None, use median heuristic)
        max_samples: if set, subsample each group to at most this many samples
    
    Returns:
        MMD^2 estimate (higher = more different distributions)
    """
    X0 = X[Y == 0]
    X1 = X[Y == 1]
    
    if len(X0) < 2 or len(X1) < 2:
        return None
    
    # Subsample if requested to reduce memory
    if max_samples is not None:
        if len(X0) > max_samples:
            idx0 = np.random.choice(len(X0), size=max_samples, replace=False)
            X0 = X0[idx0]
        if len(X1) > max_samples:
            idx1 = np.random.choice(len(X1), size=max_samples, replace=False)
            X1 = X1[idx1]
    
    # Median heuristic for bandwidth (on subsampled data)
    X_sub = np.vstack([X0, X1])
    if gamma is None:
        # Use a smaller subsample for median heuristic to save memory
        n_med = min(2000, len(X_sub))
        idx_med = np.random.choice(len(X_sub), size=n_med, replace=False)
        X_med = X_sub[idx_med]
        all_dists = np.linalg.norm(X_med[:, np.newaxis] - X_med[np.newaxis, :], axis=2)
        median_dist = np.median(all_dists[all_dists > 0])
        gamma = 1.0 / (2 * median_dist ** 2 + 1e-8)
    
    def rbf_kernel(A, B):
        sq_dist = np.sum(A**2, axis=1, keepdims=True) + np.sum(B**2, axis=1) - 2 * A @ B.T
        return np.exp(-gamma * sq_dist)
    
    K00 = rbf_kernel(X0, X0)
    K11 = rbf_kernel(X1, X1)
    K01 = rbf_kernel(X0, X1)
    
    n0, n1 = len(X0), len(X1)
    
    # Unbiased MMD^2 estimator
    mmd2 = (K00.sum() - np.trace(K00)) / (n0 * (n0 - 1)) \
         + (K11.sum() - np.trace(K11)) / (n1 * (n1 - 1)) \
         - 2 * K01.sum() / (n0 * n1)
    
    return float(max(0, mmd2))  # Clamp to non-negative


def _compute_single_latent_metrics(latents, confounders, confounder_names, bins=10, max_samples=None, per_dim=False, include_nmi=False):
    """
    Compute ROC-AUC, NMI, distance correlation, and MMD for a single latent tensor vs confounders.
    
    Args:
        latents: numpy array or tensor of shape [N, latent_dim]
        confounders: numpy array of shape [N, num_confounders]
        confounder_names: list of confounder names
        bins: number of bins for NMI discretization
        max_samples: if set, subsample for dCor/MMD to reduce O(N²) memory
        per_dim: if True, compute metrics for each latent dimension separately
        include_nmi: if True, compute normalized mutual information (NMI) scores

    Returns:
        If per_dim=False: dict with ROC-AUC, NMI, distance correlation, and MMD scores for each confounder
        If per_dim=True: nested dict {dim: {confounder_name: {...}}} with metrics per dimension
    """
    
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    
    results = {}
    if per_dim:
        # Compute metrics for each dimension separately too
        for d in range(latents.shape[1]):
            z_dim = latents[:, d:d+1]  # Keep as [N, 1] for compatibility
            results[f"dim_{d}"] = _compute_single_latent_metrics(
                z_dim, confounders, confounder_names, bins=bins, max_samples=max_samples, per_dim=False, include_nmi=include_nmi
            )
    
    for j, conf_name in enumerate(confounder_names):
        conf = confounders[:, j].astype(int)
                
        # ROC-AUC: predict confounder from latents using logistic regression
        try:
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(latents, conf)
            probs = clf.predict_proba(latents)[:, 1]
            auc = roc_auc_score(conf, probs)
        except Exception as e:
            print(f"ROC-AUC computation failed for {conf_name}: {e}")
            auc = None
        
        # NMI: average NMI across all latent dimensions
        if include_nmi:
            nmi_scores = []
            for d in range(latents.shape[1]):
                z_dim = latents[:, d]
                z_binned = np.digitize(z_dim, np.histogram(z_dim, bins=bins)[1])
                nmi = normalized_mutual_info_score(z_binned, conf)
                nmi_scores.append(nmi)
            avg_nmi = float(np.mean(nmi_scores))
        
        # Distance correlation: measures nonlinear dependence
        try:
            dcor = _distance_correlation(latents, conf.astype(float), max_samples=max_samples)
        except Exception as e:
            print(f"Distance correlation failed for {conf_name}: {e}")
            dcor = None
        
        # # MMD: measures distribution shift between groups
        # try:
        #     mmd = _mmd_rbf(latents, conf, max_samples=max_samples)
        # except Exception as e:
        #     print(f"MMD computation failed for {conf_name}: {e}")
        #     mmd = None
        
        dict_ = {
            "roc_auc": float(auc) if auc is not None else None,
            "dcor": float(dcor) if dcor is not None else None,
            "nmi": float(avg_nmi) if include_nmi else None
            }
        
        results[conf_name] = dict_

        if not per_dim:
            # Mutual Information per-dimension (for MIG) and predictability per-dimension (for SAP)
            mi_scores = []
            predict_scores = []
            for d in range(latents.shape[1]):
                z_dim = latents[:, d]
                # discretize latents for MI
                z_binned = np.digitize(z_dim, np.histogram(z_dim, bins=bins)[1])
                try:
                    mi = mutual_info_score(z_binned, conf)
                except Exception:
                    mi = 0.0
                mi_scores.append(float(mi))

                # predictability (SAP) using cross-validated accuracy on single-dimension
                try:
                    z_col = z_dim.reshape(-1, 1)
                    clf_dim = LogisticRegression(max_iter=1000, solver='liblinear')
                    scores = cross_val_score(clf_dim, z_col, conf, cv=3, scoring='accuracy')
                    predict_scores.append(float(np.mean(scores)))
                except Exception:
                    predict_scores.append(0.0)

            # MIG: (top MI - second MI) / H(factor)
            try:
                p_vals = np.bincount(conf) / float(len(conf))
                entropy = -np.sum([p * np.log2(p) for p in p_vals if p > 0])
            except Exception:
                entropy = 0.0

            if entropy <= 0:
                mig_k = None
            else:
                sorted_mi = sorted(mi_scores, reverse=True)
                if len(sorted_mi) >= 2:
                    mig_k = float((sorted_mi[0] - sorted_mi[1]) / (entropy + 1e-12))
                else:
                    mig_k = float(sorted_mi[0] / (entropy + 1e-12))

            # SAP: difference between top two predictability scores
            sorted_acc = sorted(predict_scores, reverse=True)
            if len(sorted_acc) >= 2:
                sap_k = float(sorted_acc[0] - sorted_acc[1])
            elif len(sorted_acc) == 1:
                sap_k = float(sorted_acc[0])
            else:
                sap_k = None

            results[conf_name].update({
                "mig": mig_k,
                "sap": sap_k,
            })

            # DCI (Disentanglement, Completeness, Informativeness)
            try:
                D = latents.shape[1]
                K = len(confounder_names)
                if D >= 2 and K >= 1:
                    importances = np.zeros((D, K), dtype=float)
                    informativeness_scores = []
                    for k in range(K):
                        yk = confounders[:, k]
                        if len(np.unique(yk)) < 2:
                            continue
                        try:
                            clf = RandomForestClassifier(n_estimators=100, random_state=0)
                            clf.fit(latents, yk)
                            importances[:, k] = clf.feature_importances_
                            try:
                                scores = cross_val_score(clf, latents, yk, cv=3, scoring='roc_auc')
                                informativeness_scores.append(float(np.mean(scores)))
                            except Exception:
                                informativeness_scores.append(0.0)
                        except Exception:
                            importances[:, k] = 0.0

                    total_imp = importances.sum()
                    eps = 1e-12

                    # Disentanglement per factor
                    importance_per_factor = importances.sum(axis=0) / (total_imp + eps)
                    disent_per_factor = []
                    for k in range(K):
                        imp = importances[:, k]
                        s = imp.sum()
                        if s <= 0:
                            disent_per_factor.append(0.0)
                            continue
                        p = imp / (s + eps)
                        ent = -np.sum([pp * np.log(pp + eps) for pp in p]) / np.log(D + eps)
                        disent_per_factor.append(1.0 - ent)

                    overall_disent = float(np.sum(importance_per_factor * np.array(disent_per_factor)))

                    # Completeness per dimension
                    importance_per_dim = importances.sum(axis=1) / (total_imp + eps)
                    complete_per_dim = []
                    for j in range(D):
                        impj = importances[j, :]
                        s = impj.sum()
                        if s <= 0:
                            complete_per_dim.append(0.0)
                            continue
                        q = impj / (s + eps)
                        ent = -np.sum([qq * np.log(qq + eps) for qq in q]) / np.log(K + eps)
                        complete_per_dim.append(1.0 - ent)

                    overall_complete = float(np.sum(importance_per_dim * np.array(complete_per_dim)))
                    informativeness = float(np.mean(informativeness_scores)) if len(informativeness_scores) > 0 else 0.0

                    results[conf_name]['dci_disentanglemen'] = overall_disent
                    results[conf_name]['dci_completeness'] = overall_complete
                    results[conf_name]['dci_informativeness'] = informativeness
            except Exception as e:
                print(f"DCI computation failed: {e}")

    return results


def analyze_latents(z_c_list, c_arr, shortcut_arr,  y_arr, concept_names, shortcut_names, include_mathews=True, bins=10):
    """
    For each latent confounder dimension, compute Pearson, NMI, and cross-correlation
    with concepts, outcome, and shortcuts.
    
    Args:
        z_c_list: list of [N, latent_per_concept] arrays (per concept latent chunk)
        c_arr: [N, num_concepts]
        y_arr: [N, 1]
        shortcut_arr: [N, num_shortcuts]
    """
    results = {}
    shortcut_results = {}

    for i, z_chunk in enumerate(z_c_list):  # [N, latent_per_concept]
        results[i] = {"pearson": {}, "nmi": {}, "xcorr": {}}
        print(f"\n=== Concept {i} latent chunk ===")

        for d in range(z_chunk.shape[1]):  # latent dim loop
            z_dim = z_chunk[:, d].numpy() if isinstance(z_chunk, torch.Tensor) else z_chunk[:, d]
            z_dim = z_dim.astype(np.float32)

            print(f" Latent dim {d}:")
            # With each concept
            for j in range(c_arr.shape[1]):
                pear = pearson_cc(z_dim, c_arr[:, j])
                nmi = normalized_mutual_info_score(
                    np.digitize(z_dim, np.histogram(z_dim, bins=bins)[1]),
                    c_arr[:, j]
                )
                xcorr = cross_correlation_loss(torch.tensor(z_dim).unsqueeze(1), torch.tensor(c_arr[:, j]).unsqueeze(1))

                results[i]["pearson"][f"dim{d}_C{j}"] = pear
                results[i]["nmi"][f"dim{d}_C{j}"] = nmi
                results[i]["xcorr"][f"dim{d}_C{j}"] = xcorr

                print(f"   ↳ vs C{j}: pearson={pear:.3f}, nmi={nmi:.3f}, xcorr={xcorr:.3f}")

            # With Y
            pear = pearson_cc(z_dim, y_arr.flatten())
            nmi = normalized_mutual_info_score(
                np.digitize(z_dim, np.histogram(z_dim, bins=bins)[1]),
                y_arr.flatten()
            )
            xcorr = cross_correlation_loss(torch.tensor(z_dim).unsqueeze(1), torch.tensor(y_arr))
            results[i]["pearson"][f"dim{d}_Y"] = pear
            results[i]["nmi"][f"dim{d}_Y"] = nmi
            results[i]["xcorr"][f"dim{d}_Y"] = xcorr

            print(f"   ↳ vs Y: pearson={pear:.3f}, nmi={nmi:.3f}, xcorr={xcorr:.3f}")

            shortcut_results[f'{concept_names[i]}_{d}'] = {}
            # With each shortcut
            for j in range(shortcut_arr.shape[1]):
                pear = pearson_cc(z_dim, shortcut_arr[:, j])
                nmi = normalized_mutual_info_score(
                    np.digitize(z_dim, np.histogram(z_dim, bins=bins)[1]),
                    shortcut_arr[:, j]
                )

                results[i]["pearson"][f"dim{d}_S{j}"] = pear
                results[i]["nmi"][f"dim{d}_S{j}"] = nmi

                # xcorr = cross_correlation_loss(torch.tensor(z_dim).unsqueeze(1), torch.tensor(shortcut_arr[:, j]).unsqueeze(1))
                if include_mathews:
                    xcorr = np.abs(matthews_corrcoef(shortcut_arr[:, j], z_dim))
                    results[i]["xcorr"][f"dim{d}_S{j}"] = xcorr
                else:
                    xcorr = pear

                shortcut_results[f'{concept_names[i]}_{d}'][j] = (xcorr.item(), nmi)

                print(f"   ↳ vs S{j}: pearson={pear:.3f}, nmi={nmi:.3f}, xcorr={xcorr:.3f}")

    return shortcut_results

def analyze_latents_simple(z_c, c, s, y, concept_names, shortcut_names, include_mathews=True, bins=10):
    """
    Args:
        z_c: [N, num_concepts] latent confounders
        c: [N, num_concepts] ground truth concepts
        s: [N, num_shortcuts] shortcut features
        y: [N, 1] outcome
        concept_names: list of names for concepts
        shortcut_names: list of names for shortcuts
    """
    N, num_concepts = c.shape
    results = []
    shortcut_results = {}

    corr = matthews_corrcoef if include_mathews else pearson_cc

    for i in range(num_concepts):
        z_dim = z_c[:, i]
        z_tensor = z_dim

        # with each concept
        for j in range(num_concepts):
            target = c[:, j]
            nmi = normalized_mutual_info_score(
                np.digitize(z_dim, np.histogram(z_dim, bins=bins)[1]),
                target
            )
            xcorr = np.abs(corr(target, z_tensor))
            results.append((f"z_c[{concept_names[i]}] ↔ C[{concept_names[j]}]", xcorr, nmi))

        # with outcome
        target = y.flatten()
        nmi = normalized_mutual_info_score(
            np.digitize(z_dim, np.histogram(z_dim, bins=bins)[1]),
            target
        )
        xcorr = np.abs(corr(target, z_tensor))
        results.append((f"z_c[{concept_names[i]}] ↔ Y", xcorr, nmi))

        shortcut_results[concept_names[i]] = {}
        # with shortcuts
        for j in range(s.shape[1]):
            target = s[:, j]
            nmi = normalized_mutual_info_score(
                np.digitize(z_dim, np.histogram(z_dim, bins=bins)[1]),
                target
            )
            xcorr = np.abs(corr(target, z_tensor))
            results.append((f"z_c[{concept_names[i]}] ↔ S[{shortcut_names[j]}]", xcorr, nmi))
            shortcut_results[concept_names[i]][shortcut_names[j]] = (xcorr, nmi)

    # sort by abs correlation
    results.sort(key=lambda x: -x[1])

    print("\n=== Latent Associations (sorted by cross-correlation) ===")
    for name, corr, nmi in results:
        print(f"{name:30s} | corr={corr:.3f}, nmi={nmi:.3f}")

    return shortcut_results

def analyze_latents_flat_shortcut(z, c, s, y, concept_names, shortcut_names, bins=10):
    N, num_shortcuts = s.shape
    shortcut_results = {}

    for i in range(num_shortcuts):
        target = s[:, i]
        shortcut_results[shortcut_names[i]] = {}

        for j in range(z.shape[1]):
            z_dim = z[:, j]
            nmi = normalized_mutual_info_score(
                np.digitize(z_dim, np.histogram(z_dim, bins=bins)[1]),
                target
            )
            pear = pearson_cc(z_dim, target)
            shortcut_results[shortcut_names[i]][j] = (pear.item(), nmi.item())
            print(f"z[{j}] ↔ S[{shortcut_names[i]}]: corr={pear:.3f}, nmi={nmi:.3f}")

    return shortcut_results

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return model

def generate_checkpoint_callback(model_name, dir_path, monitor="val_loss", mode="min", save_last=False, top=1):
    checkpoint_callback = ModelCheckpoint(
    dirpath=dir_path,
    filename= model_name + '-{epoch:02d}',
    monitor=monitor,  # Disable monitoring for checkpoint saving,
    mode = mode,
    save_top_k=top,
    save_last=save_last
    )
    return checkpoint_callback

def generate_early_stopping_callback(patience=5, min_delta = 0.001, monitor="val_loss", mode="min"):
    early_stopping_callback = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, mode = mode)
    return early_stopping_callback

def generate_ema_callback(decay=0.999):
    ema_callback=EMA(decay=decay)
    return ema_callback

def flatten_list(list):
     return sum(list, [])

def get_config(config_dir, default):
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of the config file to choose", type=str, default=default)
    args = argParser.parse_args()
    config = load(open(config_dir + args.name + ".json", "r"))
    return config

def override(f):
    return f

def overload(f):
    return f

def linear_warmup(warmup_iters):
    def f(iter):
        return 1.0 if iter > warmup_iters else iter / warmup_iters

    return f

def init_bias(m):
    if type(m) == nn.Conv2d:
        nn.init.zeros_(m.bias)

def init_weights(layer, std=0.01):
    name = layer.__class__.__name__
    if name == 'ConvBlock':
        return
    if name.startswith('Conv'):
        torch.nn.init.normal_(layer.weight, mean=0, std=std)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)

def continuous_feature_map(c: torch.Tensor, size: tuple = (32, 32)):
    return c.reshape((c.size(0), 1, 1, 1)).repeat(1, 1, *size)

def rgbify(image, normalized=True):
    if image.shape[1] == 1:
        if normalized and torch.min(image) < -0.5:
            # MorphoMNIST: [-1, 1] -> [0, 1]
            image = (image + 1) / 2
        image = image.repeat(1, 3, 1, 1)

    return torch.clamp(image, min=0, max=1)

def orthogonality_penalty(z_c, z_s, eps=1e-8):
    """
    Penalize overlap between z_c and z_s by projecting z_c onto span(z_s).
    """
    B = z_c.size(0)

    # center
    z_c_centered = z_c - z_c.mean(0, keepdim=True)   # [B, d_c]
    z_s_centered = z_s - z_s.mean(0, keepdim=True)   # [B, d_s]

    # compute (Zs^T Zs)^{-1}
    cov_ss = z_s_centered.T @ z_s_centered           # [d_s, d_s]
    cov_ss_inv = torch.linalg.pinv(
        cov_ss + eps * torch.eye(cov_ss.size(0), device=z_s.device)
    )                                                # [d_s, d_s]

    # projection: Zc_proj = Zs ( (Zs^T Zs)^-1 Zs^T Zc )
    proj = z_s_centered @ (cov_ss_inv @ (z_s_centered.T @ z_c_centered))  # [B, d_c]

    # penalty: mean squared projection magnitude
    penalty = (proj ** 2).mean()
    return penalty


def cross_correlation_loss(x1, x2, eps=1e-8):
    x1_centered = x1 - x1.mean(dim=0, keepdim=True)
    x2_centered = x2 - x2.mean(dim=0, keepdim=True)

    x1_std = x1_centered.std(dim=0, unbiased=False).clamp(min=eps)
    x2_std = x2_centered.std(dim=0, unbiased=False).clamp(min=eps)

    corr = (x1_centered.T @ x2_centered) / (x1.shape[0] * x1_std * x2_std)

    return corr.abs().sum()

def pearson_cc(x: np.ndarray, y: np.ndarray, eps=1e-8) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    num = np.sum(x_centered * y_centered)
    denom = np.sqrt(np.sum(x_centered**2)) * np.sqrt(np.sum(y_centered**2)) + eps
    return num / denom

def l2_normalize(x, eps=1e-8):
    # Normalize each column (feature) to unit L2 norm across rows (samples)
    norm = torch.norm(x, dim=0, keepdim=True).clamp(min=eps)
    return x / norm

def cosine_alignment_loss(x1, x2, lambda_orth=1.0, eps=1e-8):
    # 1. Normalize for cosine similarity
    x1_normed = l2_normalize(x1, eps=eps)  # (N, M)
    x2_normed = l2_normalize(x2, eps=eps)         # (N, M)

    # Element-wise cosine similarity between matching features
    sim = (x1_normed * x2_normed).sum(dim=0)  # shape (M,)
    loss = ((1 - sim) ** 2).sum()        # minimum when sim == 1

    return sim, loss

def double_robust_ate_loss(
    r,             # residuals (tensor, requires_grad=True)
    c,           # treatment indicator tensor (0 or 1)
    y,             # observed outcome tensor
    groundtruth_ates, # scalar tensor or float with known ATE for supervision
    lr=1e-3,
    epochs=100,
    batch_size=256,
    device="cuda"
):
    B, K = c.shape
    total_loss = 0.0

    for i in range(K):
        c_i = c[:, i:i+1]
        r_i = r[:, i:i+1]

        # Move inputs to device
        r_i = r_i.to(device).float()
        c_i = c_i.to(device).float()
        y = y.to(device).float()

        # Initialize models
        propensity_model = PropensityScoreModel(input_dim=r_i.shape[1]).to(device)
        outcome_treated = OutcomeEstimator(input_dim=r_i.shape[1]).to(device)
        outcome_control = OutcomeEstimator(input_dim=r_i.shape[1]).to(device)

        # 1) Train propensity score model P(C_i=1|R_i)
        propensity_model.train_model(r_i, c_i, lr=lr, epochs=epochs, batch_size=batch_size, device=device)

        # 2) Train outcome models on treated and untreated separately
        treated_idx = (c_i == 1).nonzero(as_tuple=True)[0]
        control_idx = (c_i == 0).nonzero(as_tuple=True)[0]

        outcome_treated.train_model(r_i[treated_idx], y[treated_idx], lr=lr, epochs=epochs, batch_size=batch_size, device=device)
        outcome_control.train_model(r_i[control_idx], y[control_idx], lr=lr, epochs=epochs, batch_size=batch_size, device=device)

        # 3) Predict propensity scores and potential outcomes
        e_i = propensity_model.get_propensity_score(r_i, device=device).view(-1, 1)  # shape (N,1)
        y1_hat = outcome_treated.predict(r_i, device=device).view(-1, 1)
        y0_hat = outcome_control.predict(r_i, device=device).view(-1, 1)

        # 4) Compute DR estimate
        dr_term = ((c_i.view(-1,1) - e_i) / (e_i * (1 - e_i))) * (y.view(-1,1) - c_i.view(-1,1) * y1_hat - (1 - c_i.view(-1,1)) * y0_hat) + (y1_hat - y0_hat)

        ate_hat = dr_term.mean()

        # 5) Supervised loss comparing estimated ATE to groundtruth ATE
        loss = (ate_hat - groundtruth_ates[i].to(device)) ** 2
        total_loss += loss

    return total_loss / K


def linear_and_orthogonal_loss(x1, x2, lambda_orth=1.0, eps=1e-8):
    """
    Args:
        x1: Tensor (N, M+1)
        x2: Tensor (N, M)
        lambda_orth: weight for orthogonality loss
    Returns:
        scalar loss enforcing:
            - first M cols of x1 to be correlated with x2
            - last col of x1 to be orthogonal to x2
    """
    N, M = x2.shape
    assert (x1.shape[1] == M + 1)

    # Center the columns
    x1_centered = x1 - x1.mean(dim=0, keepdim=True)
    x2_centered = x2 - x2.mean(dim=0, keepdim=True)

    # Std dev for normalization (avoid divide by zero)
    x1_std = x1_centered.std(dim=0, unbiased=False).clamp(min=eps)
    x2_std = x2_centered.std(dim=0, unbiased=False).clamp(min=eps)

    # 1) Element-wise correlation for first M columns
    cov_first_M = (x1_centered[:, :M] * x2_centered).sum(dim=0) / (N - 1)  # shape (M,)
    corr_first_M = cov_first_M / (x1_std[:M] * x2_std[:M])  # shape (M,)

    loss_corr = -corr_first_M.abs().sum()  # maximize absolute correlation

    # 2) Covariance between last column of x1 and all columns of x2
    last_x1 = x1_centered[:, M].unsqueeze(1)  # (N,1)
    cov_orth = (last_x1.T @ x2_centered).squeeze() / (N - 1)  # (M,)
    loss_orth = cov_orth.abs().sum()  # minimize covariance

    loss = loss_corr + lambda_orth * loss_orth
    return loss


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=0.5):
        return (input > threshold).to(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

# CaCe Score (https://arxiv.org/pdf/1907.07165)
def cace_score(c_pred_c0, c_pred_c1):
    """
    Computes the CACE score.
    Args:
        c_pred_c0: torch.Tensor, shape (batch_size, number of concepts)
                   concept values where we did do intervention making a concept equal to 0
        c_pred_c1: torch.Tensor, shape (batch_size, number of concepts)
                   concept values where we did do intervention making a concept equal to 1
    Returns:
        cace: torch.Tensor, shape (number of concepts)
    """
    cace = torch.abs(c_pred_c1.mean(dim=0) - c_pred_c0.mean(dim=0))
    return cace


def kl_cyclic_cosine(step, cycle_len=50, n_cycles=4, max_beta=1.0):
    cycle = step % cycle_len
    progress = cycle / cycle_len
    beta = max_beta * (1 - math.cos(progress * math.pi)) / 2  # smooth ramp
    if step >= cycle_len * n_cycles:
        beta = max_beta
    beta = max(beta, 1e-2)  # avoid 0
    return beta


def pcf_fit(
    z_s,
    c,
    y,
    concept_names,
    n_components=None,
    random_state=0,
    top_k=3,
):
    """
    Fit PCA-PCF on TRAIN: run PCA on z_s, compute p-values for each component
    in `c_j ~ z_i` and `y ~ [c, z_i]`, and select indices minimizing p_x + p_y.

    Returns:
        pca: fitted PCA object
        selected_indices: list[int] component index per concept (best by p_x+p_y)
        rankings: list[list[dict]] per concept with entries {"idx","sum_p","p_x","p_y"}
        topK_indices: list[list[int]] per concept of the top-K component indices
    """
    # local imports
    from sklearn.decomposition import PCA
    import statsmodels.api as sm

    # to numpy
    z_s_np = z_s.detach().cpu().numpy() if isinstance(z_s, torch.Tensor) else np.asarray(z_s)
    c_np = c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else np.asarray(c)
    y_np = (y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)).reshape(-1)

    N, D_s = z_s_np.shape
    _, K = c_np.shape
    n_comp = D_s if n_components is None else int(n_components)

    pca = PCA(n_components=n_comp, random_state=random_state, whiten=True)
    Z = pca.fit_transform(z_s_np)

    rankings = []
    topK_indices = []
    for j in range(K):
        c_j = c_np[:, j]
        comp_stats = []
        for i in range(n_comp):
            z_i = Z[:, i].reshape(-1, 1)
            # p_c_j from c_j ~ z_i
            try:
                X_c = sm.add_constant(z_i)
                res_c = sm.Logit(c_j, X_c).fit(disp=0)
                p_c_j = float(res_c.pvalues[1])
            except Exception:
                p_c_j = 1.0
            # p_y from y ~ [c, z_i]
            try:
                X_y = np.concatenate([c_np, z_i], axis=1)
                X_y = sm.add_constant(X_y)
                res_y = sm.Logit(y_np, X_y).fit(disp=0)
                p_y = float(res_y.pvalues[X_y.shape[1]-1])
            except Exception:
                p_y = 1.0
            sum_p = p_c_j + p_y
            comp_stats.append({"idx": i, "sum_p": sum_p, "p_c_j": p_c_j, "p_y": p_y})
        # Lower (p_c_j + p_y) means more predictive; so sort by sum_p.
        comp_stats.sort(key=lambda d: d["sum_p"])  # descending by predictiveness
        rankings.append(comp_stats)
        # collect top-K indices (descending predictiveness)
        k = max(1, int(top_k))
        topK_indices.append([d["idx"] for d in comp_stats[:k]])

    return pca, rankings, topK_indices


def pcf_apply(pca, topK_indices, z_s):
    """
    Apply trained PCA and selection to new z_s to produce z_c estimate.
    Returns z_c_est: numpy array [N, K]
    """
    z_s_np = z_s.detach().cpu().numpy() if isinstance(z_s, torch.Tensor) else np.asarray(z_s)
    Z = pca.transform(z_s_np)
    z_c_est = np.zeros((len(topK_indices), Z.shape[0], len(topK_indices[0])), dtype=float)
    for i, indices in enumerate(topK_indices):
        for j, idx in enumerate(indices):
            z_c_est[i, :, j] = Z[:, idx]
    return z_c_est

def compute_true_ite_ate(
    C_full: Union[torch.Tensor, np.ndarray],
    indices,
    concept_idx: int,
    coeffs: Union[dict, np.ndarray, Sequence[float]],
    causal_concepts: Optional[Sequence[str]] = None,
    causal_concept_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Compute ground-truth ITE/ATE given concatenated attributes C_full = [shortcuts | concepts].
    - If coeffs is a dict: uses base + per-concept + optional pairwise interaction betas (preferred).
    - Else: legacy vector-of-coefficients path (threshold at 0.5).
    Returns (ite_true, ate_true).
    """
    if isinstance(C_full, np.ndarray):
        C_full = torch.from_numpy(C_full)
    C_full = C_full.float()
    device = C_full.device

    # Normalize indices to Python lists of ints
    concept_ids = [int(x) for x in (indices['concepts'].cpu().numpy().tolist() if torch.is_tensor(indices['concepts']) else indices['concepts'])]
    shortcut_ids = [int(x) for x in (indices['shortcut'].cpu().numpy().tolist() if torch.is_tensor(indices['shortcut']) else indices['shortcut'])]
    shortcut_ids = [x for x in shortcut_ids if x not in concept_ids]

    attr_ids = concept_ids + shortcut_ids  # filtered attr_ids

    # Build a mapping from attr_id -> filtered C_full column index
    # Filtered layout is [remaining_shortcuts | concepts]
    attr_id_to_col = {attr_id: col for col, attr_id in enumerate(attr_ids)}

    if isinstance(coeffs, dict):
        if causal_concepts is None or causal_concept_indices is None:
            raise ValueError("For dict coeffs, provide causal_concepts and causal_concept_indices.")
        # Do NOT overwrite attr_ids; use attr_id_to_col computed above

        def build_p(C: torch.Tensor) -> torch.Tensor:
            p = torch.full((C.shape[0],), float(coeffs.get("base", 0.5)), dtype=torch.float32, device=device)
            # main effects
            for name, attr_id in zip(causal_concepts, causal_concept_indices):
                beta = float(coeffs.get(name, 0.0))
                col = attr_id_to_col[int(attr_id)]
                p = p + beta * C[:, col].float()
            # pairwise interactions
            for i in range(len(causal_concepts)):
                for j in range(i + 1, len(causal_concepts)):
                    key = f"{causal_concepts[i]}_{causal_concepts[j]}"
                    if key in coeffs:
                        beta_ij = float(coeffs[key])
                        col_i = attr_id_to_col[int(causal_concept_indices[i])]
                        col_j = attr_id_to_col[int(causal_concept_indices[j])]
                        p = p + beta_ij * (C[:, col_i].float() * C[:, col_j].float())
            return p

        C1 = C_full.clone(); C1[:, concept_idx] = 1.0
        C0 = C_full.clone(); C0[:, concept_idx] = 0.0
        y1_true = build_p(C1)
        y0_true = build_p(C0)
        ite_true = (y1_true - y0_true).cpu().numpy()
        ate_true = float(ite_true.mean())
    else:
        # Vector coeffs path: align coeffs to filtered attr_ids
        coeffs_np_full = np.array(coeffs, dtype=float)
        # Slice coeffs by filtered attr_ids ordering
        coeffs_np = coeffs_np_full[attr_ids]
        C_np = C_full.cpu().numpy()
        C1 = C_np.copy(); C0 = C_np.copy()
        C1[:, concept_idx] = 1.0
        C0[:, concept_idx] = 0.0

        p1 = (C1 @ coeffs_np).astype(float)
        # y1_true = np.random.binomial(n=1, p=p1, size=C1.shape[0]).astype(float)
        p0 = (C0 @ coeffs_np).astype(float)
        # y0_true = np.random.binomial(n=1, p=p0, size=C0.shape[0]).astype(float)
        # ite_true = y1_true - y0_true
        ite_true = p1 - p0
        ate_true = float(ite_true.mean())
    
    return ite_true, ate_true

class _GradientReverseFn(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GRLWrapper(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return _GradientReverseFn.apply(x, float(self.lambd))
    