# models/aipw_utils.py
import warnings
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Any, Callable, Dict
from scipy.special import expit
from scipy.optimize import root_scalar
import statsmodels.api as sm
from scipy.stats import ks_2samp


def _build_z_from_out(out, latent_keys: Optional[list] = None):
    """Build a single (B,D) tensor z from an inference output dict `out`.
    """
    parts = []
    if latent_keys is not None and len(latent_keys) > 0:
        for key in latent_keys:
            val = out.get(key)
            if val is None:
                continue
            parts.append(val)
        if len(parts) > 0:
            try:
                return torch.cat(parts, dim=1)
            except Exception:
                return parts[0]

    # fallback ordering: z_c -> z
    if out.get('z_c') is not None:
        zc = out.get('z_c')
        return zc
    if out.get('z') is not None:
        return out.get('z')
    return None

def collect_latents_from_dataloader(model: torch.nn.Module, dataloader, device='cuda', latent_keys: Optional[list] = None, mc_samples: int = 100):
    """
    Collect learned latent proxies and attributes from a dataloader.

    If the model exposes per-concept latents (`L>0`) and a `concept_idx` is provided,
    this returns the per-concept latent slice for that concept. Otherwise it will
    return the model's `z_c` if present, or the combined `z` latent. If no learned
    latents are available, it will fall back to observed shortcuts (if present in
    `model.indices['shortcut']`).
    """
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
    else:
        model.to(device)

    zs, Cs, ys = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x, attr = batch
            y = attr[:, model.indices['task']]
            c = attr[:, model.indices['concepts']]
            x, c, y = x.to(device), c.to(device).float(), y.to(device).float()

            samples = []
            for _ in range(mc_samples):
                out_s = model.infer_latents(x, c, y, binarize=True)
                z_s = _build_z_from_out(out_s, latent_keys=latent_keys)
                if z_s is None:
                    # fallback to deterministic observed shortcuts or concepts
                    if model.indices.get('shortcut') is not None and len(model.indices['shortcut']) > 0:
                        z_s = attr[:, model.indices['shortcut']].to(device).float()
                    else:
                        z_s = c
                samples.append(z_s.unsqueeze(0))
            z_stack = torch.cat(samples, dim=0)  # (S, B, D)
            z_mean = z_stack.mean(dim=0)  # (B, D)

            zs.append(z_mean.cpu().numpy())
            Cs.append(c.cpu().numpy())
            if y.dim() > 1:
                y = y.squeeze(1)
            ys.append(y.cpu().numpy())

    z_arr = np.concatenate(zs, axis=0)
    c_arr = np.concatenate(Cs, axis=0)
    y_arr = np.concatenate(ys, axis=0)
    return z_arr, c_arr, y_arr


def identify_confounders(
    z_data,
    c_data,
    y_data,
    ks_threshold: float = 0.05,
    pred_pval: float = 0.05,
    min_samples: int = 20,
    coef_tol: float = 1e-3,
    random_state: int = 0,
):
    """
    Identify candidate confounder dimensions among latent `z_data` and return a
    minimal adjustment set.

    Procedure:
    - Require a dimension to be predictive of both `C` and `Y` (two-sample t-test,
      p < `pred_pval`) to be a candidate.
    - Use a centered KS-test between Z|C=0 and Z|C=1 to distinguish confounder
      (different shapes) from mediator (shifted distribution).
    - From candidate confounders, greedily select a minimal adjustment set that
      reduces the absolute treatment coefficient in a logistic regression of
      `Y ~ A + Z_selected` (stop when no substantial improvement > `coef_tol`).

    Returns dict with keys:
        'confounders': numpy.bool array length D marking candidate confounders
        'adjustment_set': numpy.bool array length D marking the selected minimal set
    """

    z = np.asarray(z_data)
    c = np.asarray(c_data).astype(int).ravel()
    y = np.asarray(y_data).astype(int).ravel()

    if z.ndim != 2:
        raise ValueError("z_data must be 2D array-like (N, D)")

    # assume caller scales latents if desired

    N, D = z.shape
    candidate_mask = np.zeros(D, dtype=bool)

    for d in range(D):
        z_d = z[:, d]
        z0 = z_d[c == 0]
        z1 = z_d[c == 1]

        # Conservative: if too few samples per group, mark as candidate (keep)
        if len(z0) < min_samples or len(z1) < min_samples:
            candidate_mask[d] = True
            continue

        # Predictive of C via logistic regression c ~ z_d
        X_c = sm.add_constant(z_d.reshape(-1, 1))
        res_c = sm.Logit(c, X_c).fit(disp=0)
        p_c = float(res_c.pvalues[1])
        if p_c >= pred_pval:
            continue

    # Predictive of Y conditional on C via logistic regression y ~ [c, z_d]
        c_np = c.reshape(-1, 1)
        X_y = np.concatenate([c_np, z_d.reshape(-1, 1)], axis=1)
        X_y = sm.add_constant(X_y)
        res_y = sm.Logit(y, X_y).fit(disp=0)
        p_y = float(res_y.pvalues[-1])

        if p_y >= pred_pval:
            continue

        # KS test on centered distributions to detect shape differences
        stat, p_ks = ks_2samp(z0 - z0.mean(), z1 - z1.mean())
        if p_ks < ks_threshold:
            candidate_mask[d] = True

    # If no candidates, return empty masks
    if not candidate_mask.any():
        return {"confounders": candidate_mask, "adjustment_set": np.zeros(D, dtype=bool)}

    candidates = list(np.where(candidate_mask)[0])

    A = c.astype(float)
    Y = y.astype(int)

    def fit_abs_treatment_coef(selected_indices):
        # Fit logistic regression of Y ~ A (+ selected Zs) and return abs coef for A.
        if len(selected_indices) == 0:
            X = A.reshape(-1, 1)
        else:
            X = np.column_stack([A, z[:, selected_indices]])
        try:
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state, penalty='l2', solver='liblinear'))
            lr.fit(X, Y)
            coef_A = float(lr.coef_[0][0])
        except Exception:
            # fallback: use difference-in-means as crude proxy
            treated_mean = Y[A == 1].mean() if (A == 1).any() else 0.0
            control_mean = Y[A == 0].mean() if (A == 0).any() else 0.0
            coef_A = treated_mean - control_mean
        return abs(coef_A)

    # Greedy forward selection minimizing abs(coef_A)
    remaining = candidates.copy()
    selected = []
    best_val = fit_abs_treatment_coef([])
    improved = True

    while improved and remaining:
        improved = False
        best_candidate = None
        best_candidate_val = best_val
        for idx in remaining:
            val = fit_abs_treatment_coef(selected + [idx])
            if val < best_candidate_val - coef_tol:
                best_candidate_val = val
                best_candidate = idx
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_val = best_candidate_val
            improved = True

    adjustment_mask = np.zeros(D, dtype=bool)
    if len(selected) > 0:
        adjustment_mask[selected] = True

    return {"confounders": candidate_mask, "adjustment_set": adjustment_mask}


def aipw_crossfit(
    train_data,
    test_data,
    concept_idx: int,
    n_splits: int = 1,   # note: currently only single train->test is supported
    propensity_model_builder: Optional[Callable[[], Any]] = None,
    outcome_model_builder: Optional[Callable[[], Any]] = None,
    random_state: int = 0,
    run_double_ml: bool = False,
    clip_extreme_propensities: bool = False,
    detect_confounders: bool = False,
    adjust_for_other_concepts: bool = True,
) -> Dict[str, Tuple[float, float, Tuple[float, float]]]:
    """
    Single train->test AIPW + partially-linear DML for binary Y and binary A.

    NOTE: despite the name, this implementation currently performs a
    single train -> test nuisance fit (n_splits must be 1). If you want
    full K-fold cross-fitting, we can implement it as a follow-up.

    Returns dict with keys 'ipw', 'adjustment', 'aipw' and optionally 'double_ml'.
    Each maps to (tau, se, (ci_low, ci_high)).
    """
    if propensity_model_builder is None:
        def propensity_model_builder():
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state, penalty='l2', solver='liblinear'))
        
    if outcome_model_builder is None:
        def outcome_model_builder():
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state, penalty='l2', solver='liblinear'))

    # Require single train/test for now
    assert n_splits == 1 and train_data is not None, "Currently only single train->test fit is supported."

    # Unpack
    Z, C_all, Y = test_data
    Z_train, C_train, Y_train = train_data

    # ensure numpy arrays and proper shapes
    Z = np.asarray(Z)
    Z_train = np.asarray(Z_train)
    C_all = np.asarray(C_all)
    C_train = np.asarray(C_train)
    Y = np.asarray(Y).squeeze()
    Y_train = np.asarray(Y_train).squeeze()

    N = Z.shape[0]
    k = C_all.shape[1]
    assert 0 <= concept_idx < k

    # treatment vectors
    A = C_all[:, concept_idx].astype(float)
    A_train = C_train[:, concept_idx].astype(float)

    # enforce binary treatment and binary outcome
    a_vals = np.unique(A)
    a_train_vals = np.unique(A_train)
    y_vals = np.unique(Y)
    y_train_vals = np.unique(Y_train)

    if not set(a_vals).issubset({0, 1}):
        raise ValueError(f"Test treatment values must be binary 0/1, got {a_vals}")
    if not set(a_train_vals).issubset({0, 1}):
        raise ValueError(f"Train treatment values must be binary 0/1, got {a_train_vals}")

    if not set(y_vals).issubset({0, 1}):
        raise ValueError(f"Test outcome values must be binary 0/1, got {y_vals}")
    if not set(y_train_vals).issubset({0, 1}):
        raise ValueError(f"Train outcome values must be binary 0/1, got {y_train_vals}")

    # NOTE: do not scale Z alone here. We'll scale the full covariate
    # matrices (Z + observed concepts) below so nuisance models see
    # covariates on the same scale as the baselines (e.g. pipelines).

    # observed concepts to optionally adjust for (all concepts except treatment)
    other_idx = [i for i in range(k) if i != concept_idx]
    if adjust_for_other_concepts and len(other_idx) > 0:
        obs_train = C_train[:, other_idx]
        obs_test = C_all[:, other_idx]
    else:
        obs_train = None
        obs_test = None

    # Identify confounder candidates and minimal adjustment set if requested
    confounder_info = None
    adjustment_mask = None
    if detect_confounders:
        confounder_info = identify_confounders(Z_train, A_train, Y_train, random_state=random_state)
        if confounder_info is not None:
            confounder_mask = confounder_info.get('confounders')
            adjustment_mask = confounder_info.get('adjustment_set')
            kept = int(confounder_mask.sum())
            total = int(len(confounder_mask))
            adj_kept = int(adjustment_mask.sum()) if adjustment_mask is not None else 0
            print(f"Adjustment set detection: Found {kept}/{total} candidate confounders; minimal adjustment set size {adj_kept}.")
            print(f"Confounder indices: {np.where(confounder_mask)[0].tolist()}")
            print(f"Adjustment set indices: {np.where(adjustment_mask)[0].tolist() if adjustment_mask is not None else None}")
            if adjustment_mask is not None and adjustment_mask.sum() > 0:
                Z_train = Z_train[:, adjustment_mask]
                Z = Z[:, adjustment_mask]

    # build covariate matrices for nuisance fits (Z + observed concepts)
    if obs_train is not None:
        X_train_cov = np.column_stack([Z_train, obs_train])
        X_test_cov = np.column_stack([Z, obs_test])
    else:
        X_train_cov = Z_train
        X_test_cov = Z

    # storage for nuisance fits (single-split)
    pi_hat = np.zeros(N, dtype=float)
    mu0_hat = np.zeros(N, dtype=float)
    mu1_hat = np.zeros(N, dtype=float)
    m_hat = np.zeros(N, dtype=float)   # log-odds baseline for DML (if computed)

    # ---------- Propensity model ----------
    pm = propensity_model_builder()
    if not hasattr(pm, "predict_proba"):
        raise ValueError("Propensity model must implement predict_proba")

    try:
        pm.fit(X_train_cov, A_train)
        pi_val = pm.predict_proba(X_test_cov)[:, 1]
        pi_hat = np.clip(pi_val, 1e-6, 1 - 1e-6)
    except Exception as e:
        raise RuntimeError(f"Propensity model training failed (train->test): {e}")

    # ---------- Outcome S-learner (for mu0, mu1 used by AIPW) ----------
    try:
        # degenerate case: all treated or all untreated in training
        if np.unique(A_train).size == 1:
            p = float(Y_train.mean())
            mu1_val = np.full(N, p)
            mu0_val = np.full(N, p)
        else:
            om = outcome_model_builder()
            if not hasattr(om, "predict_proba"):
                raise ValueError("Outcome model must implement predict_proba for binary outcomes")

            X_train = np.column_stack([X_train_cov, A_train])
            om.fit(X_train, Y_train)

            X1 = np.column_stack([X_test_cov, np.ones(N)])
            X0 = np.column_stack([X_test_cov, np.zeros(N)])

            mu1_val = om.predict_proba(X1)[:, 1]
            mu0_val = om.predict_proba(X0)[:, 1]

        mu1_hat = np.clip(mu1_val, 1e-6, 1 - 1e-6)
        mu0_hat = np.clip(mu0_val, 1e-6, 1 - 1e-6)
    except Exception as e:
        raise RuntimeError(f"Outcome model training failed (train->test S-learner): {e}")

    # ---------- DML outcome baseline m(Z) = logit P(Y=1 | A=0, Z) ----------
    if run_double_ml:
        idx0 = (A_train == 0)
        if idx0.sum() == 0:
            raise RuntimeError("No untreated units in training data; cannot fit DML baseline m(Z).")

        m_model = outcome_model_builder()
        if not hasattr(m_model, "predict_proba"):
            raise ValueError("Outcome model for m(Z) must implement predict_proba")

        try:
            m_model.fit(X_train_cov[idx0], Y_train[idx0])
            p_m = np.clip(m_model.predict_proba(X_test_cov)[:, 1], 1e-6, 1 - 1e-6)
            m_hat = np.log(p_m / (1 - p_m))   # keep as log-odds; DO NOT clip
        except Exception as e:
            raise RuntimeError(f"Failed to fit m(Z) on untreated units: {e}")

    # ---------- Trim extreme propensities ----------
    # overlap diagnostic
    frac_extreme = np.mean((pi_hat < 0.05) | (pi_hat > 0.95))
    if frac_extreme > 0.05:
        warnings.warn(
            f"{frac_extreme*100:.1f}% of propensity scores are < 0.05 or > 0.95; "
            "AIPW/DML may be unstable."
        )

    if clip_extreme_propensities and frac_extreme > 0.05:
        mask = (pi_hat > 0.05) & (pi_hat < 0.95)
        Z = Z[mask]
        Y = Y[mask]
        A = A[mask]
        pi_hat = pi_hat[mask]
        mu0_hat = mu0_hat[mask]
        mu1_hat = mu1_hat[mask]
        m_hat = m_hat[mask] if run_double_ml else m_hat
        N = mask.sum()
        print(f"Trimming samples from {mask.size} to {N} based on clipping propensities in (0.05, 0.95).")

    # ---------- AIPW influence and estimates ----------
    z95 = 1.96
    eps = 1e-6

    term1 = mu1_hat - mu0_hat
    term2 = A * (Y - mu1_hat) / (pi_hat + eps)
    term3 = (1 - A) * (Y - mu0_hat) / (1 - pi_hat + eps)
    IF = term1 + term2 - term3
    tau_aipw = float(IF.mean())
    se_aipw = float(np.sqrt(IF.var(ddof=1) / N))
    ci_aipw = (tau_aipw - z95 * se_aipw, tau_aipw + z95 * se_aipw)

    # IPW estimator
    with np.errstate(divide='ignore', invalid='ignore'):
        psi_ipw = A * Y / (pi_hat + eps) - (1 - A) * Y / (1 - pi_hat + eps)
    tau_ipw = float(np.nanmean(psi_ipw))
    var_ipw = float(np.nanvar(psi_ipw, ddof=1))
    se_ipw = float(np.sqrt(var_ipw / N))
    ci_ipw = (tau_ipw - z95 * se_ipw, tau_ipw + z95 * se_ipw)

    # Outcome regression (adjustment)
    psi_reg = mu1_hat - mu0_hat
    tau_reg = float(np.mean(psi_reg))
    var_reg = float(np.var(psi_reg, ddof=1))
    se_reg = float(np.sqrt(var_reg / N))
    ci_reg = (tau_reg - z95 * se_reg, tau_reg + z95 * se_reg)

    results = {
        'ipw': (tau_ipw, se_ipw, ci_ipw),
        'adjustment': (tau_reg, se_reg, ci_reg),
        'aipw': (tau_aipw, se_aipw, ci_aipw),
    }

    # ---------- Double-ML (partially linear logistic) ----------
    if run_double_ml:
        g_hat = pi_hat
        A_res = A - g_hat

        def score(tau):
            lin = m_hat + tau * A
            mu = expit(lin)
            return float(np.mean(A_res * (Y - mu)))

        # ensure root exists on bracket (-10, 10)
        f_lo = score(-10.0)
        f_hi = score(10.0)
        if f_lo * f_hi > 0:
            raise RuntimeError(
                "DML root-finding failed: score(-10) and score(10) have same sign. "
                "Try broader bracket or check nuisances."
            )

        sol = root_scalar(score, bracket=(-10.0, 10.0), method="bisect")
        tau_dml = float(sol.root)

        lin = m_hat + tau_dml * A
        mu = expit(lin)
        psi = A_res * (Y - mu)

        J = -np.mean(A_res * A * mu * (1 - mu))
        if abs(J) < 1e-8:
            raise RuntimeError("Near-zero Jacobian in DML; cannot compute standard error.")

        IF_dml = psi / J
        se_dml = float(np.sqrt(IF_dml.var(ddof=1) / N))
        ci_dml = (tau_dml - z95 * se_dml, tau_dml + z95 * se_dml)

        results["double_ml"] = (tau_dml, se_dml, ci_dml)

    return results
