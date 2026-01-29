import numpy as np
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import numpy as np
import joblib
import pytorch_lightning as pl
from .utils import detect_confounding, pehe, compute_true_ite_ate
from .image_modules import MorphoMNISTEncoder as ImageEncoder


class PropensityModel(pl.LightningModule):
    def __init__(self, feat_dim, num_concepts, indices, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.indices = indices
        self.num_concepts = num_concepts

        # shared encoder (optional: you can use separate encoders)
        self.encoder = ImageEncoder(feat_dim=feat_dim)

        # classifiers: one per concept
        self.classifiers = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())
            for _ in range(num_concepts)
        ])

        # metrics
        self.accuracies = nn.ModuleList([Accuracy(task='binary') for _ in range(num_concepts)])

    def forward(self, x, idx):
        """
        x: (B, C, H, W)
        idx: index of concept to predict
        returns: propensity score (B,)
        """
        h = self.encoder(x)
        return self.classifiers[idx](h).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, attr = batch
        loss = 0.0
        for i in range(self.num_concepts):
            t = attr[:, self.indices['concepts'][i]].float()
            p_hat = self.forward(x, i)
            loss += F.binary_cross_entropy(p_hat, t)
            self.accuracies[i](p_hat, t.int())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, attr = batch
        val_loss = 0.0
        for i in range(self.num_concepts):
            t = attr[:, self.indices['concepts'][i]].float()
            p_hat = self.forward(x, i)
            val_loss += F.binary_cross_entropy(p_hat, t)
            self.log(f"val_acc_{i}", self.accuracies[i](p_hat, t.int()), prog_bar=True)
        self.log("val_loss", val_loss)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, attr = batch
        for i in range(self.num_concepts):
            t = attr[:, self.indices['concepts'][i]].float()
            p_hat = self.forward(x, i)
            self.log(f"test_acc_{i}", self.accuracies[i](p_hat, t.int()), prog_bar=True)
        return

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


    @torch.no_grad()
    def compare_ates(
        self,
        dataloader,
        concept_idx,
        naive=None,
        coeffs=None,
        pseudo_oracle=None,
        device="cuda",
        n_boot=1000,
        causal_concepts=None,
        causal_concept_indices=None,
        seed=42,
        doubly_robust: bool = True,
    ):
        """
        Compute ATEs and PEHEs for naive, pseudo-oracle, and IPW estimators.
        Also returns abs_diff, ate_error, and a confounded flag.
        """
        self.eval()
        self.to(device)

        y1_naive_list, y0_naive_list = [], []
        y1_po_list, y0_po_list = [], []
        obs_t_list, obs_y_list, prop_list, all_C = [], [], [], []
        pred_concepts_list = []

        for x, attr in dataloader:
            x = x.to(device)
            # observed treatment and outcome
            obs_t = attr[:, self.indices['concepts']][:, concept_idx].cpu().numpy().astype(float)
            obs_y = attr[:, self.indices['task']].cpu().numpy().astype(float)
            obs_t_list.append(obs_t)
            obs_y_list.append(obs_y)

            # features for S-learners
            X_concepts = attr[:, self.indices['concepts']].cpu().numpy()
            extra_indices = [s for s in self.indices['shortcut'] if s not in self.indices['concepts']]
            X_all = np.concatenate([X_concepts,
                                    attr[:, extra_indices].cpu().numpy()], axis=1)
            

            # naive S-learner
            if naive is not None:
                X1 = X_concepts.copy(); X1[:, concept_idx] = 1
                X0 = X_concepts.copy(); X0[:, concept_idx] = 0
                y1_naive_list.append(naive.predict_proba(X1)[:, 1])
                y0_naive_list.append(naive.predict_proba(X0)[:, 1])

            # pseudo-oracle S-learner
            if pseudo_oracle is not None:
                X1_all = X_all.copy(); X1_all[:, concept_idx] = 1
                X0_all = X_all.copy(); X0_all[:, concept_idx] = 0
                y1_po_list.append(pseudo_oracle.predict_proba(X1_all)[:, 1])
                y0_po_list.append(pseudo_oracle.predict_proba(X0_all)[:, 1])

            # propensity scores
            ps = self.forward(x, concept_idx).cpu().numpy()
            prop_list.append(ps)

            # --- PREDICTED CONCEPTS (CBM-style) for fairness
            pred_probs_batch = []
            for i_con in range(self.num_concepts):
                p_i = self.forward(x, i_con).cpu().numpy()          # shape (B,)
                pred_probs_batch.append(p_i)
            # stack -> (B, C)
            pred_probs_batch = np.stack(pred_probs_batch, axis=1)
            # binary predicted concepts (threshold at 0.5)
            X_concepts_pred = (pred_probs_batch > 0.5).astype(float)
            pred_concepts_list.append(X_concepts_pred)

            # collect ground-truth concepts for true ATE/PEHE
            if coeffs is not None:
                # c_full_batch = torch.cat([attr[:, self.indices['shortcut']], attr[:, self.indices['concepts']]], dim=1)
                all_C.append(X_all)

        # aggregate predicted concepts across batches
        if pred_concepts_list:
            X_concepts_pred_all = np.concatenate(pred_concepts_list, axis=0)  # (N, C)
            # CBM naive S-learner: set treatment on predicted concepts
            X1_pred = X_concepts_pred_all.copy(); X1_pred[:, concept_idx] = 1
            X0_pred = X_concepts_pred_all.copy(); X0_pred[:, concept_idx] = 0
            # evaluate using the (sklearn) naive predictor
            if naive is not None:
                y1_naive_pred = naive.predict_proba(X1_pred)[:, 1]
                y0_naive_pred = naive.predict_proba(X0_pred)[:, 1]
                ite_naive_pred = y1_naive_pred - y0_naive_pred
                ate_naive_pred = float(ite_naive_pred.mean())
            else:
                ite_naive_pred = None
                ate_naive_pred = None
        else:
            ite_naive_pred = None
            ate_naive_pred = None


        # --- aggregate naive S-learner
        ite_naive = np.concatenate(y1_naive_list) - np.concatenate(y0_naive_list) if y1_naive_list else None
        ate_naive = float(ite_naive.mean()) if ite_naive is not None else None

        # --- aggregate pseudo-oracle S-learner
        ite_po = np.concatenate(y1_po_list) - np.concatenate(y0_po_list) if y1_po_list else None
        ate_po = float(ite_po.mean()) if ite_po is not None else None

        # --- aggregate IPW with stabilized weights
        ite_ipw, ate_ipw = None, None
        if prop_list:
            T = np.concatenate(obs_t_list)
            Y = np.concatenate(obs_y_list)
            e = np.concatenate(prop_list)

            # # Discard units with estimated propensity outside [0.01, 0.99]
            # unclipped_e = e.copy()
            # keep_mask = (unclipped_e >= 0.01) & (unclipped_e <= 0.99)
            # if not np.any(keep_mask):
            #     raise ValueError("No units remain after propensity trimming in [0.01,0.99].")
            # # apply mask to observed arrays used for IPW / DR / bootstrap
            # T = T[keep_mask]
            # Y = Y[keep_mask]
            # e = unclipped_e[keep_mask]

            eps = 1e-6
            e = np.clip(e, eps, 1-eps)
            p = T.mean()  # marginal treatment probability
            w = T * p / e + (1 - T) * (1 - p) / (1 - e)  # stabilized weights

            # # Cap extreme weights at a high quantile to reduce variance from extreme propensities
            try:
                cap_q = 0.9
                w_cap = np.quantile(w, cap_q)
                if w_cap > 0:
                    w = np.minimum(w, w_cap)
            except Exception:
                # if quantile computation fails, continue without capping
                pass

            # Simple diagnostics to help debugging large ATEs
            try:
                q = np.quantile(e, [0.01, 0.1, 0.5, 0.9, 0.99])
                print(f"[IPW] propensity quantiles 1/10/50/90/99%: {q}")
                print(f"[IPW] max weight after capping: {w.max():.3e}, mean weight: {w.mean():.3e}")
            except Exception:
                pass

            ate_ipw = float(np.sum(w * Y * T) / np.sum(w * T) - np.sum(w * Y * (1 - T)) / np.sum(w * (1 - T)))
            # per-sample influence (standard IPW influence function)
            ite_ipw = (T * Y / e) - ((1 - T) * Y / (1 - e))

            # Doubly-robust (augmented IPW) using naive outcome model predictions
            ate_dr, ite_dr = None, None
            if doubly_robust:
                if len(y1_naive_list) == 0:
                    raise ValueError("Naive outcome predictions required for doubly robust estimator.")
                m1_all = np.concatenate(y1_naive_list)
                m0_all = np.concatenate(y0_naive_list)
                # ensure lengths align
                if len(m1_all) != len(T):
                    # In case predicted lists were empty or shapes mismatch
                    raise ValueError("Length mismatch between naive predictions and observed data for DR estimator.")

                ite_dr = m1_all - m0_all + (T * (Y - m1_all) / e) - ((1 - T) * (Y - m0_all) / (1 - e))
                ate_dr = float(np.mean(ite_dr))

        # --- True ITE and PEHE computations (if coeffs provided)
        if coeffs is not None:
            C_full = np.concatenate(all_C, axis=0)  # (N, C_total)
            ite_true, ate_true = compute_true_ite_ate(
                C_full=C_full,
                indices=self.indices,
                concept_idx=concept_idx,
                coeffs=coeffs,
                causal_concepts=causal_concepts,
                causal_concept_indices=causal_concept_indices,
            )

            pehe_naive = pehe(ite_naive, ite_true) if ite_naive is not None else None
            pehe_po = pehe(ite_po, ite_true) if ite_po is not None else None
            pehe_ipw = pehe(ite_ipw, ite_true) if ite_ipw is not None else None
            pehe_naive_pred = pehe(ite_naive_pred, ite_true) if ite_naive_pred is not None else None
            pehe_dr = pehe(ite_dr, ite_true) if doubly_robust else None
        else:
            ate_true = None
            pehe_naive = None
            pehe_po = None
            pehe_ipw = None
            pehe_naive_pred = None

        # --- diagnostics
        abs_diff_ipw = abs((ate_ipw if ate_ipw is not None else 0) - (ate_naive if ate_naive is not None else 0))
        ate_error_ipw = abs((ate_ipw if ate_ipw is not None else 0) - (ate_true if ate_true is not None else 0)) if ate_true is not None else None

        abs_diff_cbm = abs((ate_naive_pred if ate_naive_pred is not None else 0) - (ate_naive if ate_naive is not None else 0))
        ate_error_cbm = abs((ate_naive_pred if ate_naive_pred is not None else 0) - (ate_true if ate_true is not None else 0)) if ate_true is not None else None

        ate_error_naive = abs((ate_naive if ate_naive is not None else 0) - (ate_true if ate_true is not None else 0)) if ate_true is not None else None
        ate_error_pseudo_oracle = abs((ate_po if ate_po is not None else 0) - (ate_true if ate_true is not None else 0)) if ate_true is not None else None

        results = {
            "ate_naive_obs": ate_naive,        # original, uses attr (ground truth)
            "ate_cbm": ate_naive_pred,         # CBM: naive S-learner on predicted concepts
            "ate_pseudo_oracle": ate_po,
            "ate_ipw": ate_ipw,
            "ate_dr": ate_dr if doubly_robust else None,
            "ate_true": ate_true,
            "pehe_naive_obs": pehe_naive,
            "pehe_cbm": pehe_naive_pred,
            "pehe_pseudo_oracle": pehe_po,
            "pehe_ipw": pehe_ipw,
            "pehe_dr": pehe_dr if doubly_robust else None,
            "abs_diff_ipw": abs_diff_ipw,
            "ate_error_ipw": ate_error_ipw,
            "abs_diff_cbm": abs_diff_cbm,
            "ate_error_cbm": ate_error_cbm,
            "ate_error_naive": ate_error_naive,
            "ate_error_pseudo_oracle": ate_error_pseudo_oracle
        }

        # Confounding detection via bootstrap requires naive model
        if ite_naive is not None:
            n_boot = n_boot
            rng = np.random.default_rng(seed)
            n = len(T)

            ate_boot = np.empty(n_boot)
            ate_naive_boot = np.empty(n_boot)
            for b in range(n_boot):
                idxs = rng.choice(n, size=n, replace=True)
                T_b = T[idxs]
                Y_b = Y[idxs]
                e_b = e[idxs]
                # recompute stabilized weights on bootstrap sample
                p_b = T_b.mean()
                # clip to avoid extreme ratios
                e_b = np.clip(e_b, 1e-6, 1 - 1e-6)
                w_b = T_b * p_b / e_b + (1 - T_b) * (1 - p_b) / (1 - e_b)
                num1 = np.sum(w_b * Y_b * T_b)
                den1 = np.sum(w_b * T_b)
                num0 = np.sum(w_b * Y_b * (1 - T_b))
                den0 = np.sum(w_b * (1 - T_b))
                ate_boot[b] = (num1 / den1) - (num0 / den0)
                ate_naive_boot[b] = ite_naive[idxs].mean()

            confounded = detect_confounding(ate_boot, ate_naive_boot)
        else:
            raise ValueError("Naive model is required for confounding detection.")

        results["confounded_flag_ipw"] = int(confounded)

        return results