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
import math
from .utils import detect_confounding, STEFunction, pehe, compute_true_ite_ate, DINOV2_EMBED_DIM
from .ipw import ImageEncoder
from .image_modules import CelebAEncoder as CelebAImageEncoder
from sklearn.utils import resample


class ResidualCBM(pl.LightningModule):
    def __init__(self, concept_predictor, feat_dim, residual_dim, num_concepts, indices, lr=1e-3, frozen=True, kl_r=False, channels: int = 1, use_dinov2_embeddings: bool = False, dinov2_feat_dim: int = DINOV2_EMBED_DIM):
        super().__init__()
        self.lr = lr
        self.indices = indices
        self.num_concepts = num_concepts
        self.kl_r = kl_r
        self.channels = channels
        self.use_dinov2_embeddings = use_dinov2_embeddings

        # pre-trained encoder
        self.concept_predictor = concept_predictor
        if frozen:
            self.concept_predictor.eval()
            for param in self.concept_predictor.parameters():
                    param.requires_grad = False
        # residual feature extractor: supports images (1 or 3 channels) or DINOv2 embeddings
        if use_dinov2_embeddings:
            in_feat = dinov2_feat_dim
            self.r_cnn = nn.Sequential(
                nn.Identity(),
                nn.Linear(in_feat, residual_dim)
            )
        else:
            if channels == 3:
                base_encoder = CelebAImageEncoder(in_channels=3, feat_dim=feat_dim)
            else:
                base_encoder = ImageEncoder(in_channels=channels, feat_dim=feat_dim)
            self.r_cnn = nn.Sequential(
                base_encoder,
                nn.Linear(feat_dim, residual_dim)
            )

        self.y_predictor = nn.Sequential(
            nn.Linear(num_concepts + residual_dim, 1),
            nn.Sigmoid()
        )

        # metrics
        self.accuracy = Accuracy(task='binary')

    def forward(self, x, r_hard=False):
        """
        x: (B, C, H, W)
        returns: outcome probs (B,)
        """
        concept_probs = []
        for i in range(self.num_concepts):
            c = self.concept_predictor(x, i)
            concept_probs.append(STEFunction.apply(c, 0.5))
        
        c = torch.stack(concept_probs).squeeze(-1).T

        r_logits = self.r_cnn(x)
        r_probs = torch.sigmoid(r_logits)
        # r = self.sample_gumbel_max(r_probs, hard=r_hard, tau=self.tau_weight())
        y = self.y_predictor(torch.cat([c, r_logits], dim=1))
        return y, r_logits, r_probs

    def tau_weight(self):
        epoch = float(self.current_epoch)
        tau = 1.0 * math.exp(-0.05 * epoch)
        return max(tau, 0.1)
    
    @staticmethod
    def sample_gumbel_max(q_probs, tau=0.5, hard=False, eps=1e-6):
        """
        Differentiable Bernoulli sampling from probabilities using Gumbel-Sigmoid.

        Args:
            q_probs: Tensor, probabilities in (0,1), shape [B, k]
            tau: temperature (lower -> more discrete)
            hard: if True, return hard {0,1} sample with straight-through estimator
            eps: numerical stability
        """
        # clamp for numerical stability
        q = torch.clamp(q_probs, eps, 1 - eps)

        # convert to logits
        logits = torch.log(q) - torch.log(1 - q)

        # gumbel noise
        u = torch.rand_like(q)
        g = -torch.log(-torch.log(u + eps) + eps)

        # relaxed sample
        z = torch.sigmoid((logits + g) / tau)

        if hard:
            z = STEFunction.apply(z, 0.5)

        return z

    def kl_bernoulli(self, q_probs: torch.Tensor, p: float = 0.5, eps: float = 1e-6):
        """
        KL(q || p) between Bernoulli distributions.

        Args:
            q_probs: posterior probs q(z=1|x), shape [B, k]
            p: prior prob (default 0.5, i.e. uniform Bernoulli)
            eps: small constant to avoid log(0)

        Returns:
            KL divergence (scalar)
        """
        q = torch.clamp(q_probs, eps, 1 - eps)
        p = torch.clamp(torch.tensor(p, device=q.device, dtype=q.dtype), eps, 1 - eps)

        kl = q * (torch.log(q) - torch.log(p)) + (1 - q) * (torch.log(1 - q) - torch.log(1 - p))
        return kl.sum() # sum over latent dims and batch


    def training_step(self, batch, batch_idx):
        x, attr = batch
        y = attr[:, self.indices['task']]

        y_hat, r, r_probs = self.forward(x)
        loss = F.binary_cross_entropy(y_hat, y.unsqueeze(1))

        if self.kl_r:
            loss += self.kl_bernoulli(r_probs)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, attr = batch
        y = attr[:, self.indices['task']]

        y_hat, r, r_probs = self.forward(x)
        loss = F.binary_cross_entropy(y_hat, y.unsqueeze(1))

        if self.kl_r:
            loss += self.kl_bernoulli(r_probs)

        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, attr = batch
        y = attr[:, self.indices['task']]

        y_hat, r, r_probs = self.forward(x)
        
        self.log(f"test_acc", self.accuracy(y_hat, y.unsqueeze(1).int()), prog_bar=True)
        return

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def compare_ates(
        self,
        dataloader,
        concept_idx,
        naive=None,
        pseudo_oracle=None,
        prop_model=None,
        coeffs=None,
        device="cuda",
        n_bootstrap=1000,
        alpha=0.05,
        causal_concepts=None,
        causal_concept_indices=None,
        seed=42
    ):
        """
        Compare ATEs and compute PEHE, error metrics, and confounded flag.
        """

        self.eval().to(device)

        y1_naive_list, y0_naive_list = [], []
        y1_po_list, y0_po_list = [], []
        obs_t_list, obs_y_list, prop_list = [], [], []
        all_C = []

        for x, attr in dataloader:
            x = x.to(device)
            obs_t = attr[:, self.indices['concepts']][:, concept_idx].cpu().numpy().astype(float)
            obs_y = attr[:, self.indices['task']].cpu().numpy().astype(float)

            obs_t_list.append(obs_t)
            obs_y_list.append(obs_y)

            # observed features
            X_concepts = attr[:, self.indices['concepts']].cpu().numpy()
            extra_indices = [s for s in self.indices['shortcut'] if s not in self.indices['concepts']]
            X_all = np.concatenate([X_concepts,
                                    attr[:, extra_indices].cpu().numpy()], axis=1)

            # --- Naive S-learner
            if naive is not None:
                X1 = X_concepts.copy(); X1[:, concept_idx] = 1
                X0 = X_concepts.copy(); X0[:, concept_idx] = 0
                y1_naive_list.append(naive.predict_proba(X1)[:, 1])
                y0_naive_list.append(naive.predict_proba(X0)[:, 1])

            # --- Pseudo-oracle S-learner
            if pseudo_oracle is not None:
                X1_all = X_all.copy(); X1_all[:, concept_idx] = 1
                X0_all = X_all.copy(); X0_all[:, concept_idx] = 0
                y1_po_list.append(pseudo_oracle.predict_proba(X1_all)[:, 1])
                y0_po_list.append(pseudo_oracle.predict_proba(X0_all)[:, 1])

            # --- Propensity scores from residuals
            if prop_model is not None:
                # r_feats = self.r_cnn(x).cpu().numpy()
                # ps = prop_model.predict_proba(r_feats)[:, 1]
                _, r, _ = self.forward(x)
                ps = prop_model.predict_proba(r.cpu().numpy())[:, 1]
                prop_list.append(ps)

            # collect ground-truth concepts for true ATE/PEHE
            if coeffs is not None:
                # c_full_batch = torch.cat([attr[:, self.indices['shortcut']], attr[:, self.indices['concepts']]], dim=1)
                all_C.append(X_all)

        # Concatenate
        T = np.concatenate(obs_t_list)
        Y = np.concatenate(obs_y_list)

        # --- Naive
        ite_naive = np.concatenate(y1_naive_list) - np.concatenate(y0_naive_list) if y1_naive_list else None
        ate_naive = float(ite_naive.mean()) if ite_naive is not None else None

        # --- Pseudo-oracle
        ite_po = np.concatenate(y1_po_list) - np.concatenate(y0_po_list) if y1_po_list else None
        ate_po = float(ite_po.mean()) if ite_po is not None else None

        # --- IPW (stabilized)
        ate_ipw, ite_ipw = None, None
        if prop_list:
            e = np.concatenate(prop_list)
            eps = 1e-6
            e = np.clip(e, eps, 1 - eps)
            p = T.mean()
            w = T * p / e + (1 - T) * (1 - p) / (1 - e)
            ate_ipw = float(
                np.sum(w * Y * T) / np.sum(w * T) -
                np.sum(w * Y * (1 - T)) / np.sum(w * (1 - T))
            )
            ite_ipw = w * Y * (T / e - (1 - T) / (1 - e))

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
        else:
            ate_true = None
            pehe_naive = None
            pehe_po = None
            pehe_ipw = None

        abs_diff = abs(ate_ipw - ate_naive) if ate_naive is not None else None

        results = {
            "ate_naive": ate_naive,
            "ate_pseudo_oracle": ate_po,
            "ate": ate_ipw,
            "ate_true": ate_true,
            "pehe_naive": pehe_naive,
            "pehe_pseudo_oracle": pehe_po,
            "pehe": pehe_ipw,
            "ate_error": abs(float(ate_ipw) - float(ate_true)) if ate_true is not None else None,
            "abs_diff": abs_diff
        }

        # Bootstrap only if naive is available
        if ite_naive is not None:
            n_boot = 1000
            rng = np.random.default_rng(seed)
            n = len(ite_ipw)

            ate_boot = np.empty(n_boot)
            ate_naive_boot = np.empty(n_boot)
            for b in range(n_boot):
                idxs = rng.choice(n, size=n, replace=True)
                ate_boot[b] = ite_ipw[idxs].mean()
                ate_naive_boot[b] = ite_naive[idxs].mean()

            confounded = detect_confounding(ate_boot, ate_naive_boot)

            results["confounded_flag"] = int(confounded)
        else:
            raise ValueError("Naive model is required for confounding detection.")

        return results
