from typing import Dict
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, Accuracy
import matplotlib.pyplot as plt
import os
from .mutual_information import MutualInformationLoss
from .utils import STEFunction, detect_confounding, pehe, compute_true_ite_ate


class ImageEncoder(nn.Module):
    def __init__(self, in_channels=1, feat_dim=128, resolution=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, resolution, 3, stride=2, padding=1),  # 32x32 -> 16x16
            # nn.BatchNorm2d(resolution),
            nn.LeakyReLU(),
            
            nn.Conv2d(resolution, resolution*2, 3, stride=2, padding=1),  # 16x16 -> 8x8
            # nn.BatchNorm2d(resolution*2),
            nn.LeakyReLU(),
            
            nn.Conv2d(resolution*2, resolution*4, 3, stride=2, padding=1),  # 8x8 -> 4x4
            # nn.BatchNorm2d(resolution*4),
            nn.LeakyReLU(),
            
            nn.Conv2d(resolution*4, resolution*4, 3, stride=1, padding=1),  # 4x4 -> 4x4
            # nn.BatchNorm2d(resolution*4),
            nn.LeakyReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(resolution*4*4*4, feat_dim)  # 512 -> 128

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ImageDecoder(nn.Module):
    def __init__(self, out_channels=1, feat_dim=128, resolution=32):
        super().__init__()
        self.fc = nn.Linear(feat_dim, resolution*4*4*4)  # match encoder output
        self.unflatten = nn.Unflatten(1, (resolution*4, 4, 4))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(resolution*4, resolution*4, 3, stride=1, padding=1),  # 4x4
            # nn.BatchNorm2d(resolution*4),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(resolution*4, resolution*2, 3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            # nn.BatchNorm2d(resolution*2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(resolution*2, resolution, 3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            # nn.BatchNorm2d(resolution),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(resolution, out_channels, 3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        x = self.deconv(x)
        return x


class CaCE(pl.LightningModule):
    def __init__(
        self,
        num_concepts: int,
        feat_dim: int = 128,
        shared_latent_dim: int = 0,
        latent_per_concept: int = 1,
        style_latent_dim: int = 16,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        kl_anneal_start: int = 0,
        kl_anneal_end: int = 50,
        tau_anneal_start: int = 1.0,
        tau_anneal_min: float = 0.1,
        tau_anneal_decay: int = 0.05,
        use_aux: bool = True,
        aux_weight_c: float = 1.0,
        aux_weight_y: float = 1.0,
        indices=None,
        channels=1,
    ):
        """
        Unconditional-prior CaCE LightningModule.

        Args:
            num_concepts: k
            latent_per_concept: L (per-concept latent dimensionality)
            hidden_dim: MLP hidden dims
            lr: learning rate
            kl_anneal_start, kl_anneal_end: epochs for linear KL annealing
            use_aux: whether to include auxiliary q(c|x) and q(y|x,c) terms (recommended)
            aux_weight_c/y: weights for auxiliary log-likelihoods
        """
        super().__init__()
        self.save_hyperparameters(ignore=["indices"])
        self.channels = channels

        self.k = num_concepts
        self.L = latent_per_concept
        self.shared_latent_dim = shared_latent_dim

        self.z_c_dim = self.k * self.L if self.L > 0 else 0
        self.z_c_dim += self.shared_latent_dim

        self.style_latent_dim = style_latent_dim

        self.latent_dim = self.z_c_dim + self.style_latent_dim

        self.lr = lr
        self.hidden_dim = hidden_dim
        self.indices = indices or {}

        self.feat_dim = feat_dim  # encoder feature dim

        self.img_encoder = ImageEncoder(in_channels=1, feat_dim=self.feat_dim)
        self.img_decoder = ImageDecoder(out_channels=1, feat_dim=self.feat_dim)


        # no conditional prior: p(z) = N(0, I)

        # ---------- posterior q_phi(Z | X, C, Y): outputs mu_q, logvar_q ----------
        # input: features || C (k) || y (1)
        self.post_net = nn.Sequential(
            nn.Linear(self.feat_dim + self.k + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Bernoulli for z_c (concepts), Gaussian for z_s (style)

        if self.z_c_dim > 0:
            self.zc_logits = nn.Linear(hidden_dim, self.z_c_dim)        # Bernoulli probs
        else:
            self.zc_logits = None
            self.shared_latent_dim = self.style_latent_dim   # use style latent as shared latent if no per-concept latents
            self.z_c_dim = self.style_latent_dim
            self.latent_dim = self.style_latent_dim
        
        if self.style_latent_dim > 0:
            self.mu_s = nn.Linear(hidden_dim, self.style_latent_dim)  # style Gaussian
            self.logvar_s = nn.Linear(hidden_dim, style_latent_dim)
        else:
            self.mu_s = None
            self.logvar_s = None

        # ---------------- image decoder p(X | Z, C) (feature reconstruction) ----------------
        # We decode to encoder feature space (feat_dim) rather than raw pixels (recommended).
        self.x_decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.feat_dim)   # reconstruct x_feat
        )

        # ---------- optional auxiliary networks q(C | X), q(Y | X, C) ----------
        self.use_aux = use_aux
        self.aux_weight_c = aux_weight_c
        self.aux_weight_y = aux_weight_y
        if self.use_aux:
            self.aux_qc = nn.Sequential(
                nn.Linear(self.feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.k)  # logits for k Bernoulli
            )
            self.aux_qy = nn.Sequential(
                nn.Linear(self.feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)  # logit for y
            )

        # KL annealing schedule
        self.kl_anneal_start = kl_anneal_start
        self.kl_anneal_end = kl_anneal_end

        # temperature annealing schedule for Gumbel-Sigmoid
        self.tau_anneal_start = tau_anneal_start
        self.tau_anneal_min = tau_anneal_min
        self.tau_anneal_decay = tau_anneal_decay

        self.f1_score = BinaryF1Score()
        self.accuracy = Accuracy(task='binary')

        self.likelihood_log_sigma = nn.Parameter(torch.tensor(0.0))

    # --------------------- helpers ---------------------
    def _post_params(self, x_feat: torch.Tensor, c: torch.Tensor, y: torch.Tensor):
        if y.dim() == 1:
            y = y.unsqueeze(1)
        h_in = torch.cat([x_feat, c, y], dim=1)
        h = self.post_net(h_in)

        if self.zc_logits is not None:
            zc_logits = self.zc_logits(h)           # [B, k * L] or [B, shared_latent_dim]
            zc_probs = torch.sigmoid(zc_logits)     # [B, k * L or [B, shared_latent_dim]
        else:
            zc_probs = None
        
        if self.style_latent_dim == 0:
            return zc_probs, None, None
        else:
            mu_s = self.mu_s(h)                     # [B, style_latent_dim]
            logvar_s = self.logvar_s(h)             # [B, style_latent_dim]
            return zc_probs, mu_s, logvar_s

    def _split_latents(self, z: torch.Tensor):
        """
        z: [B, k * L]
        Returns:
        z_chunks: list of [B,L] (concept-specific latents)
        """
        B = z.size(0)
        z_concepts = z.view(B, self.k, self.L)
        z_chunks = [z_concepts[:, i, :] for i in range(self.k)]
        return z_chunks

    def _combine_latents(self, z_c, z_s):
        if z_c is not None and z_s is not None:
            z_all = torch.cat([z_c, z_s], dim=1)
        elif z_c is not None:
            z_all = z_c
        elif z_s is not None:
            z_all = z_s
        else:
            raise ValueError("Both z_c and z_s are None!")
        return z_all

    @staticmethod
    def sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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

    def kl_standard_normal(self, mu_q, logvar_q):
        # KL( N(mu_q, var_q) || N(0, I) )
        var_q = torch.exp(logvar_q).clamp(min=1e-8)

        term = mu_q.pow(2) + var_q - logvar_q - 1.0
        kl = 0.5 * torch.sum(term) # sum over latent dims and batch
        return kl

    def kl_weight(self):
        epoch = float(self.current_epoch)
        if self.kl_anneal_end <= self.kl_anneal_start:
            return 1.0
        if epoch <= self.kl_anneal_start:
            return 0.0
        if epoch >= self.kl_anneal_end:
            return 1.0
        p = (epoch - self.kl_anneal_start) / (self.kl_anneal_end - self.kl_anneal_start)
        return float(p)

    def tau_weight(self):
        epoch = float(self.current_epoch)
        tau = self.tau_anneal_start * math.exp(-self.tau_anneal_decay * epoch)
        return max(tau, self.hparams.tau_anneal_min)

    def gaussian_nll(self, x_recon, x):
        """
        Negative log-likelihood under Gaussian with learnable sigma.
        """
        logvar = 2 * self.likelihood_log_sigma  # log σ²
        recon_loss = 0.5 * ((x - x_recon) ** 2 / torch.exp(logvar) + logvar + math.log(2 * math.pi))
        return recon_loss.sum()  # sum over pixels and batch

    # --------------------- forward / generative ---------------------
    def forward(self, x, c, y):
        # encode x -> features
        x_feat = self.img_encoder(x)

        # posterior params q(Z|X,C,Y)
        zc_probs, mu_q_s, logvar_q_s = self._post_params(x_feat, c, y)

        # sample z_c (binary) and z_s (Gaussian)
        if zc_probs is not None:
            z_c = self.sample_gumbel_max(zc_probs, tau=self.tau_weight(), hard=False)
        else:
            z_c = None
        if self.style_latent_dim > 0:
            z_s = self.sample_gaussian(mu_q_s, logvar_q_s)
        else:
            z_s = None

        # X reconstruction uses *all latents*
        z_all = self._combine_latents(z_c, z_s)
        x_recon_feat = self.x_decoder(torch.cat([z_all, c], dim=1))
        # Decode to image space (overridable for conditional decoders)
        x_recon = self.decode_image(
            x_recon_feat,
            decoder_input=torch.cat([z_all, c], dim=1),
            z_all=z_all,
            c=c,
        )

        return {
            "zc_probs": zc_probs, "mu_s": mu_q_s, "logvar_s": logvar_q_s,
            "z_c": z_c, "z_s": z_s,
            "x_recon": x_recon, "x_feat": x_feat
        }
                
    # --------------------- loss computation ---------------------
    def compute_elbo(self, batch: torch.Tensor, stage: str = "train") -> Dict[str, torch.Tensor]:
        x, attr = batch
        y = attr[:, self.indices['task']]
        c = attr[:, self.indices['concepts']]
        if y.dim() == 1:
            y = y.unsqueeze(1)
        out = self.forward(x, c, y)

        zc_probs, mu_s, logvar_s = out["zc_probs"], out["mu_s"], out["logvar_s"]

        B = x.size(0)

        # compute KLs
        # KL for z_c (Bernoulli)
        if zc_probs is not None:
            kl_c = self.kl_bernoulli(zc_probs, p=0.5) / B
        else:
            kl_c = torch.tensor(0.0, device=self.device)

        if self.style_latent_dim > 0:
            # KL for z_s (Gaussian); priors (standard normal): zeros for means and logvar
            kl_s = self.kl_standard_normal(mu_s, logvar_s) / B
        else:
            kl_s = torch.tensor(0.0, device=self.device)

        kl_w = self.kl_weight()
        kl_total = kl_w * (kl_s + kl_c)

        # # --- covariance penalty between z_c_flat and z_s ---
        # z_c_flat = out["z_c"]
        # z_s_centered = (mu_s - mu_s.mean(0, keepdim=True))
        # z_c_centered = (z_c_flat - z_c_flat.mean(0, keepdim=True))
        # # empirical cross-covariance matrix [k*Lc, Lr]
        # cov = (z_c_centered.t() @ z_s_centered) / (B - 1)
        # cov_penalty = (cov ** 2).sum() * 1.0 # TODO add weight if needed

        # image reconstruction loss
        x_recon = out["x_recon"]
        loss_X = self.gaussian_nll(x_recon, x) / B
        # loss_X = torch.tensor(0.0, device=self.device)

        total_loss = loss_X + kl_total

        # optional auxiliary losses (as Louizos): q(C|X) and q(Y|X,C)
        aux_loss = torch.tensor(0.0, device=self.device)
        aux_c_loss = torch.tensor(0.0, device=self.device)
        aux_y_loss = torch.tensor(0.0, device=self.device)
        if self.use_aux:
            # q(C|X) logits
            qc_logits = self.aux_qc(out["x_feat"])  # [B, k]
            aux_c_loss = (F.binary_cross_entropy_with_logits(qc_logits, c, reduction='sum') / B) * self.aux_weight_c
            # q(Y | X)
            qy_in = out["x_feat"]
            qy_logits = self.aux_qy(qy_in)
            aux_y_loss = (F.binary_cross_entropy_with_logits(qy_logits, y, reduction='sum') / B) * self.aux_weight_y
            aux_loss = aux_c_loss + aux_y_loss
            total_loss = total_loss + aux_loss

        logs = {
            f"{stage}_loss": total_loss,
            f"{stage}_loss_X": loss_X,
            f"{stage}_kl": kl_total,
            f"{stage}_kl_w": torch.tensor(kl_w, device=self.device),
            f"{stage}_aux_c": aux_c_loss,
            f"{stage}_aux_y": aux_y_loss,
        }

        if stage == 'test':
            y_aux_acc = self.accuracy(qy_logits, y.long().view(-1, 1)) if self.use_aux else torch.tensor(0.0)
            y_aux_f1 = self.f1_score(qy_logits, y.long().view(-1, 1)) if self.use_aux else torch.tensor(0.0)
            
            c_aux_acc = torch.stack([
                self.accuracy(c_hat.view(-1,1), c.view(-1,1).long())
                for c, c_hat in zip(c.T, qc_logits.T)
            ]).mean() if self.use_aux else torch.tensor(0.0)
            c_aux_f1 = torch.stack([
                self.f1_score(c_hat.view(-1,1), c.view(-1,1).long())
                for c, c_hat in zip(c.T, qc_logits.T)
            ]).mean() if self.use_aux else torch.tensor(0.0)
            logs = {
                f"{stage} aux y accuracy": y_aux_acc,
                f"{stage} aux y f1": y_aux_f1,
                f"{stage} aux c accuracy": c_aux_acc,
                f"{stage} aux c f1": c_aux_f1,
            }
        return {"loss": total_loss, "logs": logs}
    
    # --------------------- inference ---------------------
    def infer_latents(self, x, c=None, y=None, use_aux=True, binarize=True, x_embedding=False):
        """
        Run inference pipeline:
        - If use_aux=True, predict c_hat = q(C|X), y_hat = q(Y|X, c_hat)
        - Else use provided c, y
        - Compute posterior params q(z_c|X,C,Y), q(z_s|X,C,Y)
        - Sample z_c (Bernoulli) and z_s (Gaussian)
        Returns dict with raw + (optional) binarized outputs
        """
        x_feat = x if x_embedding else self.img_encoder(x)  # [B, feat_dim]

        if use_aux and getattr(self, "use_aux", False):
            qc_logits = self.aux_qc(x_feat)
            c_hat_prob = torch.sigmoid(qc_logits)
            c_hat = torch.bernoulli(c_hat_prob) if binarize else c_hat_prob

            qy_logits = self.aux_qy(x_feat)
            y_hat_prob = torch.sigmoid(qy_logits)
            y_hat = torch.bernoulli(y_hat_prob) if binarize else y_hat_prob
        else:
            print("WARNING: using ground-truth c,y for inference!")
            c_hat = c.float()
            y_hat = y.float()
            if y_hat.dim() == 1:
                y_hat = y_hat.unsqueeze(1)

        # posterior params
        zc_probs, mu_s, logvar_s = self._post_params(x_feat, c_hat, y_hat)

        # sample z_c (binary) and z_s (Gaussian)
        z_c = torch.bernoulli(zc_probs) if zc_probs is not None else None
        
        z_s = self.sample_gaussian(mu_s, logvar_s) if self.style_latent_dim > 0 else None

        z_chunks = self._split_latents(z_c) if self.L > 0 else None

        out = {
            "c_hat": c_hat,
            "c_hat_prob": c_hat_prob,
            "y_hat": y_hat,
            "y_hat_prob": y_hat_prob,
            "zc_probs": zc_probs,
            "z_c": z_c,
            "mu_s": mu_s,
            "logvar_s": logvar_s,
            "z_s": z_s,
            "z_chunks": z_chunks
        }
        return out

    # --------------------- Lightning hooks ---------------------
    def training_step(self, batch, batch_idx):
        out = self.compute_elbo(batch, stage="train")
        self.log_dict(out["logs"], prog_bar=False, on_step=False, on_epoch=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.compute_elbo(batch, stage="val")
        self.log_dict(out["logs"], prog_bar=True, on_step=False, on_epoch=True)
        return out["loss"]

    def test_step(self, batch, batch_idx):
        out = self.compute_elbo(batch, stage="test")
        self.log_dict(out["logs"], prog_bar=False, on_step=False, on_epoch=True)
        return out["loss"]

    def predict_step(self, batch, batch_idx):
        x, attr = batch
        y = attr[:, self.indices['task']]
        c = attr[:, self.indices['concepts']]
        shortcuts = attr[:, self.indices['shortcut']]
        predict_dict = self.infer_latents(x, c, y, use_aux=True, binarize=True)
        predict_dict.update({
            "y": y,
            "c": c,
            "shortcuts": shortcuts
        })
        return predict_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # # --------------------- ATE estimation helper ---------------------
    @torch.no_grad()
    def compare_ates(
        self,
        dataloader,
        concept_idx,
        num_samples=100,
        device="cuda",
        coeffs=None,  # (C,) tensor for true ATE
        naive=None,
        pseudo_oracle=None,
        causal_concepts=None,
        causal_concept_indices=None,
        seed=42
    ):
        self.eval()
        self.to(device)

        # ---- containers for aggregated
        y_c1, y_c0 = [], []

        # ---- containers for per-sample ITEs
        per_sample_y1_adj = []   # per-sample mean y1 (adjusted)
        per_sample_y0_adj = []
        per_sample_y1_naive = []
        per_sample_y0_naive = []
        per_sample_y1_po = []
        per_sample_y0_po = []

        if coeffs is not None:
            all_C = []

        for batch in dataloader:
            x, attr = batch
            x = x.to(device)
            c = attr[:, self.indices['concepts']].to(device)

            x_feat = self.img_encoder(x)
            out_fixed = self.infer_latents(x_feat, use_aux=True, binarize=True, x_embedding=True)
            c_hat_fixed = out_fixed["c_hat"].clone().to(device)                      # (B, C_concepts)
            y_hat_fixed = out_fixed["y_hat"].clone().to(device)                      # (B,)

            X_concepts = c_hat_fixed.cpu().numpy()
            extra_indices = [s for s in self.indices['shortcut'] if s not in self.indices['concepts']]
            X_all = np.concatenate([c.cpu().numpy(),
                                    attr[:, extra_indices].cpu().numpy()], axis=1)

            if naive is not None:
                X1 = X_concepts.copy(); X1[:, concept_idx] = 1
                X0 = X_concepts.copy(); X0[:, concept_idx] = 0
                y1_naive = naive.predict_proba(X1)[:, 1]
                y0_naive = naive.predict_proba(X0)[:, 1]
                per_sample_y1_naive.append(y1_naive)
                per_sample_y0_naive.append(y0_naive)

            if pseudo_oracle is not None:
                X1_all = X_all.copy(); X1_all[:, concept_idx] = 1
                X0_all = X_all.copy(); X0_all[:, concept_idx] = 0
                y1_po = pseudo_oracle.predict_proba(X1_all)[:, 1]
                y0_po = pseudo_oracle.predict_proba(X0_all)[:, 1]
                per_sample_y1_po.append(y1_po)
                per_sample_y0_po.append(y0_po)

            # naive ATE group-level by conditioning (using fixed c/y)
            if (c_hat_fixed[:, concept_idx] == 1).any():
                y_c1.append(y_hat_fixed[c_hat_fixed[:, concept_idx] == 1].mean().item())
            if (c_hat_fixed[:, concept_idx] == 0).any():
                y_c0.append(y_hat_fixed[c_hat_fixed[:, concept_idx] == 0].mean().item())

            # collect ground-truth concepts for true ATE/PEHE
            if coeffs is not None:
                # c_full_batch = torch.cat([attr[:, self.indices['shortcut']], attr[:, self.indices['concepts']]], dim=1)
                all_C.append(X_all)

            # Monte Carlo draws per batch (for adjusted estimator)
            y1_draws = []  # (S, B)
            y0_draws = []
            for _ in range(num_samples):
                out = self.infer_latents(x_feat, use_aux=True, binarize=True, x_embedding=True)

                c_hat = out["c_hat"].clone().to(device)
                y_hat = out["y_hat"].clone().to(device)

                zc_probs, mu_s, logvar_s = self._post_params(x_feat, c_hat, y_hat)
                z_s = self.sample_gaussian(mu_s, logvar_s) if (mu_s is not None and logvar_s is not None) else None
                z_c = torch.bernoulli(zc_probs) if zc_probs is not None else None
                z_all = self._combine_latents(z_c, z_s)

                c1 = c_hat.clone(); c1[:, concept_idx] = 1.0
                c0 = c_hat.clone(); c0[:, concept_idx] = 0.0

                x_cf_feat1 = self.x_decoder(torch.cat([z_all, c1], dim=1))
                x_cf_feat0 = self.x_decoder(torch.cat([z_all, c0], dim=1))

                x_cf_1 = self.decode_image(x_cf_feat1, decoder_input=torch.cat([z_all, c1], dim=1), z_all=z_all, c=c1).clamp(0, 1)
                x_cf_0 = self.decode_image(x_cf_feat0, decoder_input=torch.cat([z_all, c0], dim=1), z_all=z_all, c=c0).clamp(0, 1)
                x_cf_feat1 = self.img_encoder(x_cf_1)
                x_cf_feat0 = self.img_encoder(x_cf_0)
                y1_logit = self.aux_qy(x_cf_feat1)
                y0_logit = self.aux_qy(x_cf_feat0)

                y1_draws.append(torch.sigmoid(y1_logit).cpu().numpy())
                y0_draws.append(torch.sigmoid(y0_logit).cpu().numpy())

            # After MC draws: mean over S per sample in batch
            y1_mean_per_sample = np.stack(y1_draws, axis=0).mean(axis=0)
            y0_mean_per_sample = np.stack(y0_draws, axis=0).mean(axis=0)
            per_sample_y1_adj.append(y1_mean_per_sample)
            per_sample_y0_adj.append(y0_mean_per_sample)

        # --- aggregate ATE as before
        y1_all = np.concatenate(per_sample_y1_adj, axis=0)
        y0_all = np.concatenate(per_sample_y0_adj, axis=0)
        ate = y1_all.mean() - y0_all.mean()
        ite = y1_all - y0_all

        ite_naive, ite_po, ate_naive, ate_po = None, None, None, None
        
        if per_sample_y1_naive:
            y1_naive_all = np.concatenate(per_sample_y1_naive, axis=0)
            y0_naive_all = np.concatenate(per_sample_y0_naive, axis=0)
            ite_naive = y1_naive_all - y0_naive_all
            ate_naive = ite_naive.mean()

        if per_sample_y1_po:
            y1_po_all = np.concatenate(per_sample_y1_po, axis=0)
            y0_po_all = np.concatenate(per_sample_y0_po, axis=0)
            ite_po = y1_po_all - y0_po_all
            ate_po = ite_po.mean()

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

            pehe_adj = pehe(ite, ite_true)
            pehe_naive = pehe(ite_naive, ite_true) if ite_naive is not None else None
            pehe_po = pehe(ite_po, ite_true) if ite_po is not None else None
        else:
            # if no ground-truth coefficients, use pseudo-oracle as proxy for true ITE/ATE when available
            if per_sample_y1_po:
                ite_true = ite_po
                ate_true = ite_true.mean()
                pehe_adj = pehe(ite, ite_true)
                pehe_naive = pehe(ite_naive, ite_true) if ite_naive is not None else None
                pehe_po = pehe(ite_po, ite_true) if ite_po is not None else None
            else:
                ate_true = None
                pehe_adj = None
                pehe_naive = None
                pehe_po = None

        ate_cond = (np.mean(y_c1) - np.mean(y_c0)) if (len(y_c1) > 0 and len(y_c0) > 0) else None
        abs_diff = abs(ate - ate_naive) if ate_naive is not None else None

        # --- package results
        results = {
            "ate": float(ate),
            "ate_naive": float(ate_naive) if ate_naive is not None else None,
            "abs_diff": float(abs_diff) if abs_diff is not None else None,
            "ate_pseudo_oracle": float(ate_po) if ate_po is not None else None,
            "ate_cond": float(ate_cond) if ate_cond is not None else None,
            "ate_true": float(ate_true) if ate_true is not None else None,
            "ate_error": abs(float(ate) - float(ate_true)) if ate_true is not None else None,
            # PEHEs:
            "pehe_adjusted": pehe_adj,
            "pehe_naive": pehe_naive,
            "pehe_pseudo_oracle": pehe_po,
        }

        # Confounding detection via bootstrap on ITEs requires naive model
        if ite_naive is not None:
            n_boot = 1000
            rng = np.random.default_rng(seed)
            n = len(ite)

            ate_boot = np.empty(n_boot)
            ate_naive_boot = np.empty(n_boot)
            for b in range(n_boot):
                idxs = rng.choice(n, size=n, replace=True)
                ate_boot[b] = ite[idxs].mean()
                ate_naive_boot[b] = ite_naive[idxs].mean()

            confounded = detect_confounding(ate_boot, ate_naive_boot)
            results["confounded_flag"] = int(confounded)
        else:
            raise ValueError("Naive model is required for confounding detection.")

        return results
    
    # # --------------------- Counterfactual Visualization ---------------------
    @torch.no_grad()
    def create_counterfactuals(
        self,
        dataloader,
        out_dir: str,
        concept_indices=None,
        num_examples: int = 10,
        concept_names=None,
        device: str = "cuda",
    ):
        """
        Generate visual counterfactuals:
        Row 1: factual image + counterfactuals by flipping concepts
        Row 2: reconstruction + counterfactuals by intervening on each z_c

        Args:
            dataloader: DataLoader providing (x, attr)
            out_dir: directory where visualizations are saved
            concept_indices: list of concept indices to intervene on (default: all)
            num_examples: how many examples to visualize
            concept_names: list of names for concepts (default: index numbers)
            device: 'cuda' or 'cpu'
        """
        self.eval()
        self.to(device)

        if concept_indices is None:
            concept_indices = list(range(self.k))
        if concept_names is None:
            concept_names = [f"c{i}" for i in range(self.k)]

        example_count = 0

        for batch in dataloader:
            x, attr = batch
            x = x.to(device)

            out = self.infer_latents(x, use_aux=True, binarize=True)
            c_hat = out["c_hat"]
            z_c_all = out["z_c"]
            z_s_all = out["z_s"]


            for i in range(x.size(0)):
                if example_count >= num_examples:
                    return

                img = x[i : i + 1]
                c0 = c_hat[i : i + 1].clone()
                y0 = out["y_hat"][i : i + 1]

                # Latents
                z_c = z_c_all[i : i + 1] if z_c_all is not None else None
                z_s = z_s_all[i : i + 1] if z_s_all is not None else None
                z_all = self._combine_latents(z_c, z_s)

                # Reconstruction
                decoder_input = torch.cat([z_all, c0], dim=1)
                x_recon_feat = self.x_decoder(decoder_input)
                x_recon = self.decode_image(x_recon_feat, decoder_input, z_all, c0).clamp(0, 1).squeeze()
                x_recon = x_recon.cpu().numpy() if self.channels == 1 else x_recon.permute(1, 2, 0).cpu().numpy()

                # Make grid: 2 rows
                ncols = len(concept_indices) + 1
                fig, axes = plt.subplots(
                    2, ncols, figsize=(5 * ncols, 10)
                )

                # --- Row 1: factual + flip concepts ---
                img = img.squeeze().cpu().numpy() if self.channels == 1 else img.squeeze().permute(1, 2, 0).cpu().numpy()
                axes[0, 0].imshow(img, cmap="gray")
                axes[0, 0].set_title(
                    f"Factual\nc={c0[0].cpu().numpy().astype(int).tolist()}, y={y0.item():.2f}"
                )
                axes[0, 0].axis("off")

                for j, concept_idx in enumerate(concept_indices, start=1):
                    c_cf = c0.clone()
                    c_cf[:, concept_idx] = 1.0 - c_cf[:, concept_idx]

                    decoder_input_cf = torch.cat([z_all, c_cf], dim=1)
                    x_cf_feat = self.x_decoder(decoder_input_cf)
                    x_cf = self.decode_image(x_cf_feat, decoder_input_cf, z_all, c_cf).clamp(0, 1)

                    x_cf_feat = self.img_encoder(x_cf)
                    y_cf_logit = self.aux_qy(x_cf_feat)
                    y_cf = torch.sigmoid(y_cf_logit).item()

                    x_cf = x_cf.squeeze().cpu().numpy() if self.channels == 1 else x_cf.squeeze().permute(1, 2, 0).cpu().numpy()

                    axes[0, j].imshow(x_cf, cmap="gray")
                    axes[0, j].set_title(
                        f"Flip {concept_names[concept_idx]}\n"
                        f"c={c_cf[0].cpu().numpy().astype(int).tolist()}, y={y_cf:.2f}"
                    )
                    axes[0, j].axis("off")

                # --- Row 2: reconstruction + intervene on each z_c ---
                axes[1, 0].imshow(x_recon, cmap="gray")
                axes[1, 0].set_title(
                    f"Reconstruction\nc={c0[0].cpu().numpy().astype(int).tolist()}, y={y0.item():.2f}"
                )
                axes[1, 0].axis("off")

                for j, concept_idx in enumerate(concept_indices, start=1):
                    # swap 0↔1 in the one-hot group corresponding to this concept
                    per_concept = self.L
                    if self.L == 0:
                        print(f"WARNING: per-concept z_ci: {self.L}, z_c: {self.z_c_dim}, z_s: {self.style_latent_dim}")
                        print("Counterfactual intervention on z_ci not possible")
                        per_concept = self.z_c_dim // self.k if z_c is not None else self.style_latent_dim // self.k
                        
                    group_start = concept_idx * per_concept
                    group_end = (concept_idx + 1) * per_concept

                    if z_c is not None:
                        # intervene on a single z_c entry
                        z_cf = z_c.clone()
                        z_cf[:, group_start:group_end] = 1.0 - z_cf[:, group_start:group_end]
                        z_all_cf = self._combine_latents(z_cf, z_s)

                        x_cf_feat = self.x_decoder(torch.cat([z_all_cf, c0], dim=1))
                    else:
                        z_sf = z_s.clone()
                        z_sf[:, group_start:group_end] = 1.0 - z_sf[:, group_start:group_end]
                        x_cf_feat = self.x_decoder(torch.cat([z_sf, c0], dim=1))

                    x_cf = self.decode_image(x_cf_feat, torch.cat([z_all_cf if z_c is not None else z_sf, c0], dim=1), z_all_cf if z_c is not None else z_sf, c0).clamp(0, 1)

                    x_cf_feat = self.img_encoder(x_cf)
                    y_cf_logit = self.aux_qy(x_cf_feat)
                    y_cf = torch.sigmoid(y_cf_logit).item()

                    x_cf = x_cf.squeeze().cpu().numpy() if self.channels == 1 else x_cf.squeeze().permute(1, 2, 0).cpu().numpy()

                    axes[1, j].imshow(x_cf, cmap="gray")
                    axes[1, j].set_title(
                        f"Intervene z_c[{concept_names[concept_idx]}]\n"
                        f"c={c0[0].cpu().numpy().astype(int).tolist()}, y={y_cf:.2f}"
                    )
                    axes[1, j].axis("off")

                plt.tight_layout()
                save_path = os.path.join(out_dir, f"cf_example_{example_count}.png")
                plt.savefig(save_path, dpi=150)
                plt.close(fig)

                example_count += 1

    # Centralized image decoding hook (override in subclasses as needed)
    def decode_image(
        self,
        x_recon_feat: torch.Tensor,
        decoder_input: torch.Tensor | None = None,
        z_all: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.img_decoder(x_recon_feat)
