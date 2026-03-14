from typing import Dict, Optional, Tuple
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
from .utils import detect_confounding, pehe, compute_true_ite_ate, GRLWrapper
from .image_modules import MorphoMNISTEncoder, MorphoMNISTDecoder
from .uncovaer_nfivae import ExponentialFamilyPrior

# Backwards compatibility aliases
ImageEncoder = MorphoMNISTEncoder
ImageDecoder = MorphoMNISTDecoder

class UnCoVAEr(pl.LightningModule):
    def __init__(
        self,
        num_concepts: int,
        img_encoder: Optional[nn.Module] = None,
        causal_parents: Optional[list] = None,
        img_decoder: Optional[nn.Module] = None,
        x_decoder: Optional[nn.Module] = None,
        channels: int = 1,
        feat_dim: int = 128,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        indices=None,
        shared_latent_dim: int = 0,
        latent_per_concept: int = 1,
        t_latent_dim: int = 16,
        y_latent_dim: int = 16,
        style_latent_dim: int = 0,
        kl_anneal_start: int = 0,
        kl_anneal_end: int = 0,
        use_aux: bool = True,
        aux_weight_c: float = 1.0,
        aux_weight_y: float = 1.0,
        mim_weight: float = 0.0,
        use_adversarial_independence: bool = False,
        conditional_prior: bool = False,
        conditional_prior_type: str = "idvae",
        pure_idvae: bool = False,
        no_X: bool = False,
        no_C: bool = False,
        x_on_c: bool = True,
        x_on_y: bool = False,
        z_on_c: bool = True,
        z_on_y: bool = True,
        separate_encoders: bool = False,
        marginalize_c: bool = True,
        beta: float = 1.0,
        separate_prior_encoders: bool = True,
        mse_loss: bool = False,
    ):
        """
        UnCoVAEr (Unobserved Confounding Variational AutoEncoder) LightningModule.

        Causal Model (TEDVAE-style):
            Zc (confounder) --> C --> Y
                |                    ^
                +--------------------+
            Zt (treatment-only) --> C
            Zy (outcome-only)   --> Y
            (not used) Zx (style)          --> X only
            All Z components    --> X

        Generative model:
            p(X | Zc, Zt, Zy, Zx)  -- image decoder
            p(C | Zc, Zt)          -- concept decoder
            p(Y | C, Zc, Zy)       -- outcome decoder

        Identification Strategy:
            Zc is identified by being the ONLY latent that predicts both C and Y.
            Optional independence constraints enforce separation:
                - I(Zt; Y | C) -> 0  (treatment latent independent of outcome given concepts)
                - I(Zy; C) -> 0      (outcome latent independent of concepts)
                - I(Zx; C) -> 0      (style latent independent of concepts)
                - I(Zx; Y) -> 0      (style latent independent of outcome)

        Args:
            num_concepts: k (number of concepts)
            latent_per_concept: L (per-concept latent dimensionality)
            t_latent_dim: dimension of treatment-only latent Zt
            y_latent_dim: dimension of outcome-only latent Zy
            style_latent_dim: dimension of style latent Zx (unrelated to C,Y)
            hidden_dim: MLP hidden dims
            lr: learning rate
            kl_anneal_start, kl_anneal_end: epochs for linear KL annealing
            use_aux: whether to include auxiliary q(c|x) and q(y|x,c) terms
            aux_weight_c/y: weights for auxiliary log-likelihoods
            mim_weight: weight for independence constraints (MI or adversarial)
            use_adversarial_independence: if True, use adversarial predictors instead of MI
                - MI approach: minimize upper bound on I(Z; target) using CLUB estimator
                - Adversarial approach: maximize prediction loss (fool the predictor)
            conditional_prior: if True, use conditional prior p(z|c) (experimental, see notes)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["indices", "img_encoder", "img_decoder", "x_decoder"])

        self.channels = channels
        self.k = num_concepts
        self.L = latent_per_concept
        self.shared_latent_dim = shared_latent_dim

        if self.L > 0:
            assert self.shared_latent_dim == 0, "Either per-concept latents or shared latents can be used, not both."
            self.z_c_dim = self.k * self.L
        elif self.shared_latent_dim > 0:
            assert self.L == 0, "Either per-concept latents or shared latents can be used, not both."
            self.z_c_dim = self.shared_latent_dim
        else:
            self.z_c_dim = 0  # No confounder latent
        
        self.t_latent_dim = t_latent_dim
        self.y_latent_dim = y_latent_dim
        self.style_latent_dim = style_latent_dim

        self.latent_dim = self.z_c_dim + self.t_latent_dim + self.y_latent_dim + self.style_latent_dim

        self.lr = lr
        self.hidden_dim = hidden_dim
        self.indices = indices or {}
        
        # causal_parents: list of lists where causal_parents[i] gives parent concept indices for concept i
        self.causal_parents = causal_parents if causal_parents is not None else [[] for _ in range(self.k)]
        self.feat_dim = feat_dim  # encoder feature dim

        self.img_encoder = img_encoder if img_encoder is not None else ImageEncoder(in_channels=self.channels, feat_dim=self.feat_dim)
        self.img_decoder = img_decoder if img_decoder is not None else ImageDecoder(out_channels=self.channels, feat_dim=self.feat_dim)

        self.separate_encoders = separate_encoders
        self.separate_prior_encoders = separate_prior_encoders
        
        self.marginalize_c = marginalize_c

        self.conditional_prior = conditional_prior
        assert conditional_prior_type in ("idvae", "ivae", "nfivae"), "conditional_prior_type must be 'idvae', 'ivae' or 'nfivae'"
        self.conditional_prior_type = conditional_prior_type
        self.beta = beta
        self.mse_loss = mse_loss

        self.no_X = no_X
        self.no_C = no_C
        self.pure_idvae = pure_idvae
        self.x_on_c = x_on_c
        self.x_on_y = x_on_y
        self.z_on_c = z_on_c
        self.z_on_y = z_on_y

        if self.conditional_prior:
            # =======================================================================
            # TEDVAE-style Partition-Specific Conditional Priors
            # =======================================================================
            # 
            # Key insight: Different conditional priors for different latent partitions
            # enforce independence constraints via KL divergence, without needing MI estimators.
            #
            # | Latent | Prior           | KL enforces                    |
            # |--------|-----------------|--------------------------------|
            # | Zc     | p(Zc | C, Y)    | Can encode both C and Y info   |
            # | Zt     | p(Zt | C)       | Only C info → Zt ⊥ Y | C       |
            # | Zy     | p(Zy | Y)       | Only Y info → Zy ⊥ C           |
            # | Zx     | p(Zx) = N(0,I)  | No C,Y info → Zx ⊥ (C,Y)       |
            #
            # KL(q(Zt|X,C,Y) || p(Zt|C)) penalizes encoding Y info that isn't in prior.
            # Since p(Zt|C) doesn't see Y, posterior is pushed to NOT encode Y beyond C.
            #
            # Combined with IDVAE bidirectional structure:
            # - Part ①: KL(q(Z|X,u) || p(Z|u_partition)) with partition-specific u
            # - Part ②: KL(p(Z|u) || N(0,I)) + E[log p(u|Z)] for each partition
            # =======================================================================
            
            self.prior_encoder = None
            self.prior_encoder_zc = None
            self.prior_encoder_zt = None
            self.prior_encoder_zy = None

            self.prior_lambda_clamp = 10.0
            
            # ----- Prior encoder for Zc: p(Zc | C, Y) -----
            # Confounder latent can encode info about BOTH C and Y
            if self.separate_prior_encoders:
                if self.z_c_dim > 0:
                    if self.L > 0:
                        # Per-concept prior encoders: p(Zc_i | C_i, Y) for each concept i
                        # Each prior only sees its own concept value and the outcome
                        self.prior_encoders_zc = nn.ModuleList()
                        self.prior_encoders_zc_mu = nn.ModuleList()
                        self.prior_encoders_zc_logvar = nn.ModuleList()
                        for i in range(self.k):
                            encoder = nn.Sequential(
                                nn.Linear(1 + 1, hidden_dim),  # C_i (1) + Y (1)
                                nn.ELU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ELU(),
                            )
                            self.prior_encoders_zc.append(encoder)
                            self.prior_encoders_zc_mu.append(nn.Linear(hidden_dim, self.L))
                            self.prior_encoders_zc_logvar.append(nn.Linear(hidden_dim, self.L))
                        # Set single encoder to None to avoid confusion
                        self.prior_encoder_zc = None
                        self.prior_encoder_zc_mu = None
                        self.prior_encoder_zc_logvar = None
                    else:
                        # Shared prior encoder: p(Zc | C, Y) - sees all concepts
                        self.prior_encoder_zc = nn.Sequential(
                            nn.Linear(self.k + 1, hidden_dim),  # C (k) + Y (1)
                            nn.ELU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ELU(),
                        )
                        self.prior_encoder_zc_mu = nn.Linear(hidden_dim, self.z_c_dim)
                        self.prior_encoder_zc_logvar = nn.Linear(hidden_dim, self.z_c_dim)
                        # Set per-concept encoders to None
                        self.prior_encoders_zc = None
                        self.prior_encoders_zc_mu = None
                        self.prior_encoders_zc_logvar = None
                
                # ----- Prior encoder for Zt: p(Zt | C) -----
                # Treatment-only latent: conditioned ONLY on C (not Y)
                # KL will enforce Zt ⊥ Y | C
                if self.t_latent_dim > 0:
                    self.prior_encoder_zt = nn.Sequential(
                        nn.Linear(self.k, hidden_dim),  # Only C (k concepts)
                        nn.ELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ELU(),
                    )
                    self.prior_encoder_zt_mu = nn.Linear(hidden_dim, self.t_latent_dim)
                    self.prior_encoder_zt_logvar = nn.Linear(hidden_dim, self.t_latent_dim)
                
                # ----- Prior encoder for Zy: p(Zy | Y) -----
                # Outcome-only latent: conditioned ONLY on Y (not C)
                # KL will enforce Zy ⊥ C
                if self.y_latent_dim > 0:
                    self.prior_encoder_zy = nn.Sequential(
                        nn.Linear(1, hidden_dim),  # Only Y (1)
                        nn.ELU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ELU(),
                    )
                    self.prior_encoder_zy_mu = nn.Linear(hidden_dim, self.y_latent_dim)
                    self.prior_encoder_zy_logvar = nn.Linear(hidden_dim, self.y_latent_dim)
                
                # ----- Prior for Zx: p(Zx) = N(0, I) -----
                # Style latent: unconditional prior (no C or Y info)
                # KL will enforce Zx ⊥ (C, Y)
                # (No encoder needed - just use standard normal)
            else:
                self.prior_encoder = nn.Sequential(
                    nn.Linear(self.k + 1, hidden_dim),  # C (k) + Y (1)
                    nn.ELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ELU(),
                )
                self.prior_encoder_mu = nn.Linear(hidden_dim, self.latent_dim)
                self.prior_encoder_logvar = nn.Linear(hidden_dim, self.latent_dim)

            # Optional: NF-iVAE exponential-family priors for partitions
            if self.conditional_prior_type == "nfivae":
                if self.z_c_dim > 0:
                    self.prior_zc = ExponentialFamilyPrior(latent_dim=self.z_c_dim, num_concepts=self.k, hidden_dim=hidden_dim, use_nf=True)
                else:
                    self.prior_zc = None

                if self.t_latent_dim > 0:
                    # Prior for Zt conditioned on C only (we will pass y=zeros when computing)
                    self.prior_zt = ExponentialFamilyPrior(latent_dim=self.t_latent_dim, num_concepts=self.k, hidden_dim=hidden_dim, use_nf=True)
                else:
                    self.prior_zt = None

                if self.y_latent_dim > 0:
                    # Prior for Zy conditioned on Y only (we will pass c=zeros when computing)
                    self.prior_zy = ExponentialFamilyPrior(latent_dim=self.y_latent_dim, num_concepts=self.k, hidden_dim=hidden_dim, use_nf=True)
                else:
                    self.prior_zy = None
            else:
                # keep attributes for compatibility
                self.prior_zc = None
                self.prior_zt = None
                self.prior_zy = None


        # ---------- share first layers of posterior encoder ----------
        post_input_dim = self.feat_dim
        if self.z_on_c:
            post_input_dim += self.k
        if self.z_on_y:
            post_input_dim += 1

        if self.separate_encoders:
            # ---------- posterior q_phi(Zt | X, C, Y) ----------
            if self.t_latent_dim > 0:
                self.post_net_t = nn.Sequential(
                    nn.Linear(post_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.mu_t = nn.Linear(hidden_dim, self.t_latent_dim)  # treatment Gaussian
                self.logvar_t = nn.Linear(hidden_dim, self.t_latent_dim)

            # ---------- posterior q_phi(Zy | X, C, Y) ----------
            if self.y_latent_dim > 0:
                self.post_net_y = nn.Sequential(
                    nn.Linear(post_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.mu_y = nn.Linear(hidden_dim, self.y_latent_dim)  # outcome Gaussian
                self.logvar_y = nn.Linear(hidden_dim, self.y_latent_dim)

            # ---------- posterior q_phi(Zc | X, C, Y) ----------
            if self.z_c_dim > 0:
                self.post_net_c = nn.Sequential(
                    nn.Linear(post_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.mu_c = nn.Linear(hidden_dim, self.z_c_dim)  # concept Gaussian
                self.logvar_c = nn.Linear(hidden_dim, self.z_c_dim)

            # ---------- posterior q_phi(Zo | X, C, Y) ----------
            if self.style_latent_dim > 0:
                self.post_net_o = nn.Sequential(
                    nn.Linear(post_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.mu_x = nn.Linear(hidden_dim, self.style_latent_dim)  # style Gaussian
                self.logvar_x = nn.Linear(hidden_dim, self.style_latent_dim)
        else:
            self.post_net = nn.Sequential(
                nn.Linear(post_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mu = nn.Linear(hidden_dim, self.latent_dim)  # single Gaussian latent
            self.logvar = nn.Linear(hidden_dim, self.latent_dim)

        # ---------------- image decoder p(X | Zc, Zt, Zy) (feature reconstruction) ----------------
        # We decode to encoder feature space (feat_dim).
        if self.no_X:
            self.x_decoder = None
        else:
            x_input_dim = self.latent_dim
            if self.x_on_c:
                x_input_dim += self.k  # c
            if self.x_on_y:
                x_input_dim += 1  # y
            self.x_decoder = x_decoder if x_decoder is not None else nn.Sequential(
                nn.Linear(x_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.feat_dim)   # reconstruct x_feat
            )

        # ---------- per-concept decoders p(C_i | Zc_i, Zt_i) (logits) ----------
        if self.L > 0:
            assert self.t_latent_dim == self.L * self.k or self.t_latent_dim == 0, "When using per-concept latents, treatment latent dim must be k * L or zero."
            tL = self.t_latent_dim // self.k
            # Build per-concept treatment heads that accept own latent, treatment latent slice, and parent latents
            self.treat_heads = nn.ModuleList()
            for i in range(self.k):
                num_parents = len(self.causal_parents[i]) if self.causal_parents is not None else 0
                input_dim = self.L + tL + num_parents
                hidden_dim_t = max(32, input_dim * 4)
                self.treat_heads.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim_t),
                    nn.ReLU(),
                    nn.Linear(hidden_dim_t, 1)
                ))
        # ---------- shared concept decoder p(C | Zc, Zt) (logits) ----------
        elif self.causal_parents is not None and any(len(parents) > 0 for parents in self.causal_parents):
            self.treat_heads = nn.ModuleList()
            for i in range(self.k):
                num_parents = len(self.causal_parents[i])
                input_dim = self.shared_latent_dim + self.t_latent_dim + num_parents
                hidden_dim_t = max(32, input_dim * 4)
                self.treat_heads.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim_t),
                    nn.ReLU(),
                    nn.Linear(hidden_dim_t, 1)
                ))
        else:
            self.shared_treat_head = nn.Sequential(
                nn.Linear(self.shared_latent_dim + self.t_latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.k)  # logits for k Bernoulli
            )

        # ---------- outcome decoder p(Y | C, Zc, Zy) (logit) ----------
        # input: concat(C (k), Zc (z_c_dim), Zy (y_latent_dim))
        y_in = self.k + self.z_c_dim + self.y_latent_dim
        self.y_head = nn.Sequential(
            nn.Linear(y_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # logit
        )

        # ---------- auxiliary networks q(C | Zc, Zt), q(Y | C, Zc, Zy) ----------
        self.use_aux = use_aux
        self.aux_weight_c = aux_weight_c
        self.aux_weight_y = aux_weight_y
        self.aux_qc = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.k)  # logits for k Bernoulli
        )
        self.aux_qy = nn.Sequential(
            nn.Linear(self.feat_dim + self.k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # logit for y
        )

        # KL annealing schedule
        self.kl_anneal_start = kl_anneal_start
        self.kl_anneal_end = kl_anneal_end

        self.f1_score = BinaryF1Score()
        self.accuracy = Accuracy(task='binary')

        self.likelihood_log_sigma = nn.Parameter(torch.tensor(0.0))

        self.mim_weight = mim_weight
        self.use_adversarial_independence = use_adversarial_independence

        # Gradient Reversal Layer for adversarial independence
        self.grl_lambda = 1.0
        self.grl = GRLWrapper(self.grl_lambda)

        # Independence constraints for identification:
        # I(Zt; Y | C) -> 0: treatment latent should not predict outcome given concepts
        # I(Zy; C) -> 0: outcome latent should not predict concepts
        # I(Zx; C) -> 0: style latent should not predict concepts
        # I(Zx; Y) -> 0: style latent should not predict outcome
        
        if self.mim_weight > 0:
            # Baseline predictor C -> Y for computing residuals (used for conditional MI)
            # This predicts E[Y|C], then we test if Zt can predict Y - E[Y|C]
            self.baseline_y_from_c = nn.Sequential(
                nn.Linear(self.k, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            if self.use_adversarial_independence:
                # Adversarial predictors: train to predict, encoder trains to fool them
                # Both losses added to total_loss with detach() controlling gradient flow
                
                # Predictor for Y-residual from Zt (after regressing out C's contribution)
                # This correctly tests I(Zt; Y | C) by predicting the residual
                if self.t_latent_dim > 0:
                    self.adv_zt_y = nn.Sequential(
                        nn.Linear(self.t_latent_dim + self.k, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1)
                    )
                else:
                    self.adv_zt_y = None
                
                # Predictor for C from Zy
                if self.y_latent_dim > 0:
                    self.adv_zy_c = nn.Sequential(
                        nn.Linear(self.y_latent_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, self.k)
                    )
                else:
                    self.adv_zy_c = None
                
                # Predictor for C from Zx
                if self.style_latent_dim > 0:
                    self.adv_zx_c = nn.Sequential(
                        nn.Linear(self.style_latent_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, self.k)
                    )
                    
                    # Predictor for Y from Zx
                    self.adv_zx_y = nn.Sequential(
                        nn.Linear(self.style_latent_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1)
                    )
                else:
                    self.adv_zx_c = None
                    self.adv_zx_y = None
                
                # Set MI estimators to None when using adversarial
                self.mi_loss_fn_z_t = None
                self.mi_loss_fn_z_y = None
                self.mi_loss_fn_zx_c = None
                self.mi_loss_fn_zx_y = None
            else:
                # MI-based independence using CLUB estimator
                # I(Zt; Y | C): use residualization - estimate I(Zt; Y - E[Y|C])
                # This correctly tests conditional independence by removing C's contribution
                if self.t_latent_dim > 0:
                    self.mi_loss_fn_z_t = MutualInformationLoss(
                        x_dim=self.t_latent_dim,  # Just Zt (we'll compute residual of Y)
                        y_dim=1,
                        hidden_dim=hidden_dim,
                        lr=lr
                    )
                else:
                    self.mi_loss_fn_z_t = None
                
                # I(Zy; C)
                if self.y_latent_dim > 0:
                    self.mi_loss_fn_z_y = MutualInformationLoss(
                        x_dim=self.y_latent_dim,
                        y_dim=self.k,
                        hidden_dim=hidden_dim,
                        lr=lr
                    )
                else:
                    self.mi_loss_fn_z_y = None
                
                # I(Zx; C) and I(Zx; Y)
                if self.style_latent_dim > 0:
                    self.mi_loss_fn_zx_c = MutualInformationLoss(
                        x_dim=self.style_latent_dim,
                        y_dim=self.k,
                        hidden_dim=hidden_dim,
                        lr=lr
                    )
                    self.mi_loss_fn_zx_y = MutualInformationLoss(
                        x_dim=self.style_latent_dim,
                        y_dim=1,
                        hidden_dim=hidden_dim,
                        lr=lr
                    )
                else:
                    self.mi_loss_fn_zx_c = None
                    self.mi_loss_fn_zx_y = None
        else:
            self.mi_loss_fn_z_t = None
            self.mi_loss_fn_z_y = None
            self.mi_loss_fn_zx_c = None
            self.mi_loss_fn_zx_y = None
        
    # --------------------- helpers ---------------------
    def _post_params(self, x_feat: torch.Tensor, c: torch.Tensor, y: torch.Tensor):
        """
        Posterior parameters for concept latents: q(zc, zt, zy, (zo) | x).
        """
        y = y.unsqueeze(1) if y.dim() == 1 else y
        post_input = [x_feat]
        if self.z_on_c:
            post_input.append(c)
        if self.z_on_y:
            post_input.append(y)
        post_input = torch.cat(post_input, dim=1)

        if self.separate_encoders:
            mu, logvar, z = None, None, None
        else:
            h = self.post_net(post_input)
            mu = self.mu(h)
            logvar = self.logvar(h)
            z = self.sample_gaussian(mu, logvar)

        mu_c, logvar_c, z_c = None, None, None
        mu_t, logvar_t, z_t = None, None, None
        mu_y, logvar_y, z_y = None, None, None
        mu_x, logvar_x, z_x = None, None, None

        if self.separate_encoders:
            if self.z_c_dim > 0:
                h_c = self.post_net_c(post_input)
                mu_c = self.mu_c(h_c)
                logvar_c = self.logvar_c(h_c)
                z_c = self.sample_gaussian(mu_c, logvar_c)

            if self.t_latent_dim > 0:
                h_t = self.post_net_t(post_input)
                mu_t = self.mu_t(h_t)
                logvar_t = self.logvar_t(h_t)
                z_t = self.sample_gaussian(mu_t, logvar_t)
            
            if self.y_latent_dim > 0:
                h_y = self.post_net_y(post_input)
                mu_y = self.mu_y(h_y)
                logvar_y = self.logvar_y(h_y)
                z_y = self.sample_gaussian(mu_y, logvar_y)
            
            if self.style_latent_dim > 0:
                h_x = self.post_net_o(post_input)
                mu_x = self.mu_x(h_x)
                logvar_x = self.logvar_x(h_x)
                z_x = self.sample_gaussian(mu_x, logvar_x)
            
            mu = self._combine_latents(mu_c, mu_t, mu_y, mu_x)
            logvar = self._combine_latents(logvar_c, logvar_t, logvar_y, logvar_x)
            z = self._combine_latents(z_c, z_t, z_y, z_x)
        else:
            # Split combined z AND mu/logvar into partitions
            z_end = 0
            if self.z_c_dim > 0:
                z_c = z[:, :self.z_c_dim]
                mu_c = mu[:, :self.z_c_dim]
                logvar_c = logvar[:, :self.z_c_dim]
                z_end += self.z_c_dim
        
            if self.t_latent_dim > 0:
                z_t = z[:, z_end:z_end + self.t_latent_dim]
                mu_t = mu[:, z_end:z_end + self.t_latent_dim]
                logvar_t = logvar[:, z_end:z_end + self.t_latent_dim]
                z_end += self.t_latent_dim
            
            if self.y_latent_dim > 0:
                z_y = z[:, z_end:z_end + self.y_latent_dim]
                mu_y = mu[:, z_end:z_end + self.y_latent_dim]
                logvar_y = logvar[:, z_end:z_end + self.y_latent_dim]
                z_end += self.y_latent_dim
            
            if self.style_latent_dim > 0:
                z_x = z[:, z_end:z_end + self.style_latent_dim]
                mu_x = mu[:, z_end:z_end + self.style_latent_dim]
                logvar_x = logvar[:, z_end:z_end + self.style_latent_dim]
                z_end += self.style_latent_dim

        return z_c, z_t, z_y, z_x, z, mu_c, logvar_c, mu_t, logvar_t, mu_y, logvar_y, mu_x, logvar_x, mu, logvar
    
    def _split_latents(self, z: torch.Tensor):
        """
        z: [B, k * L]
        Returns:
        z_chunks: list of [B,L] (concept-specific latents)
        """
        if z is None:
            return [None for _ in range(self.k)]
        B = z.size(0)
        z_concepts = z.view(B, self.k, self.L)
        z_chunks = [z_concepts[:, i, :] for i in range(self.k)]
        return z_chunks

    def _combine_latents(self, *args):
        to_combine = [arg for arg in args if arg is not None]
        return torch.cat(to_combine, dim=1)

    def decode_image(self, x_recon_feat: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        """Decode latent features back to image space. Subclasses may override."""
        return self.img_decoder(x_recon_feat)

    @staticmethod
    def sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def prior_params_from_u(self, c: torch.Tensor, y: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute TEDVAE-style partition-specific conditional prior params.
        
        Returns a dict with prior params for each partition:
        - 'zc': (mu, logvar) from p(Zc | C, Y) - confounder prior
        - 'zt': (mu, logvar) from p(Zt | C)   - treatment prior (Zt ⊥ Y | C)
        - 'zy': (mu, logvar) from p(Zy | Y)   - outcome prior (Zy ⊥ C)
        - 'zx': (mu, logvar) = (0, 0)         - style prior N(0,I) (Zx ⊥ C,Y)
        
        The key insight: by conditioning priors differently, KL divergence
        naturally enforces independence constraints without MI estimators.
        """
        if not self.conditional_prior:
            raise RuntimeError("prior_params_from_u called but conditional_prior=False")

        # Ensure y has correct shape
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        B = c.size(0)
        device = c.device
        prior_params = {}
        
        if self.separate_prior_encoders:
            # ----- p(Zc | C, Y): confounder prior (both C and Y) -----
            if self.z_c_dim > 0:
                if self.L > 0 and self.prior_encoders_zc is not None:
                    # Per-concept priors: p(Zc_i | C_i, Y) for each concept i
                    mu_zc_list = []
                    logvar_zc_list = []
                    for i in range(self.k):
                        u_zc_i = torch.cat([c[:, i:i+1], y], dim=1)  # [B, 2]: (C_i, Y)
                        h_zc_i = self.prior_encoders_zc[i](u_zc_i)
                        mu_zc_i = self.prior_encoders_zc_mu[i](h_zc_i)  # [B, L]
                        logvar_zc_i = self.prior_encoders_zc_logvar[i](h_zc_i)  # [B, L]
                        mu_zc_list.append(mu_zc_i)
                        logvar_zc_list.append(logvar_zc_i)
                    mu_zc = torch.cat(mu_zc_list, dim=1)  # [B, k*L]
                    logvar_zc = torch.clamp(torch.cat(logvar_zc_list, dim=1),
                                            -self.prior_lambda_clamp, self.prior_lambda_clamp)
                else:
                    # Shared prior: p(Zc | C, Y) - sees all concepts
                    u_zc = torch.cat([c, y], dim=1)  # [B, k+1]
                    h_zc = self.prior_encoder_zc(u_zc)
                    mu_zc = self.prior_encoder_zc_mu(h_zc)
                    logvar_zc = torch.clamp(self.prior_encoder_zc_logvar(h_zc), 
                                            -self.prior_lambda_clamp, self.prior_lambda_clamp)
                prior_params['zc'] = (mu_zc, logvar_zc)
            else:
                prior_params['zc'] = (None, None)
            
            # ----- p(Zt | C): treatment prior (C only, NOT Y) -----
            # This enforces Zt ⊥ Y | C via KL
            if self.t_latent_dim > 0:
                u_zt = c  # [B, k] - only concepts, no outcome!
                h_zt = self.prior_encoder_zt(u_zt)
                mu_zt = self.prior_encoder_zt_mu(h_zt)
                logvar_zt = torch.clamp(self.prior_encoder_zt_logvar(h_zt),
                                        -self.prior_lambda_clamp, self.prior_lambda_clamp)
                prior_params['zt'] = (mu_zt, logvar_zt)
            else:
                prior_params['zt'] = (None, None)
            
            # ----- p(Zy | Y): outcome prior (Y only, NOT C) -----
            # This enforces Zy ⊥ C via KL
            if self.y_latent_dim > 0:
                u_zy = y  # [B, 1] - only outcome, no concepts!
                h_zy = self.prior_encoder_zy(u_zy)
                mu_zy = self.prior_encoder_zy_mu(h_zy)
                logvar_zy = torch.clamp(self.prior_encoder_zy_logvar(h_zy),
                                        -self.prior_lambda_clamp, self.prior_lambda_clamp)
                prior_params['zy'] = (mu_zy, logvar_zy)
            else:
                prior_params['zy'] = (None, None)
            
            # ----- p(Zx) = N(0, I): style prior (unconditional) -----
            # This enforces Zx ⊥ (C, Y) via KL to standard normal
            if self.style_latent_dim > 0:
                mu_zx = torch.zeros(B, self.style_latent_dim, device=device)
                logvar_zx = torch.zeros(B, self.style_latent_dim, device=device)
                prior_params['zx'] = (mu_zx, logvar_zx)
            else:
                prior_params['zx'] = (None, None)
        else:
            # Single prior encoder for all latents: p(Z | C, Y)
            u = torch.cat([c, y], dim=1)  # [B, k+1]
            h = self.prior_encoder(u)
            mu = self.prior_encoder_mu(h)
            logvar = torch.clamp(self.prior_encoder_logvar(h),
                                 -self.prior_lambda_clamp, self.prior_lambda_clamp)
            
            z_end = 0
            prior_params['zc'] = (mu[:, :self.z_c_dim] if self.z_c_dim > 0 else None,
                                  logvar[:, :self.z_c_dim] if self.z_c_dim > 0 else None)
            z_end += self.z_c_dim
            prior_params['zt'] = (mu[:, z_end:z_end + self.t_latent_dim] if self.t_latent_dim > 0 else None,
                                  logvar[:, z_end:z_end + self.t_latent_dim] if self.t_latent_dim > 0 else None)
            z_end += self.t_latent_dim
            prior_params['zy'] = (mu[:, z_end:z_end + self.y_latent_dim] if self.y_latent_dim > 0 else None,
                                  logvar[:, z_end:z_end + self.y_latent_dim] if self.y_latent_dim > 0 else None)
            z_end += self.y_latent_dim
            prior_params['zx'] = (mu[:, z_end:z_end + self.style_latent_dim] if self.style_latent_dim > 0 else None,
                                  logvar[:, z_end:z_end + self.style_latent_dim] if self.style_latent_dim > 0 else None)
            
        return prior_params

    def kl_gaussian(self, mu_q, logvar_q, mu_p=None, logvar_p=None):
        """
        KL( N(mu_q, var_q) || N(mu_p, var_p) ) for diagonal Gaussians.
        Defaults to standard normal prior if mu_p/logvar_p are None.
        """
        if mu_p is None:
            mu_p = torch.zeros_like(mu_q)
        if logvar_p is None:
            logvar_p = torch.zeros_like(logvar_q)  # log(1) = 0

        var_q = torch.exp(logvar_q).clamp(min=1e-8)
        var_p = torch.exp(logvar_p).clamp(min=1e-8)

        term = (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0 + (logvar_p - logvar_q)
        kl = 0.5 * torch.sum(term)  # sum over latent dims and batch
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

    def gaussian_nll(self, x_recon, x):
        """
        Negative log-likelihood under Gaussian with learnable sigma.
        """
        log_sigma = torch.clamp(self.likelihood_log_sigma, -4.0, 2.0)
        logvar = 2 * log_sigma  # log σ²
        recon_loss = 0.5 * ((x - x_recon) ** 2 / torch.exp(logvar) + logvar + math.log(2 * math.pi))
        return recon_loss.sum()  # sum over pixels and batch

    # --------------------- forward / generative ---------------------
    def forward(self, x, c, y):
        # encode x -> features
        x_feat = self.img_encoder(x)
        
        # Encode and sample latents q(Zc, Zt, Zy, Zx | X, C, Y)
        post = self._post_params(x_feat, c, y)
        # unpack returned tuple (with optional trailing global-gmm items)
        (z_c, z_t, z_y, z_x, z,
         mu_c, logvar_c,
         mu_t, logvar_t,
         mu_y, logvar_y,
         mu_x, logvar_x,
         mu, logvar) = post

        y_logits = self.y_head(self._combine_latents(c, z_c, z_y))
        
        # C reconstruction uses Zc, Zt (if not zero)
        if self.no_C:
            c_logits = None
        else:
            if self.L > 0:
                z_c_chunks = self._split_latents(z_c)  # list of [B,L]
                z_t_chunks = self._split_latents(z_t)  # list of [B,L] (may contain None)
                c_logits_list = []
                for i, head in enumerate(self.treat_heads):
                    parts = [z_c_chunks[i], z_t_chunks[i]]
                    for p in (self.causal_parents[i] if self.causal_parents is not None else []):
                        parts.append(c[:, p].unsqueeze(-1))
                    inp = self._combine_latents(*parts)
                    c_logits_list.append(head(inp))
                c_logits = torch.cat(c_logits_list, dim=1)
            elif self.causal_parents is not None and any(len(parents) > 0 for parents in self.causal_parents):
                c_logits_list = []
                for i, head in enumerate(self.treat_heads):
                    parts = [z_c, z_t]
                    for p in self.causal_parents[i]:
                        parts.append(c[:, p].unsqueeze(-1))
                    inp = self._combine_latents(*parts)
                    c_logits_list.append(head(inp))
                c_logits = torch.cat(c_logits_list, dim=1)
            else:
                c_logits = self.shared_treat_head(self._combine_latents(z_c, z_t))

        # X reconstruction uses *all latents*
        if self.no_X:
            x_recon = None
        else:
            decoder_input = z
            if self.x_on_c:
                decoder_input = torch.cat((decoder_input, c), dim=1)
            if self.x_on_y:
                decoder_input = torch.cat((decoder_input, y), dim=1)
            x_recon_feat = self.x_decoder(decoder_input)
            x_recon = self.decode_image(x_recon_feat, decoder_input)

        out = {
            "z_c": z_c, "z_t": z_t, "z_y": z_y, "z_x": z_x,
            "mu_c": mu_c, "logvar_c": logvar_c,
            "mu_t": mu_t, "logvar_t": logvar_t,
            "mu_y": mu_y, "logvar_y": logvar_y,
            "mu_x": mu_x, "logvar_x": logvar_x,
            "c_logits": c_logits, "y_logits": y_logits,
            "x_recon": x_recon, "x_feat": x_feat,
            "z": z,
            "mu": mu, "logvar": logvar,
        }

        return out
                
    # --------------------- loss computation ---------------------
    def compute_elbo(self, batch: torch.Tensor, stage: str = "train") -> Dict[str, torch.Tensor]:
        x, attr = batch
        y = attr[:, self.indices['task']]
        c = attr[:, self.indices['concepts']]
        y = y.unsqueeze(1) if y.dim() == 1 else y
        out = self.forward(x, c, y)

        mu, logvar = out["mu"], out["logvar"]
        x_feat = out["x_feat"]
        x_recon = out["x_recon"]

        B = x.size(0)

        total_loss = 0.0

        # =======================================================================
        # TEDVAE-style IDVAE with Partition-Specific Conditional Priors
        # =======================================================================
        # 
        # Key insight: Different priors for different partitions enforce independence
        # via KL divergence, without needing MI estimators!
        #
        # | Latent | Prior         | What KL enforces                    |
        # |--------|---------------|-------------------------------------|
        # | Zc     | p(Zc | C, Y)  | Can encode both C and Y info        |
        # | Zt     | p(Zt | C)     | Only C info → Zt ⊥ Y | C            |
        # | Zy     | p(Zy | Y)     | Only Y info → Zy ⊥ C                |
        #
        # Part ①: KL(q(Z_partition|X,C,Y) || p(Z_partition|u_partition))
        #         Each partition has its own conditional prior with different u!
        #
        # Part ②: KL(p(Z|u) || N(0,I)) + E[log p(u|Z)]
        #         Prior VAE regularization (IDVAE style)
        # =======================================================================
        
        # Initialize partition-specific KL terms
        kl_zc = torch.tensor(0.0, device=x.device)
        kl_zt = torch.tensor(0.0, device=x.device)
        kl_zy = torch.tensor(0.0, device=x.device)
        kl_zx = torch.tensor(0.0, device=x.device)
        
        # Part ② terms (prior regularization)
        kl_prior_zc = torch.tensor(0.0, device=x.device)
        kl_prior_zt = torch.tensor(0.0, device=x.device)
        kl_prior_zy = torch.tensor(0.0, device=x.device)
        # Zx prior is already N(0,I), so no extra KL needed
        
        loss_C_prior = torch.tensor(0.0, device=x.device)
        loss_Y_prior = torch.tensor(0.0, device=x.device)
        loss_sm = torch.tensor(0.0, device=x.device)
        
        # Get posterior params from forward pass
        mu_c, logvar_c = out.get("mu_c"), out.get("logvar_c")
        mu_t, logvar_t = out.get("mu_t"), out.get("logvar_t")
        mu_y, logvar_y = out.get("mu_y"), out.get("logvar_y")
        mu_x, logvar_x = out.get("mu_x"), out.get("logvar_x")
        
        z_c = out["z_c"]
        z_t = out["z_t"]
        z_y = out["z_y"]
        z_x = out["z_x"]

        if self.conditional_prior:
            # =================================================================
            # IDVAE with Partition-Specific Conditional Priors
            # =================================================================
            #
            # Part ①: Main VAE (encoder φ, decoder θ)
            #   - Reconstruction: p(X | z) where z ~ q_φ(z | X, C, Y)
            #   - KL: KL(q_φ(z | X, C, Y) || p_ψ(z | C, Y))  [detached prior]
            #
            # Part ②: Prior VAE (prior encoder ψ)
            #   - Reconstruction: p(C, Y | z) where z ~ p_ψ(z | C, Y)  
            #   - KL: KL(p_ψ(z | C, Y) || N(0, I))
            #
            # The detach() on prior params approximates alternating optimization:
            # Part ① doesn't update the prior encoder, Part ② does.
            # =================================================================
            
            # Get partition-specific prior params (computed ONCE)
            prior_params = self.prior_params_from_u(c, y)

            if self.conditional_prior_type == "idvae":
                # ===== Part ① KL terms (posterior → conditional prior, DETACHED) =====
                # These update the ENCODER only, not the prior encoder
                # Zc: p(Zc | C, Y) - confounder can encode both
                if self.z_c_dim > 0 and mu_c is not None:
                    prior_mu_zc, prior_logvar_zc = prior_params['zc']
                    kl_zc = self.kl_gaussian(mu_c, logvar_c,
                                              prior_mu_zc.detach(), prior_logvar_zc.detach()) / B

                # Zt: p(Zt | C) - treatment only depends on C, NOT Y
                if self.t_latent_dim > 0 and mu_t is not None:
                    prior_mu_zt, prior_logvar_zt = prior_params['zt']
                    kl_zt = self.kl_gaussian(mu_t, logvar_t,
                                              prior_mu_zt.detach(), prior_logvar_zt.detach()) / B

                # Zy: p(Zy | Y) - outcome only depends on Y, NOT C
                if self.y_latent_dim > 0 and mu_y is not None:
                    prior_mu_zy, prior_logvar_zy = prior_params['zy']
                    kl_zy = self.kl_gaussian(mu_y, logvar_y,
                                              prior_mu_zy.detach(), prior_logvar_zy.detach()) / B

                # Zx: p(Zx) = N(0, I) - style is unconditional
                if self.style_latent_dim > 0 and mu_x is not None:
                    kl_zx = self.kl_gaussian(mu_x, logvar_x, None, None) / B

                # ===== Part ② KL terms (conditional prior → standard normal) =====
                # These update the PRIOR ENCODER
                if self.z_c_dim > 0:
                    prior_mu_zc, prior_logvar_zc = prior_params['zc']
                    kl_prior_zc = self.kl_gaussian(prior_mu_zc, prior_logvar_zc, None, None) / B

                if self.t_latent_dim > 0:
                    prior_mu_zt, prior_logvar_zt = prior_params['zt']
                    kl_prior_zt = self.kl_gaussian(prior_mu_zt, prior_logvar_zt, None, None) / B

                if self.y_latent_dim > 0:
                    prior_mu_zy, prior_logvar_zy = prior_params['zy']
                    kl_prior_zy = self.kl_gaussian(prior_mu_zy, prior_logvar_zy, None, None) / B

                # ===== Part ② reconstruction: p(C, Y | z) from PRIOR samples =====
                # Sample from prior (reuse prior_params, don't recompute!)
                z_prior_samples = []
                for key in ['zc', 'zt', 'zy', 'zx']:
                    mu, logvar = prior_params[key]
                    if mu is not None:
                        z_prior_samples.append(self.sample_gaussian(mu, logvar))
                z_prior = torch.cat(z_prior_samples, dim=1) if z_prior_samples else None

                # Split prior samples into partitions
                z_c_prior = z_prior[:, :self.z_c_dim] if self.z_c_dim > 0 else None
                z_end = self.z_c_dim
                z_t_prior = z_prior[:, z_end:z_end + self.t_latent_dim] if self.t_latent_dim > 0 else None
                z_end += self.t_latent_dim
                z_y_prior = z_prior[:, z_end:z_end + self.y_latent_dim] if self.y_latent_dim > 0 else None

                # p(C | z_prior) - Part ② reconstruction (updates prior encoder!)
                if not self.no_C:
                    if self.L > 0:
                        z_c_chunks_prior = self._split_latents(z_c_prior)
                        z_t_chunks_prior = self._split_latents(z_t_prior)
                        c_logits_prior_list = []
                        for i, head in enumerate(self.treat_heads):
                            parts = [z_c_chunks_prior[i], z_t_chunks_prior[i]]
                            for p_idx in (self.causal_parents[i] if self.causal_parents is not None else []):
                                parts.append(c[:, p_idx].unsqueeze(-1))
                            inp = self._combine_latents(*parts)
                            c_logits_prior_list.append(head(inp))
                        c_logits_prior = torch.cat(c_logits_prior_list, dim=1)
                    elif self.causal_parents is not None and any(len(parents) > 0 for parents in self.causal_parents):
                        c_logits_prior_list = []
                        for i, head in enumerate(self.treat_heads):
                            parts = [z_c_prior, z_t_prior]
                            for p_idx in self.causal_parents[i]:
                                parts.append(c[:, p_idx].unsqueeze(-1))
                            inp = self._combine_latents(*parts)
                            c_logits_prior_list.append(head(inp))
                        c_logits_prior = torch.cat(c_logits_prior_list, dim=1)
                    else:
                        c_logits_prior = self.shared_treat_head(self._combine_latents(z_c_prior, z_t_prior))
                    loss_C_prior = F.binary_cross_entropy_with_logits(c_logits_prior, c, reduction='sum') / B

                # p(Y | z_prior, C) - Part ② reconstruction (updates prior encoder!)
                y_logits_prior = self.y_head(self._combine_latents(c, z_c_prior, z_y_prior))
                loss_Y_prior = F.binary_cross_entropy_with_logits(y_logits_prior, y, reduction='sum') / B            
            elif self.conditional_prior_type == "nfivae":
                # NF-iVAE branch: use unnormalized log-prob for priors and score-matching to train priors
                # Compute negative entropy terms for Gaussian posteriors per-partition
                if self.z_c_dim > 0 and mu_c is not None:
                    neg_entropy_zc = -0.5 * torch.sum(1.0 + logvar_c + math.log(2 * math.pi)) / B
                    # E_q[log p_unnorm(z|u)] with detached prior params (encoder update)
                    if getattr(self, 'prior_zc', None) is not None:
                        log_p_zc = self.prior_zc.log_prob_unnormalized(z_c, c, y, detach_prior=True).mean()
                    else:
                        log_p_zc = torch.tensor(0.0, device=x.device)
                    kl_zc = neg_entropy_zc - log_p_zc
                    # score matching loss for prior training (detached z)
                    if getattr(self, 'prior_zc', None) is not None:
                        sm_zc = self.prior_zc.score_matching_loss(z_c.detach(), c, y) if stage != "test" else torch.tensor(0.0, device=x.device)
                    else:
                        sm_zc = torch.tensor(0.0, device=x.device)
                else:
                    kl_zc = torch.tensor(0.0, device=x.device)
                    sm_zc = torch.tensor(0.0, device=x.device)

                if self.t_latent_dim > 0 and mu_t is not None:
                    neg_entropy_zt = -0.5 * torch.sum(1.0 + logvar_t + math.log(2 * math.pi)) / B
                    # Prior p(Zt|C): pass y zeros to ignore Y
                    y_zeros = torch.zeros_like(y)
                    if getattr(self, 'prior_zt', None) is not None:
                        log_p_zt = self.prior_zt.log_prob_unnormalized(z_t, c, y_zeros, detach_prior=True).mean()
                        sm_zt = self.prior_zt.score_matching_loss(z_t.detach(), c, y_zeros) if stage != "test" else torch.tensor(0.0, device=x.device)
                    else:
                        log_p_zt = torch.tensor(0.0, device=x.device)
                        sm_zt = torch.tensor(0.0, device=x.device)
                    kl_zt = neg_entropy_zt - log_p_zt
                else:
                    kl_zt = torch.tensor(0.0, device=x.device)
                    sm_zt = torch.tensor(0.0, device=x.device)

                if self.y_latent_dim > 0 and mu_y is not None:
                    neg_entropy_zy = -0.5 * torch.sum(1.0 + logvar_y + math.log(2 * math.pi)) / B
                    # Prior p(Zy|Y): pass c zeros to ignore C
                    c_zeros = torch.zeros_like(c)
                    if getattr(self, 'prior_zy', None) is not None:
                        log_p_zy = self.prior_zy.log_prob_unnormalized(z_y, c_zeros, y, detach_prior=True).mean()
                        sm_zy = self.prior_zy.score_matching_loss(z_y.detach(), c_zeros, y) if stage != "test" else torch.tensor(0.0, device=x.device)
                    else:
                        log_p_zy = torch.tensor(0.0, device=x.device)
                        sm_zy = torch.tensor(0.0, device=x.device)
                    kl_zy = neg_entropy_zy - log_p_zy
                else:
                    kl_zy = torch.tensor(0.0, device=x.device)
                    sm_zy = torch.tensor(0.0, device=x.device)

                # Zx remains standard normal KL
                if self.style_latent_dim > 0 and mu_x is not None:
                    kl_zx = self.kl_gaussian(mu_x, logvar_x, None, None) / B
                else:
                    kl_zx = torch.tensor(0.0, device=x.device)

                # Sum score-matching losses (these train the prior networks)
                loss_sm = sm_zc + sm_zt + sm_zy
                # In NF-iVAE we do not produce loss_C_prior/loss_Y_prior via prior samples
                loss_C_prior = torch.tensor(0.0, device=x.device)
                loss_Y_prior = torch.tensor(0.0, device=x.device)

            else:
                # iVAE (Khemakhem et al.) simple conditional prior:
                # Only enforce Part ①: KL(q(z|x) || p(z|u)). Do NOT perform the Part ② prior-VAE updates.
                # Compute only posterior -> conditional prior KL terms
                if self.z_c_dim > 0 and mu_c is not None:
                    prior_mu_zc, prior_logvar_zc = prior_params['zc']
                    kl_zc = self.kl_gaussian(mu_c, logvar_c,
                                              prior_mu_zc, prior_logvar_zc) / B

                if self.t_latent_dim > 0 and mu_t is not None:
                    prior_mu_zt, prior_logvar_zt = prior_params['zt']
                    kl_zt = self.kl_gaussian(mu_t, logvar_t,
                                              prior_mu_zt, prior_logvar_zt) / B

                if self.y_latent_dim > 0 and mu_y is not None:
                    prior_mu_zy, prior_logvar_zy = prior_params['zy']
                    kl_zy = self.kl_gaussian(mu_y, logvar_y,
                                              prior_mu_zy, prior_logvar_zy) / B

                if self.style_latent_dim > 0 and mu_x is not None:
                    kl_zx = self.kl_gaussian(mu_x, logvar_x, None, None) / B
                # In iVAE mode we do NOT compute kl_prior_* or loss_C_prior/loss_Y_prior
        else:
            # Standard VAE: all partitions use N(0,I) prior
            if mu_c is not None:
                kl_zc = self.kl_gaussian(mu_c, logvar_c, None, None) / B
            if mu_t is not None:
                kl_zt = self.kl_gaussian(mu_t, logvar_t, None, None) / B
            if mu_y is not None:
                kl_zy = self.kl_gaussian(mu_y, logvar_y, None, None) / B
            if mu_x is not None:
                kl_zx = self.kl_gaussian(mu_x, logvar_x, None, None) / B
        
        kl_weight = self.kl_weight()
        
        # Part ① KL: posterior → conditional prior (partition-specific)
        kl_part1 = kl_weight * (kl_zc + kl_zt + kl_zy + kl_zx) * self.beta
        
        # Part ② KL: conditional prior → standard normal
        kl_part2 = kl_weight * (kl_prior_zc + kl_prior_zt + kl_prior_zy)
        
        kl_total = kl_part1 + kl_part2
        total_loss += kl_total

        # Add score-matching loss for NF-iVAE priors (zero when unused)
        total_loss = total_loss + kl_weight * loss_sm
        
        # Part ② reconstruction from prior samples (trains prior encoder + C,Y decoders)
        if self.conditional_prior:
            total_loss = total_loss + loss_C_prior + loss_Y_prior

        # image reconstruction loss - Part ① (trains main encoder + image decoder)
        if self.no_X:
            loss_X = torch.tensor(0.0, device=x.device)
        else:
            if self.mse_loss:
                loss_X = F.mse_loss(x_recon, x, reduction='sum') / B
            else:
                loss_X = self.gaussian_nll(x_recon, x) / B
        total_loss += loss_X

        # =====================================================================
        # NOTE on loss_C and loss_Y from posterior samples:
        # 
        # In pure IDVAE, Part ① only reconstructs X, and C,Y are reconstructed 
        # only in Part ② from prior samples.
        #
        # However, for causal inference we need accurate p(C|Zc,Zt) and p(Y|C,Zc,Zy)
        # decoders. Training them only on prior samples may be insufficient.
        #
        # Two modes:
        # - conditional_prior=True: Use IDVAE structure (loss_C_prior, loss_Y_prior)
        #   but ALSO add loss_C, loss_Y for better decoder training
        # - conditional_prior=False: Standard VAE, always use loss_C, loss_Y
        # =====================================================================

        # q(Y | Zy, Zc, C) from posterior samples
        y_logits = out["y_logits"]
        loss_Y = F.binary_cross_entropy_with_logits(y_logits, y, reduction='sum') / B
        
        # q(C | Zc, Zt) from posterior samples
        c_logits = out["c_logits"]
        if self.no_C:
            loss_C = torch.tensor(0.0, device=x.device)
        else:
            loss_C = F.binary_cross_entropy_with_logits(c_logits, c, reduction='sum') / B
        
        # In PURE IDVAE mode (pure_idvae=True):
        #   - Part ① only has p(X|z) reconstruction (no loss_C, loss_Y from posterior)
        #   - Part ② has p(C,Y|z) from prior samples only
        #   This matches the original IDVAE paper structure.
        #
        # In HYBRID mode (pure_idvae=False, default):
        #   - Add loss_C, loss_Y from posterior to help decoder training
        if not (self.conditional_prior and self.pure_idvae):
            total_loss += loss_C + loss_Y
        
        indep_loss = torch.tensor(0.0, device=x.device)
        
        if self.mim_weight > 0:
            if self.use_adversarial_independence:
                # Adversarial approach with unified loss:
                # 1. Adversary loss (positive): train predictor using detached latents
                # 2. Encoder loss (negative): train encoder to fool predictor
                
                # Use a Gradient Reversal Layer so we can train adversary and
                # encoder in one pass with a single optimizer. The GRL ensures
                # the predictor receives normal grads while the encoder sees
                # reversed (negated) gradients.
                if z_t is not None and hasattr(self, 'adv_zt_y') and self.adv_zt_y is not None:
                    y_pred = self.adv_zt_y(self.grl(z_t), c)
                    adv_loss = F.mse_loss(y_pred, y, reduction='mean')
                    total_loss = total_loss + self.mim_weight * adv_loss

                if z_y is not None and hasattr(self, 'adv_zy_c') and self.adv_zy_c is not None:
                    c_pred = self.adv_zy_c(self.grl(z_y))
                    adv_loss = F.binary_cross_entropy_with_logits(c_pred, c, reduction='mean')
                    total_loss = total_loss + self.mim_weight * adv_loss

                if z_x is not None and hasattr(self, 'adv_zx_c') and self.adv_zx_c is not None:
                    c_pred = self.adv_zx_c(self.grl(z_x))
                    adv_loss = F.binary_cross_entropy_with_logits(c_pred, c, reduction='mean')
                    total_loss = total_loss + self.mim_weight * adv_loss

                if z_x is not None and hasattr(self, 'adv_zx_y') and self.adv_zx_y is not None:
                    y_pred = self.adv_zx_y(self.grl(z_x))
                    adv_loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction='mean')
                    total_loss = total_loss + self.mim_weight * adv_loss
            else:
                # Compute baseline prediction E[Y|C] for residualization
                # This is used to test I(Zt; Y | C) = I(Zt; Y - E[Y|C])
                with torch.no_grad():
                    y_baseline_logits = self.baseline_y_from_c(c)
                    y_baseline_prob = torch.sigmoid(y_baseline_logits)
                # Also train the baseline predictor (but don't backprop through residual computation)
                y_baseline_logits_train = self.baseline_y_from_c(c)
                baseline_loss = F.binary_cross_entropy_with_logits(y_baseline_logits_train, y, reduction='mean')
                total_loss = total_loss + baseline_loss
                # Compute residual: Y - E[Y|C] (the part of Y not explained by C)
                y_residual = y - y_baseline_prob

                # MI-based approach using CLUB estimator
                # I(Zt; Y | C): estimate I(Zt; Y_residual) where Y_residual = Y - E[Y|C]
                # This correctly tests conditional independence via residualization
                if z_t is not None and self.mi_loss_fn_z_t is not None:
                    indep_loss = indep_loss + self.mim_weight * self.mi_loss_fn_z_t(z_t, y_residual)
                
                # I(Zy; C)
                if z_y is not None and self.mi_loss_fn_z_y is not None:
                    indep_loss = indep_loss + self.mim_weight * self.mi_loss_fn_z_y(z_y, c)
                
                # I(Zx; C)
                if z_x is not None and self.mi_loss_fn_zx_c is not None:
                    indep_loss = indep_loss + self.mim_weight * self.mi_loss_fn_zx_c(z_x, c)
                
                # I(Zx; Y)
                if z_x is not None and self.mi_loss_fn_zx_y is not None:
                    indep_loss = indep_loss + self.mim_weight * self.mi_loss_fn_zx_y(z_x, y)
        
        total_loss = total_loss + indep_loss

        # Initialize aux losses (used in logs even when use_aux=False)
        aux_c_loss = torch.tensor(0.0, device=x.device)
        aux_y_loss = torch.tensor(0.0, device=x.device)
        qc_logits = None
        qy_logits = None

        if self.use_aux:
            # q(C|X) logits
            qc_logits = self.aux_qc(x_feat)  # [B, k]
            aux_c_loss = (F.binary_cross_entropy_with_logits(qc_logits, c, reduction='sum') / B) * self.aux_weight_c
            # q(Y | X, C)
            qy_in = torch.cat([x_feat, c], dim=1)
            qy_logits = self.aux_qy(qy_in)
            aux_y_loss = (F.binary_cross_entropy_with_logits(qy_logits, y, reduction='sum') / B) * self.aux_weight_y
            out["qc_logits"] = qc_logits
            out["qy_logits"] = qy_logits
            
            aux_loss = aux_c_loss + aux_y_loss
            total_loss += aux_loss

        logs = {
            f"{stage}_loss": total_loss,
            f"{stage}_loss_X": loss_X,
            f"{stage}_loss_C": loss_C,  # Part ① p(C|z) from main encoder samples
            f"{stage}_loss_Y": loss_Y,  # Part ① p(Y|z,C) from main encoder samples
            f"{stage}_loss_C_prior": loss_C_prior,  # Part ② p(C|z) from prior samples (IDVAE)
            f"{stage}_loss_Y_prior": loss_Y_prior,  # Part ② p(Y|z,C) from prior samples (IDVAE)
            f"{stage}_loss_sm": loss_sm,
            f"{stage}_kl": kl_total,
            f"{stage}_kl_part1": kl_part1,  # Part ① KL(q(z|x,u) || p(z|u))
            f"{stage}_kl_part2": kl_part2,  # Part ② KL(p(z|u) || N(0,I))
            # Partition-specific KL terms (Part ① posterior → conditional prior)
            f"{stage}_kl_zc": kl_zc,  # KL(q(Zc|x,C,Y) || p(Zc|C,Y))
            f"{stage}_kl_zt": kl_zt,  # KL(q(Zt|x,C,Y) || p(Zt|C)) → enforces Zt ⊥ Y | C
            f"{stage}_kl_zy": kl_zy,  # KL(q(Zy|x,C,Y) || p(Zy|Y)) → enforces Zy ⊥ C
            f"{stage}_kl_zx": kl_zx,  # KL(q(Zx|x,C,Y) || N(0,I)) → enforces Zx ⊥ (C,Y)
            # Part ② KL terms (conditional prior → standard normal)
            f"{stage}_kl_prior_zc": kl_prior_zc,  # KL(p(Zc|C,Y) || N(0,I))
            f"{stage}_kl_prior_zt": kl_prior_zt,  # KL(p(Zt|C) || N(0,I))
            f"{stage}_kl_prior_zy": kl_prior_zy,  # KL(p(Zy|Y) || N(0,I))
            f"{stage}_aux_c": aux_c_loss,
            f"{stage}_aux_y": aux_y_loss,
            f"{stage}_indep_loss": indep_loss,
        }

        if stage == 'test':
            y_acc = self.accuracy(torch.sigmoid(y_logits), y.long().view(-1, 1))
            y_aux_acc = self.accuracy(qy_logits, y.long().view(-1, 1)) if self.use_aux else torch.tensor(0.0)
            
            c_acc = torch.stack([
                self.accuracy(c_hat.view(-1,1), c.view(-1,1).long())
                for c, c_hat in zip(c.T, c_logits.T)
            ]).mean() if not self.no_C else torch.tensor(0.0)
            c_aux_acc = torch.stack([
                self.accuracy(c_hat.view(-1,1), c.view(-1,1).long())
                for c, c_hat in zip(c.T, qc_logits.T)
            ]).mean() if self.use_aux else torch.tensor(0.0)
           
            logs.update({
                f"{stage} y accuracy": y_acc,
                f"{stage} c accuracy": c_acc,
                f"{stage} aux y accuracy": y_aux_acc,
                f"{stage} aux c accuracy": c_aux_acc,
            })
        return {"loss": total_loss, "logs": logs}
    
    # --------------------- inference ---------------------
    def infer_latents(self, x, c=None, y=None, binarize=True, x_embedding=False):
        """
        Run inference pipeline:
        Returns dict with raw + (optional) binarized outputs
        """
        x_feat = x if x_embedding else self.img_encoder(x)  # [B, feat_dim]

        if getattr(self, "use_aux", False):
            qc_logits = self.aux_qc(x_feat)
            c_hat_prob = torch.sigmoid(qc_logits)
            c_hat = torch.bernoulli(c_hat_prob) if binarize else c_hat_prob

            qy_in = torch.cat([x_feat, c_hat], dim=1)
            qy_logits = self.aux_qy(qy_in)
            y_hat_prob = torch.sigmoid(qy_logits)
            y_hat = torch.bernoulli(y_hat_prob) if binarize else y_hat_prob
        else:
            print("WARNING: using ground-truth c,y for inference!")
            c_hat = c.float()
            y_hat = y.float()
            c_hat_prob = c_hat
            y_hat_prob = y_hat
            if y_hat.dim() == 1:
                y_hat = y_hat.unsqueeze(1)

        z_c, z_t, z_y, z_x, z, mu_c, logvar_c, mu_t, logvar_t, mu_y, logvar_y, mu_x, logvar_x, mu, logvar = self._post_params(x_feat, c_hat, y_hat)

        if self.L > 0:
            z_c_chunks = self._split_latents(z_c)  # list of [B,L]
            z_t_chunks = self._split_latents(z_t)  # list of [B,L]
        else:
            z_c_chunks = []
            z_t_chunks = []

        out = {
            "c_hat": c_hat,
            "c_hat_prob": c_hat_prob,
            "y_hat": y_hat,
            "y_hat_prob": y_hat_prob,
            "z_c": z_c,
            "z_t": z_t,
            "z_y": z_y,
            "z_x": z_x,
            "z_c_chunks": z_c_chunks,
            "z_t_chunks": z_t_chunks,
            "z": z,
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
        predict_dict = self.infer_latents(x, c, y, binarize=True)
        predict_dict.update({
            "y": y,
            "c": c,
            "shortcuts": shortcuts
        })
        return predict_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # # --------------------- ATE estimation helper ---------------------
    def intervene(self, concepts: torch.Tensor, z_c_chunks, z_t_chunks, concept_idx: int, intervene_value: float = 1.0, binarize: bool = True):
        """
        Iteratively update `concepts` in topological order using the causal graph.

        Args:
            concepts: Tensor [B, k] of current concept values (0/1 or probabilities).
            z_c_chunks: list of tensors representing chunks of z_c latents.
            z_t_chunks: list of tensors representing chunks of z_t latents.
            concept_idx: index of the concept to intervene on (set to `intervene_value`).
            intervene_value: value to set at intervention (default 1.0).
            binarize: if True, threshold probabilities at 0.5 to produce binary concepts.

        Returns:
            Tensor [B, k] of updated concepts after iterative interventions.
        """
        device = concepts.device

        c_curr = concepts.clone().to(device)

        # iterate in topological order (assume concepts already topo-ordered)
        for i in range(concept_idx, self.k):
            # intervene on the target concept
            if i == concept_idx and intervene_value is not None:
                c_curr[:, i] = intervene_value
                continue

            # compute conditional probability p(ci | pa(ci), z_i)
            parts = []
            if z_c_chunks[i] is not None:
                parts.append(z_c_chunks[i])
            if z_t_chunks[i] is not None:
                parts.append(z_t_chunks[i])
            # include parents
            for p in (self.causal_parents[i] if self.causal_parents is not None else []):
                parts.append(c_curr[:, p].unsqueeze(-1))

            inp = self._combine_latents(*parts)
            logit = self.treat_heads[i](inp).squeeze(-1)
            p_i = torch.sigmoid(logit)

            c_curr[:, i] = torch.bernoulli(p_i) if binarize else p_i

        return c_curr

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
        seed=42,
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
            y = attr[:, self.indices['task']].to(device)
            c = attr[:, self.indices['concepts']].to(device)

            x_feat = self.img_encoder(x)
            out = self.infer_latents(x_feat, c, y, binarize=True, x_embedding=True)
            c_hat_fixed = out["c_hat"].clone().to(device)                      # (B, C_concepts)
            y_hat_fixed = out["y_hat"].clone().to(device)                      # (B,)

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

            # ATE group-level by conditioning
            if (c_hat_fixed[:, concept_idx] == 1).any():
                y_c1.append(y_hat_fixed[c_hat_fixed[:, concept_idx] == 1].mean().item())
            if (c_hat_fixed[:, concept_idx] == 0).any():
                y_c0.append(y_hat_fixed[c_hat_fixed[:, concept_idx] == 0].mean().item())

            # collect ground-truth concepts for true ATE/PEHE
            if coeffs is not None:
                # c_full_batch = torch.cat([attr[:, self.indices['shortcut']], attr[:, self.indices['concepts']]], dim=1)
                all_C.append(X_all)

            # per-batch containers to collect MC draws (shapes: S x B)
            y1_draws = []
            y0_draws = []

            # Monte Carlo draws per batch (for adjusted estimator)
            for _ in range(num_samples):
                if getattr(self, "use_aux", False):
                    qc_logits = self.aux_qc(x_feat)
                    c_hat_prob = torch.sigmoid(qc_logits)
                    c_hat = torch.bernoulli(c_hat_prob)
                    qy_in = torch.cat([x_feat, c_hat], dim=1)
                    qy_logits = self.aux_qy(qy_in)
                    y_hat_prob = torch.sigmoid(qy_logits)
                    y_hat = torch.bernoulli(y_hat_prob)
                else:
                    # use fixed c_hat, y_hat from above
                    c_hat = c
                    y_hat = y

                # Use posterior samples q(z|X,C,Y)
                z_c, z_t, z_y, z_x, z, mu_c, logvar_c, mu_t, logvar_t, mu_y, logvar_y, mu_x, logvar_x, mu, logvar = self._post_params(x_feat, c_hat, y_hat)

                z_adjust = self._combine_latents(z_c, z_y)
                z_adjust = z_adjust.to(device)
                
                if not self.no_C and self.marginalize_c:
                    if self.L > 0:
                        z_c_chunks = self._split_latents(z_c)  # list of [B,L]
                        z_t_chunks = self._split_latents(z_t)  # list of [B,L]
                        c1 = self.intervene(c_hat.clone(), z_c_chunks, z_t_chunks, concept_idx, intervene_value=1.0, binarize=True)
                        c0 = self.intervene(c_hat.clone(), z_c_chunks, z_t_chunks, concept_idx, intervene_value=0.0, binarize=True)
                    elif self.causal_parents is not None and any(len(parents) > 0 for parents in self.causal_parents):
                        c1 = self.intervene(c_hat.clone(), [z_c for _ in range(self.k)], [z_t for _ in range(self.k)], concept_idx, intervene_value=1.0, binarize=True)
                        c0 = self.intervene(c_hat.clone(), [z_c for _ in range(self.k)], [z_t for _ in range(self.k)], concept_idx, intervene_value=0.0, binarize=True)                
                    else:
                        c_logits = self.shared_treat_head(self._combine_latents(z_c, z_t))
                        c_hat_prob = torch.sigmoid(c_logits)
                        c_hat = torch.bernoulli(c_hat_prob)
                        c1 = c_hat.clone(); c1[:, concept_idx] = 1.0
                        c0 = c_hat.clone(); c0[:, concept_idx] = 0.0
                else:
                    c1 = c_hat.clone(); c1[:, concept_idx] = 1.0
                    c0 = c_hat.clone(); c0[:, concept_idx] = 0.0                
                
                y1_logit = self.y_head(torch.cat([c1, z_adjust], dim=1))
                y0_logit = self.y_head(torch.cat([c0, z_adjust], dim=1))
                y1_draws.append(torch.sigmoid(y1_logit).cpu().numpy())  # shape (B,)
                y0_draws.append(torch.sigmoid(y0_logit).cpu().numpy())

            # after MC draws: stack S x B -> compute mean over S -> per-sample estimate for batch
            y1_draws = np.stack(y1_draws, axis=0)  # (S, B)
            y0_draws = np.stack(y0_draws, axis=0)  # (S, B)
            y1_mean_per_sample = y1_draws.mean(axis=0)  # (B,)
            y0_mean_per_sample = y0_draws.mean(axis=0)

            per_sample_y1_adj.append(y1_mean_per_sample)
            per_sample_y0_adj.append(y0_mean_per_sample)

        # --- aggregate ATE as before
        y1_all = np.concatenate(per_sample_y1_adj, axis=0)
        y0_all = np.concatenate(per_sample_y0_adj, axis=0)
        ate = y1_all.mean() - y0_all.mean()
        # ate_per_batch = np.array([np.mean(y1 - y0) for y1, y0 in zip(per_sample_y1_adj, per_sample_y0_adj)])
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
            C_full = np.concatenate(all_C, axis=0)   # (N, C_total)
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
            # if no ground-truth coefficients, use pseudo-oracle as pseudo truth
            if per_sample_y1_po:
                print("WARNING: using pseudo-oracle as proxy for true ITE/ATE!")
                ite_true = ite_po
                ate_true = ate_po

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

        # Bootstrap only if naive is available
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
    
    # # # --------------------- Counterfactual Visualization ---------------------

    @torch.no_grad()
    def create_counterfactuals(
        self,
        dataloader,
        out_dir: str,
        concept_indices=None,
        num_examples: int = 10,
        concept_names=None,
        device: str = "cuda",
        max_dims: Optional[int] = None,
    ):
        """
        Counterfactual grids by intervening on concepts (first row) and negating latent dimensions (subsequent rows).
        For concept interventions, propagate using the same logic as ATE estimation (via intervene method).
        """
        self.eval()
        self.to(device)

        if self.no_X or self.x_decoder is None:
            print("Cannot create counterfactuals: no_X=True or x_decoder is None.")
            return

        os.makedirs(out_dir, exist_ok=True)

        if concept_indices is None:
            concept_indices = list(range(self.k))
        if concept_names is None:
            concept_names = [f"c{i}" for i in range(self.k)]

        def to_np(img_t):
            if self.channels == 1:
                return img_t.squeeze(0).squeeze().cpu().numpy()
            else:
                return img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()

        batch = next(iter(dataloader))
        x = batch[0][:num_examples].to(device)
        attr = batch[1][:num_examples]
        c = attr[:, self.indices['concepts']].to(device)
        y = attr[:, self.indices['task']].to(device)

        out = self.infer_latents(x, c, y, binarize=True)
        c_hat = out["c_hat"]
        y_hat = out["y_hat"]
        z_c_all, z_t_all, z_y_all, z_o_all = out["z_c"], out["z_t"], out["z_y"], out["z_x"]

        for i in range(num_examples):
            img_factual = x[i:i+1]
            c0 = c_hat[i:i+1].clone()
            y0 = y_hat[i:i+1].clone()

            z_c = z_c_all[i:i+1] if z_c_all is not None else None
            z_t = z_t_all[i:i+1] if z_t_all is not None else None
            z_y = z_y_all[i:i+1] if z_y_all is not None else None
            z_x = z_o_all[i:i+1] if z_o_all is not None else None
            z = out["z"][i:i+1]

            fwd_out = self.forward(img_factual, c0, y0)
            x_recon = fwd_out["x_recon"].clamp(0, 1)

            groups = []
            groups.append(("concepts", c0, self.k))
            if self.z_c_dim > 0:
                groups.append(("z_c", z_c, self.z_c_dim))
            if self.t_latent_dim > 0:
                groups.append(("z_t", z_t, self.t_latent_dim))
            if self.y_latent_dim > 0:
                groups.append(("z_y", z_y, self.y_latent_dim))
            if self.style_latent_dim > 0:
                groups.append(("z_x", z_x, self.style_latent_dim))

            dims_list = [g[2] for g in groups]
            if max_dims is not None:
                dims_list_capped = [min(d, max_dims) for d in dims_list]
            else:
                dims_list_capped = dims_list
            max_cols_needed = max(dims_list_capped)
            ncols = 1 + max_cols_needed
            nrows = len(groups)

            fig, axes = plt.subplots(
                nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows)
            )
            if nrows == 1:
                axes = np.expand_dims(axes, 0)
            if ncols == 1:
                axes = np.expand_dims(axes, 1)

            baseline_np = to_np(x_recon)

            for row_idx, (gname, z_vec, gdim) in enumerate(groups):
                if row_idx == 0:
                    axes[row_idx, 0].imshow(to_np(img_factual), cmap="gray" if self.channels == 1 else None)
                    axes[row_idx, 0].set_title(f"Factual\ny={y0.item():.2f}")
                else:
                    axes[row_idx, 0].imshow(baseline_np, cmap="gray" if self.channels == 1 else None)
                    axes[row_idx, 0].set_title(f"Reconstruction\ny={y0.item():.2f}")
                axes[row_idx, 0].axis("off")

                cols_to_show = min(gdim, max_dims) if max_dims is not None else gdim

                for dim_idx in range(cols_to_show):
                    z_c_cf = z_c.clone() if z_c is not None else None
                    z_t_cf = z_t.clone() if z_t is not None else None
                    z_y_cf = z_y.clone() if z_y is not None else None
                    z_o_cf = z_x.clone() if z_x is not None else None
                    c_cf = c0.clone()
                    y_cf = y0.clone()

                    if gname == "concepts":
                        # Use intervene method to propagate intervention on concept dim_idx
                        if self.L > 0:
                            z_c_chunks_cf = self._split_latents(z_c_cf)
                            z_t_chunks_cf = self._split_latents(z_t_cf)
                            # Set the target concept to 1 - factual value, propagate downstream
                            c_cf = self.intervene(
                                c_cf,
                                z_c_chunks_cf,
                                z_t_chunks_cf,
                                concept_idx=dim_idx,
                                intervene_value=1.0 - c_cf[:, dim_idx].item(),
                                binarize=True
                            )
                        elif self.causal_parents is not None and any(len(parents) > 0 for parents in self.causal_parents):
                            z_c_chunks_cf = [z_c_cf for _ in range(self.k)]
                            z_t_chunks_cf = [z_t_cf for _ in range(self.k)]
                            c_cf = self.intervene(
                                c_cf,
                                z_c_chunks_cf,
                                z_t_chunks_cf,
                                concept_idx=dim_idx,
                                intervene_value=1.0 - c_cf[:, dim_idx].item(),
                                binarize=True
                            )
                        else:
                            c_cf[:, dim_idx] = 1.0 - c_cf[:, dim_idx]
                        intervention_label = f"{concept_names[dim_idx]} flip"
                        z_c_cf, z_t_cf, z_y_cf, z_o_cf, _, _, _, _, _, _, _, _, _, _, _ = self._post_params(fwd_out["x_feat"], c_cf, y0)
                    elif gname == "z_c":
                        z_c_cf[:, dim_idx] = -z_c_cf[:, dim_idx]
                        intervention_label = f"z_c[{dim_idx}] flip"
                        # For per-concept latents, propagate concepts as in ATE estimation
                        if self.L > 0:
                            z_c_chunks_cf = self._split_latents(z_c_cf)
                            z_t_chunks_cf = self._split_latents(z_t_cf)
                            # propagate concepts with factual values (no intervention)
                            c_cf = self.intervene(
                                c_cf,
                                z_c_chunks_cf,
                                z_t_chunks_cf,
                                concept_idx=0,
                                intervene_value=None,
                                binarize=True
                            )
                    elif gname == "z_t":
                        z_t_cf[:, dim_idx] = -z_t_cf[:, dim_idx]
                        intervention_label = f"z_t[{dim_idx}] flip"
                        if self.L > 0:
                            z_c_chunks_cf = self._split_latents(z_c_cf)
                            z_t_chunks_cf = self._split_latents(z_t_cf)
                            c_cf = self.intervene(
                                c_cf,
                                z_c_chunks_cf,
                                z_t_chunks_cf,
                                concept_idx=0,
                                intervene_value=None,
                                binarize=True
                            )
                    elif gname == "z_y":
                        z_y_cf[:, dim_idx] = -z_y_cf[:, dim_idx]
                        intervention_label = f"z_y[{dim_idx}] flip"
                    elif gname == "z_x":
                        z_o_cf[:, dim_idx] = -z_o_cf[:, dim_idx]
                        intervention_label = f"z_x[{dim_idx}] flip"
                    else:
                        intervention_label = f"{gname}[{dim_idx}]"

                    if gname != "concepts":
                        if self.L == 0 and not self.no_C and (self.causal_parents is None or all(len(parents) == 0 for parents in self.causal_parents)):
                            c_logits_cf = self.shared_treat_head(self._combine_latents(z_c_cf, z_t_cf))
                            c_cf = (torch.sigmoid(c_logits_cf) > 0.5).float()

                    y_in_cf = self._combine_latents(c_cf, z_c_cf, z_y_cf)
                    y_cf_logit = self.y_head(y_in_cf)
                    y_cf = torch.sigmoid(y_cf_logit).item()

                    z_all_cf = self._combine_latents(z_c_cf, z_t_cf, z_y_cf, z_o_cf)
                    decoder_input_cf = z_all_cf
                    if self.x_on_c:
                        decoder_input_cf = torch.cat((decoder_input_cf, c_cf), dim=1)
                    if self.x_on_y:
                        y_cf_tensor = torch.tensor([[y_cf]], device=device)
                        decoder_input_cf = torch.cat((decoder_input_cf, y_cf_tensor), dim=1)
                    x_cf_feat = self.x_decoder(decoder_input_cf)
                    x_cf = self.decode_image(x_cf_feat, decoder_input_cf).clamp(0, 1)

                    x_cf_np = to_np(x_cf)
                    col_idx = 1 + dim_idx
                    axes[row_idx, col_idx].imshow(x_cf_np, cmap="gray" if self.channels == 1 else None)
                    axes[row_idx, col_idx].set_title(f"{intervention_label}\ny={y_cf:.2f}")
                    axes[row_idx, col_idx].axis("off")

                for empty_col in range(1 + cols_to_show, ncols):
                    axes[row_idx, empty_col].axis("off")

            plt.tight_layout()
            save_path = os.path.join(out_dir, f"cf_example_{i}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)


    def on_train_batch_start(self, batch, batch_idx):
        """Update MI estimators (only used for MI-based independence, not adversarial)."""
        if self.mim_weight > 0 and not self.use_adversarial_independence:
            x, attr = batch
            y = attr[:, self.indices['task']]
            c = attr[:, self.indices['concepts']]
            if y.dim() == 1:
                y = y.unsqueeze(1)

            with torch.no_grad():
                out = self.forward(x, c, y)
            
            z_t = out["z_t"]
            z_y = out["z_y"]
            z_x = out["z_x"]
            
            # MI-based approach: update MI estimators
            # I(Zt; Y | C) - condition on C by concatenating
            if z_t is not None and self.mi_loss_fn_z_t is not None:
                with torch.no_grad():
                    y_baseline_logits = self.baseline_y_from_c(c)
                    y_baseline_prob = torch.sigmoid(y_baseline_logits)
                    y_residual = y - y_baseline_prob
                mi_loss = self.mi_loss_fn_z_t.step(z_t, y_residual)
                mi_est = self.mi_loss_fn_z_t(z_t, y_residual)
                self.log("mi_estimator_loss_zt_y_given_c", mi_loss, on_step=False, on_epoch=True)
                self.log("mi_estimate_zt_y_given_c", mi_est.mean(), on_step=False, on_epoch=True)

            # I(Zy; C)
            if z_y is not None and self.mi_loss_fn_z_y is not None:
                mi_loss = self.mi_loss_fn_z_y.step(z_y, c)
                mi_est = self.mi_loss_fn_z_y(z_y, c)
                self.log("mi_estimator_loss_zy_c", mi_loss, on_step=False, on_epoch=True)
                self.log("mi_estimate_zy_c", mi_est.mean(), on_step=False, on_epoch=True)
            
            # I(Zx; C)
            if z_x is not None and self.mi_loss_fn_zx_c is not None:
                mi_loss = self.mi_loss_fn_zx_c.step(z_x, c)
                mi_est = self.mi_loss_fn_zx_c(z_x, c)
                self.log("mi_estimator_loss_zx_c", mi_loss, on_step=False, on_epoch=True)
                self.log("mi_estimate_zx_c", mi_est.mean(), on_step=False, on_epoch=True)
            
            # I(Zx; Y)
            if z_x is not None and self.mi_loss_fn_zx_y is not None:
                mi_loss = self.mi_loss_fn_zx_y.step(z_x, y)
                mi_est = self.mi_loss_fn_zx_y(z_x, y)
                self.log("mi_estimator_loss_zx_y", mi_loss, on_step=False, on_epoch=True)
                self.log("mi_estimate_zx_y", mi_est.mean(), on_step=False, on_epoch=True)
