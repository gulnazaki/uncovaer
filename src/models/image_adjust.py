import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import numpy as np
import pytorch_lightning as pl
from .utils import detect_confounding, pehe, compute_true_ite_ate
from .image_modules import MorphoMNISTEncoder as ImageEncoder


class OutcomeModel(pl.LightningModule):
    def __init__(self, feat_dim, num_concepts, indices, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.indices = indices
        self.num_concepts = num_concepts

        # shared encoder (optional: you can use separate encoders)
        self.encoder = ImageEncoder(feat_dim=feat_dim)

        # classifiers: one per concept
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + num_concepts, feat_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(feat_dim // 2, 1)
            )

        # metrics
        self.accuracy = Accuracy(task='binary')

    def forward(self, x, c):
        """
        x: (B, C, H, W)
        c: (B, num_concepts) - all concepts as conditioning input
        return: (B,) - y logits
        """
        h = self.encoder(x)
        h = torch.cat([h, c], dim=1)
        return self.classifier(h).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, attr = batch
        y = attr[:, self.indices['task']].float()
        c = attr[:, self.indices['concepts']].float()
        y_hat = self.forward(x, c)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, attr = batch
        y = attr[:, self.indices['task']].float()
        c = attr[:, self.indices['concepts']].float()
        y_hat = self.forward(x, c)
        val_loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)
        val_acc = self.accuracy(torch.sigmoid(y_hat), y.int())
        self.log("val_acc", val_acc, prog_bar=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, attr = batch
        y = attr[:, self.indices['task']].float()
        c = attr[:, self.indices['concepts']].float()
        y_hat = self.forward(x, c)
        test_acc = self.accuracy(torch.sigmoid(y_hat), y.int())
        self.log("test_acc", test_acc, prog_bar=True)
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
        Compute ATE for iamge adjustment model.
        Also returns abs_diff, ate_error, and a confounded flag.
        """
        self.eval()
        self.to(device)

        y1_naive_list, y0_naive_list = [], []
        y1_po_list, y0_po_list = [], []
        y1_adj_list, y0_adj_list = [], []
        obs_t_list, obs_y_list, prop_list, all_C = [], [], [], []
        pehe_naive_pred = None

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

            # collect ground-truth concepts for true ATE/PEHE
            if coeffs is not None:
                # c_full_batch = torch.cat([attr[:, self.indices['shortcut']], attr[:, self.indices['concepts']]], dim=1)
                all_C.append(X_all)

            # outcome-adjustment (intervene on concept and predict with this model)
            # build conditioning concept vectors and predict y under intervention
            c_tensor = attr[:, self.indices['concepts']].float().to(device)
            c1 = c_tensor.clone()
            c0 = c_tensor.clone()
            c1[:, concept_idx] = 1.0
            c0[:, concept_idx] = 0.0
            y1_adj = torch.sigmoid(self.forward(x, c1)).cpu().numpy()
            y0_adj = torch.sigmoid(self.forward(x, c0)).cpu().numpy()
            y1_adj_list.append(y1_adj)
            y0_adj_list.append(y0_adj)

        # --- aggregate naive S-learner
        ite_naive = np.concatenate(y1_naive_list) - np.concatenate(y0_naive_list) if y1_naive_list else None
        ate_naive = float(ite_naive.mean()) if ite_naive is not None else None

        # --- aggregate pseudo-oracle S-learner
        ite_po = np.concatenate(y1_po_list) - np.concatenate(y0_po_list) if y1_po_list else None
        ate_po = float(ite_po.mean()) if ite_po is not None else None

        # --- aggregate outcome adjustment (S-learner using this outcome model)
        ite_adj = np.concatenate(y1_adj_list) - np.concatenate(y0_adj_list) if y1_adj_list else None
        ate_adj = float(ite_adj.mean()) if ite_adj is not None else None
        

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
            pehe_adj = pehe(ite_adj, ite_true) if ite_adj is not None else None
        else:
            ate_true = None
            pehe_naive = None
            pehe_po = None
            pehe_adj = None

        # --- diagnostics
        abs_diff_adj = abs((ate_adj if ate_adj is not None else 0) - (ate_naive if ate_naive is not None else 0))
        ate_error_adj = abs((ate_adj if ate_adj is not None else 0) - (ate_true if ate_true is not None else 0)) if ate_true is not None else None

        ate_error_naive = abs((ate_naive if ate_naive is not None else 0) - (ate_true if ate_true is not None else 0)) if ate_true is not None else None
        ate_error_pseudo_oracle = abs((ate_po if ate_po is not None else 0) - (ate_true if ate_true is not None else 0)) if ate_true is not None else None

        results = {
            "ate_naive_obs": ate_naive,        # original, uses attr (ground truth)
            "ate_pseudo_oracle": ate_po,
            "ate_adj": ate_adj,
            "ate_true": ate_true,
            "pehe_naive_obs": pehe_naive,
            "pehe_pseudo_oracle": pehe_po,
            "pehe_adj": pehe_adj,
            "abs_diff_adj": abs_diff_adj,
            "ate_error_adj": ate_error_adj,
            "ate_error_naive": ate_error_naive,
            "ate_error_pseudo_oracle": ate_error_pseudo_oracle
        }

        # # Confounding detection via bootstrap requires naive model
        # if ite_naive is not None:
        #     n_boot = n_boot
        #     rng = np.random.default_rng(seed)
        #     n = len(T)

        #     ate_boot = np.empty(n_boot)
        #     ate_naive_boot = np.empty(n_boot)
        #     for b in range(n_boot):
        #         idxs = rng.choice(n, size=n, replace=True)
        #         T_b = T[idxs]
        #         Y_b = Y[idxs]
        #         e_b = e[idxs]
        #         # recompute stabilized weights on bootstrap sample
        #         p_b = T_b.mean()
        #         # clip to avoid extreme ratios
        #         e_b = np.clip(e_b, 1e-6, 1 - 1e-6)
        #         w_b = T_b * p_b / e_b + (1 - T_b) * (1 - p_b) / (1 - e_b)
        #         num1 = np.sum(w_b * Y_b * T_b)
        #         den1 = np.sum(w_b * T_b)
        #         num0 = np.sum(w_b * Y_b * (1 - T_b))
        #         den0 = np.sum(w_b * (1 - T_b))
        #         ate_boot[b] = (num1 / den1) - (num0 / den0)
        #         ate_naive_boot[b] = ite_naive[idxs].mean()

        #     confounded = detect_confounding(ate_boot, ate_naive_boot)
        # else:
        #     raise ValueError("Naive model is required for confounding detection.")

        # results["confounded_flag_adj"] = int(confounded)

        return results