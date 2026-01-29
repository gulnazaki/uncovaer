"""
CelebA-specific UnCoVAEr variant.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .uncovaer import UnCoVAEr
from .image_modules import CelebAEncoder, CelebADecoder
from .utils import init_weights, DINOV2_EMBED_DIM


class UnCoVAErCelebA(UnCoVAEr):
    """
    CelebA-specific UnCoVAEr with:
    - CelebA image encoder/decoder (64x64 RGB)
    """
    def __init__(
        self,
        num_concepts: int,
        feat_dim: int = 256,
        shared_latent_dim: int = 0,
        latent_per_concept: int = 0,
        t_latent_dim: int = 0,
        y_latent_dim: int = 0,
        style_latent_dim: int = 16,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        kl_anneal_start: int = 0,
        kl_anneal_end: int = 10,
        use_aux: bool = True,
        aux_weight_c: float = 1.0,
        aux_weight_y: float = 1.0,
        indices=None,
        use_dinov2_embeddings: bool = False,
        channels: int = 3,
        x_on_c: bool = False,
        x_on_y: bool = False,
        **kwargs,
    ):
        self.use_dinov2_embeddings = use_dinov2_embeddings

        # Compute latent dimensions for conditioning
        if latent_per_concept > 0:
            z_c_dim = num_concepts * latent_per_concept
        else:
            z_c_dim = shared_latent_dim
        latent_dim = z_c_dim + t_latent_dim + y_latent_dim + style_latent_dim
        cond_dim = latent_dim
        if x_on_c:
            cond_dim += num_concepts  # z + c for FiLM
        if x_on_y:
            cond_dim += 1  # z + y for FiLM

        # Set up feat_dim based on embedding type
        feat_dim_actual = DINOV2_EMBED_DIM if use_dinov2_embeddings else feat_dim

        # Create encoder/decoder
        if use_dinov2_embeddings:
            encoder_module = nn.Identity()
            decoder_module = nn.Identity()
        else:
            encoder_module = CelebAEncoder(
                in_channels=channels,
                feat_dim=feat_dim_actual,
            )
            decoder_module = CelebADecoder(
                feat_dim=feat_dim_actual,
                out_channels=channels,
                cond_dim=cond_dim,
            )

        # Remove blocked keys from kwargs
        for blocked_key in ("img_encoder", "img_decoder", "x_decoder"):
            if blocked_key in kwargs:
                raise ValueError(
                    f"Cannot override '{blocked_key}' in UnCoVAErCelebA; handled internally."
                )

        super().__init__(
            num_concepts=num_concepts,
            feat_dim=feat_dim_actual,
            shared_latent_dim=shared_latent_dim,
            latent_per_concept=latent_per_concept,
            t_latent_dim=t_latent_dim,
            y_latent_dim=y_latent_dim,
            style_latent_dim=style_latent_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            kl_anneal_start=kl_anneal_start,
            kl_anneal_end=kl_anneal_end,
            use_aux=use_aux,
            aux_weight_c=aux_weight_c,
            aux_weight_y=aux_weight_y,
            indices=indices,
            img_encoder=encoder_module,
            img_decoder=decoder_module,
            channels=channels,
            x_on_c=x_on_c,
            x_on_y=x_on_y,
            **kwargs,
        )

        self.save_hyperparameters({
            "use_dinov2_embeddings": use_dinov2_embeddings,
        })

        # Initialize weights
        if not use_dinov2_embeddings:
            self.img_encoder.apply(init_weights)
            self.img_decoder.apply(init_weights)

    def decode_image(
        self,
        x_recon_feat: torch.Tensor,
        decoder_input: torch.Tensor
    ) -> torch.Tensor:
        """Decode with FiLM conditioning for CelebA."""
        if self.use_dinov2_embeddings:
            return self.img_decoder(x_recon_feat)
        # Pass both features and conditioning to FiLM decoder
        return self.img_decoder(x_recon_feat, decoder_input)
    