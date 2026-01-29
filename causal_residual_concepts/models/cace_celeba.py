import torch
import torch.nn.functional as F
import torch.nn as nn

from .cace import CaCE
from .image_modules import CelebAEncoder as CelebAImageEncoder, CelebADecoder as ConditionalImageDecoder
from .utils import init_weights, DINOV2_EMBED_DIM


class CaCECelebA(CaCE):
    def __init__(
        self,
        num_concepts: int,
        feat_dim: int = 256,
        shared_latent_dim: int = 0,
        latent_per_concept: int = 1,
        style_latent_dim: int = 16,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        kl_anneal_start: int = 0,
        kl_anneal_end: int = 10,
        tau_anneal_start: float = 1.0,
        tau_anneal_min: float = 0.1,
        tau_anneal_decay: float = 0.05,
        use_aux: bool = True,
        aux_weight_c: float = 1.0,
        aux_weight_y: float = 1.0,
        indices=None,
        channels: int = 3,
        use_dinov2_embeddings: bool = False,
        **kwargs,
    ):
        self.channels = channels
        self.use_dinov2_embeddings = use_dinov2_embeddings

        feat_dim_override = DINOV2_EMBED_DIM if use_dinov2_embeddings else feat_dim

        # Precompute latent dims to size conditional decoder input later
        z_c_dim = (num_concepts * latent_per_concept if latent_per_concept > 0 else 0) + shared_latent_dim
        latent_dim = z_c_dim + style_latent_dim

        super().__init__(
            num_concepts=num_concepts,
            feat_dim=feat_dim_override,
            shared_latent_dim=shared_latent_dim,
            latent_per_concept=latent_per_concept,
            style_latent_dim=style_latent_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            kl_anneal_start=kl_anneal_start,
            kl_anneal_end=kl_anneal_end,
            tau_anneal_start=tau_anneal_start,
            tau_anneal_min=tau_anneal_min,
            tau_anneal_decay=tau_anneal_decay,
            use_aux=use_aux,
            aux_weight_c=aux_weight_c,
            aux_weight_y=aux_weight_y,
            indices=indices,
            channels=channels,
            **kwargs,
        )

        # Replace encoder/decoder with CelebA architectures
        self.img_encoder = nn.Identity() if use_dinov2_embeddings else CelebAImageEncoder(
            in_channels=self.channels, feat_dim=feat_dim_override)
        
        self.img_decoder = nn.Identity() if use_dinov2_embeddings else ConditionalImageDecoder(
            feat_dim=feat_dim,
            out_channels=self.channels,
            cond_dim=latent_dim + num_concepts,
        )

        # Init weights similar to UnCoVAErCelebA
        self.img_encoder.apply(init_weights)
        self.img_decoder.apply(init_weights)

    # Use conditional image decoder (requires decoder_input)
    def decode_image(
        self,
        x_recon_feat: torch.Tensor,
        decoder_input: torch.Tensor | None = None,
        z_all: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_dinov2_embeddings:
            return self.img_decoder(x_recon_feat)
        
        if decoder_input is None:
            # Fallback to zeros if not provided, but normally provided by runner/model
            if z_all is None or c is None:
                raise ValueError("decoder_input or (z_all, c) must be provided for CaCECelebA decoding")
            decoder_input = torch.cat([z_all, c], dim=1)
        return self.img_decoder(x_recon_feat, decoder_input)

    # Override Gaussian NLL to use MSE for CelebA images
    def gaussian_nll(self, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_recon, x, reduction='sum')
