"""
Image encoder and decoder modules for different datasets.

MorphoMNIST: 32x32 grayscale images
CelebA: 64x64 RGB images
"""

import torch
import torch.nn as nn


# =============================================================================
# MorphoMNIST Encoder / Decoder (32x32 grayscale)
# =============================================================================

class MorphoMNISTEncoder(nn.Module):
    """
    Encoder for MorphoMNIST (32x32 grayscale).
    Downsamples 32x32 -> 16x16 -> 8x8 -> 4x4, then FC to feat_dim.
    """
    def __init__(self, in_channels=1, feat_dim=128, resolution=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, resolution, 3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.LeakyReLU(),
            
            nn.Conv2d(resolution, resolution*2, 3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.LeakyReLU(),
            
            nn.Conv2d(resolution*2, resolution*4, 3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.LeakyReLU(),
            
            nn.Conv2d(resolution*4, resolution*4, 3, stride=1, padding=1),  # 4x4 -> 4x4
            nn.LeakyReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(resolution*4*4*4, feat_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class MorphoMNISTDecoder(nn.Module):
    """
    Decoder for MorphoMNIST (32x32 grayscale).
    FC to spatial, then upsample 4x4 -> 8x8 -> 16x16 -> 32x32.
    """
    def __init__(self, out_channels=1, feat_dim=128, resolution=32):
        super().__init__()
        self.fc = nn.Linear(feat_dim, resolution*4*4*4)
        self.unflatten = nn.Unflatten(1, (resolution*4, 4, 4))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(resolution*4, resolution*4, 3, stride=1, padding=1),  # 4x4
            nn.LeakyReLU(),

            nn.ConvTranspose2d(resolution*4, resolution*2, 3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            nn.LeakyReLU(),

            nn.ConvTranspose2d(resolution*2, resolution, 3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.LeakyReLU(),

            nn.ConvTranspose2d(resolution, out_channels, 3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        x = self.deconv(x)
        return x


# =============================================================================
# CelebA Encoder / Decoder (64x64 RGB)
# =============================================================================

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Applies affine transformation conditioned on an external input.
    """
    def __init__(self, in_channels, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, in_channels)
        self.beta = nn.Linear(cond_dim, in_channels)

    def forward(self, x, cond):
        # cond: [B, cond_dim]
        gamma = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class CelebAEncoder(nn.Module):
    """
    Encoder for CelebA (64x64 RGB).
    Downsamples 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4, then FC to feat_dim.
    """
    def __init__(self, in_channels=3, feat_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),   # 64 -> 32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),            # 32 -> 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),           # 16 -> 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),          # 8 -> 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 4 * 4, feat_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)


class CelebADecoder(nn.Module):
    """
    Decoder for CelebA (64x64 RGB) with FiLM conditioning.
    FC to spatial 4x4, then upsample 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64.
    Uses FiLM layers to condition on latent/concept information.
    """
    def __init__(self, feat_dim=256, base_channels=32, out_channels=3, cond_dim=1):
        super().__init__()
        self.final_spatial = 4
        self.fc = nn.Linear(feat_dim, base_channels * 8 * self.final_spatial * self.final_spatial)
        self.unflatten = nn.Unflatten(1, (base_channels * 8, self.final_spatial, self.final_spatial))

        # Decoder channels: 256 -> 128 -> 64 -> 32
        chs = [base_channels * 8, base_channels * 4, base_channels * 2, base_channels]

        # Conv + FiLM layers
        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(chs[i], chs[i + 1], kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(),
                )
            )
            self.film_layers.append(FiLM(chs[i + 1], cond_dim))

        self.out_conv = nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z, cond):
        """
        Args:
            z: [B, feat_dim] latent features
            cond: [B, cond_dim] conditioning vector (e.g., concatenation of z_all and concepts)
        """
        h = self.fc(z)
        h = self.unflatten(h)

        # Apply FiLM conditioning after each block
        for layer, film in zip(self.layers, self.film_layers):
            h = layer(h)
            h = film(h, cond)

        return self.out_conv(h)


class CelebADecoderSimple(nn.Module):
    """
    Simple decoder for CelebA (64x64 RGB) without FiLM conditioning.
    Used when conditioning is not needed or when using embeddings directly.
    """
    def __init__(self, feat_dim=256, base_channels=32, out_channels=3):
        super().__init__()
        self.final_spatial = 4
        self.fc = nn.Linear(feat_dim, base_channels * 8 * self.final_spatial * self.final_spatial)
        self.unflatten = nn.Unflatten(1, (base_channels * 8, self.final_spatial, self.final_spatial))

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        h = self.fc(z)
        h = self.unflatten(h)
        return self.deconv(h)


# =============================================================================
# Factory functions for convenience
# =============================================================================

def get_morphomnist_modules(feat_dim=128, channels=1):
    """Return encoder and decoder for MorphoMNIST."""
    encoder = MorphoMNISTEncoder(in_channels=channels, feat_dim=feat_dim)
    decoder = MorphoMNISTDecoder(out_channels=channels, feat_dim=feat_dim)
    return encoder, decoder


def get_celeba_modules(feat_dim=256, channels=3, cond_dim=None, use_film=True):
    """
    Return encoder and decoder for CelebA.
    
    Args:
        feat_dim: Feature dimension output by encoder
        channels: Number of image channels (3 for RGB)
        cond_dim: Conditioning dimension for FiLM decoder (required if use_film=True)
        use_film: Whether to use FiLM conditioning in decoder
    """
    encoder = CelebAEncoder(in_channels=channels, feat_dim=feat_dim)
    if use_film:
        if cond_dim is None:
            raise ValueError("cond_dim required for FiLM decoder")
        decoder = CelebADecoder(feat_dim=feat_dim, out_channels=channels, cond_dim=cond_dim)
    else:
        decoder = CelebADecoderSimple(feat_dim=feat_dim, out_channels=channels)
    return encoder, decoder
