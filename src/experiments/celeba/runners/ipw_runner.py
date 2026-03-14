import os
import pickle
import sys
from typing import List

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from torchvision.transforms import RandomHorizontalFlip

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from datasets.celeba.dataset import get_dataloader
from datasets.celeba.config import ATTRIBUTES, CAUSAL_CONCEPTS, COEFFICIENTS, COEFFICIENTS_OOD, TASK
from models.ipw import PropensityModel
from models.image_modules import CelebAEncoder as CelebAImageEncoder
from models.utils import (
    generate_checkpoint_callback,
    generate_early_stopping_callback,
    generate_ema_callback,
    init_weights,
    DINOV2_EMBED_DIM
)
import joblib


class Dinov2Projector(nn.Module):
    """Simple linear projector for DINOv2 embeddings -> IPW feature space."""

    def __init__(self, input_dim: int, feat_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = torch.flatten(x, start_dim=1)
        return self.fc(x)


def get_callbacks(config, name):
    callbacks = [
        generate_checkpoint_callback(name, config["ckpt_path"], monitor="val_loss", mode="min"),
        generate_early_stopping_callback(
            patience=config["patience"], monitor="val_loss", mode="min", min_delta=1e-4
        ),
    ]
    if config.get("ema", False):
        callbacks.append(generate_ema_callback(decay=0.999))
    return callbacks


def _build_run_name(config, seed):
    attr = config.get("y", TASK)
    return f"{attr}_ipw_{config['feat_dim']}_{seed}"


def _build_shortcuts(config, attributes: List[str]):
    shortcuts = config.get("shortcuts")
    if shortcuts is None:
        shortcuts = [c for c in CAUSAL_CONCEPTS if c not in attributes]
    return shortcuts or None


def _train_transforms(config):
    return RandomHorizontalFlip(0.5) if config.get("use_augmentation", False) else None


def _attach_encoder(
    model: PropensityModel,
    feat_dim: int,
    channels: int,
    use_dinov2_embeddings: bool,
    dinov2_dim: int,
):
    if use_dinov2_embeddings:
        encoder = Dinov2Projector(input_dim=dinov2_dim, feat_dim=feat_dim)
    else:
        encoder = CelebAImageEncoder(in_channels=channels, feat_dim=feat_dim)
        encoder.apply(init_weights)
    model.encoder = encoder


def _init_model(config, num_concepts, indices):
    model = PropensityModel(
        num_concepts=num_concepts,
        feat_dim=config["feat_dim"],
        lr=float(config["lr"]),
        indices=indices,
    )
    _attach_encoder(
        model,
        feat_dim=config["feat_dim"],
        channels=config.get("channels", 3),
        use_dinov2_embeddings=config.get("use_dinov2_embeddings", False),
        dinov2_dim=config.get("dinov2_feat_dim", DINOV2_EMBED_DIM),
    )
    return model


def _load_checkpoint(model: PropensityModel, ckpt_file: str):
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    return model


def _get_baselines(config, seed):
    baselines_dir = config.get("baseline_path", os.path.join(os.path.dirname(config["ckpt_path"]), "baselines"))
    if not joblib or not os.path.isdir(baselines_dir):
        return None, None
    naive_path = os.path.join(baselines_dir, f"logreg_naive_{seed}.pkl")
    pseudo_path = os.path.join(baselines_dir, f"logreg_pseudo_oracle_{seed}.pkl")
    if not (os.path.isfile(naive_path) and os.path.isfile(pseudo_path)):
        return None, None
    try:
        return joblib.load(naive_path), joblib.load(pseudo_path)
    except Exception:
        return None, None


def train_model(config, seed):
    attributes = config.get("attributes", ATTRIBUTES)
    task = config.get("y", TASK)
    shortcuts = _build_shortcuts(config, attributes)
    use_dinov2 = config.get("use_dinov2_embeddings", False)

    coefficients = config.get("coefficients", COEFFICIENTS) if config.get('use_synthetic_label', False) else None
    
    train_loader, indices = get_dataloader(
        batch_size=config["batch_size"],
        split="train",
        attributes=attributes,
        transforms=_train_transforms(config),
        data_dir=config["data_dir"],
        task=task,
        shortcuts=shortcuts,
        use_dinov2_embeddings=use_dinov2,
        use_cached_images=config.get("use_cached_images", False),
        coefficients=coefficients,
        seed=seed
    )

    val_loader, _ = get_dataloader(
        batch_size=config["batch_size"],
        split="valid",
        attributes=attributes,
        data_dir=config["data_dir"],
        task=task,
        shortcuts=shortcuts,
        use_dinov2_embeddings=use_dinov2,
        use_cached_images=config.get("use_cached_images", False),
        coefficients=coefficients,
        seed=seed
    )

    os.makedirs(config["ckpt_path"], exist_ok=True)
    name = _build_run_name(config, seed)
    ckpts = [f for f in os.listdir(config["ckpt_path"]) if name in f and f.endswith(".ckpt")]
    if len(ckpts) > 1:
        raise RuntimeError(f"Multiple checkpoints found for run {name}: {ckpts}")

    model = _init_model(config, num_concepts=len(attributes), indices=indices)

    if ckpts:
        ckpt_file = os.path.join(config["ckpt_path"], ckpts[0])
        return _load_checkpoint(model, ckpt_file)

    with open(os.path.join(config["ckpt_path"], "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        callbacks=get_callbacks(config, name),
        default_root_dir=config["ckpt_path"],
        max_epochs=config["max_epochs"],
    )
    trainer.fit(model, train_loader, val_loader)

    return model


def _estimate_ates(model, dataloader, concepts: List[str], config, seed, coefficients=None):
    results = {concept: {} for concept in concepts}
    naive, pseudo_oracle = _get_baselines(config, seed)

    for concept_idx, concept in enumerate(concepts):
        ate_results = model.compare_ates(
            dataloader,
            concept_idx,
            device="cuda",
            coeffs=coefficients,
            naive=naive,
            pseudo_oracle=pseudo_oracle,
            causal_concepts=CAUSAL_CONCEPTS,
            causal_concept_indices=[dataloader.dataset.attr_names.index(c) for c in CAUSAL_CONCEPTS],
        )
        results[concept].update(ate_results)

    return results


def test_model(model, config, split="id", seed=0):
    attributes = config.get("attributes", ATTRIBUTES)
    task = config.get("y", TASK)
    shortcuts = _build_shortcuts(config, attributes)
    use_dinov2 = config.get("use_dinov2_embeddings", False)

    if config.get('use_synthetic_label', False):
        coefficients = config.get("coefficients", COEFFICIENTS) if split == "id" else config.get("coefficients_ood", COEFFICIENTS_OOD)
    else:
        coefficients = None

    test_loader, _ = get_dataloader(
        batch_size=config["batch_size"],
        split="test",
        attributes=attributes,
        data_dir=config["data_dir"],
        task=task,
        shortcuts=shortcuts,
        use_dinov2_embeddings=use_dinov2,
        use_cached_images=config.get("use_cached_images", False),
        coefficients=coefficients,
        seed=seed
    )

    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", default_root_dir=config["ckpt_path"])
    test_results = trainer.test(model, test_loader)[0]

    ate_results = _estimate_ates(model, test_loader, attributes, config=config, seed=seed, coefficients=coefficients)

    return {
        "ate_results": ate_results,
        "test_results": test_results,
    }
