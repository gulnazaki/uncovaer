import os
import pickle
from pytorch_lightning import Trainer
from torchvision.transforms import RandomHorizontalFlip
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from datasets.celeba.dataset import get_dataloader
from datasets.celeba.config import ATTRIBUTES, CAUSAL_CONCEPTS, COEFFICIENTS, COEFFICIENTS_OOD, TASK
from models.residual_cbm import ResidualCBM
from models.utils import (
    generate_checkpoint_callback,
    generate_early_stopping_callback,
    generate_ema_callback,
    DINOV2_EMBED_DIM
)
from experiments.celeba.runners.ipw_runner import _init_model as _init_ipw, _load_checkpoint as _load_ipw
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch

def _train_transforms(config):
    return RandomHorizontalFlip(0.5) if config.get("use_augmentation", False) else None

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


def _build_shortcuts(config, attributes):
    shortcuts = config.get("shortcuts")
    if shortcuts is None:
        shortcuts = [c for c in CAUSAL_CONCEPTS if c not in attributes]
    return shortcuts or None


def train_model(config, seed):
    attributes = config.get("attributes", ATTRIBUTES)
    task = config.get("y", TASK)
    shortcuts = _build_shortcuts(config, attributes)
    use_dinov2 = config.get("use_dinov2_embeddings", False)

    coefficients = None
    if config.get("use_synthetic_label", False):
        coefficients = config.get("coefficients", COEFFICIENTS)

    # Dataloaders
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

    # Prepare checkpointing
    os.makedirs(config["ckpt_path"], exist_ok=True)
    name = _build_run_name(config, seed)
    ckpts = [f for f in os.listdir(config["ckpt_path"]) if name in f and f.endswith(".ckpt")]
    if len(ckpts) > 1:
        raise RuntimeError(f"Multiple checkpoints found for run {name}: {ckpts}")

    # Load trained IPW concept encoder checkpoint
    concept_model_dir = os.path.join(os.path.dirname(config["ckpt_path"]), "ipw")
    ipw_ckpts = [f for f in os.listdir(concept_model_dir) if f.endswith('.ckpt') and f"{seed}" in f]
    if not ipw_ckpts:
        raise RuntimeError(f"No IPW checkpoint found in {concept_model_dir} for seed {seed}")
    concept_model_path = os.path.join(concept_model_dir, ipw_ckpts[-1])

    concept_model = _init_ipw(config, num_concepts=len(attributes), indices=indices)
    concept_model = _load_ipw(concept_model, concept_model_path)

    if ckpts:
        path = os.path.join(config["ckpt_path"], ckpts[0])
        model = ResidualCBM.load_from_checkpoint(
            path,
            concept_predictor=concept_model,
            residual_dim=config["residual_dim"],
            num_concepts=len(attributes),
            feat_dim=config["feat_dim"],
            lr=float(config["lr"]),
            indices=indices,
            kl_r=config.get("kl_r", False),
            channels=config.get("channels", 3),
            use_dinov2_embeddings=config.get("use_dinov2_embeddings", False),
            dinov2_feat_dim=config.get("dinov2_feat_dim", DINOV2_EMBED_DIM),
        )
        return model
    else:
        model = ResidualCBM(
            concept_predictor=concept_model,
            residual_dim=config["residual_dim"],
            num_concepts=len(attributes),
            feat_dim=config["feat_dim"],
            lr=float(config["lr"]),
            indices=indices,
            kl_r=config.get("kl_r", False),
            channels=config.get("channels", 3),
            use_dinov2_embeddings=config.get("use_dinov2_embeddings", False),
            dinov2_feat_dim=config.get("dinov2_feat_dim", DINOV2_EMBED_DIM),
        )

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


def fit_propensity_models(model, train_loader, indices, num_concepts, device="cuda"):
    """
    Fit a logistic regressor per concept: P(c_i | r(x)).
    Uses residual features from ResidualCBM.
    """
    model.eval().to(device)

    R_all, T_all = [], [[] for _ in range(num_concepts)]

    with torch.no_grad():
        for x, attr in train_loader:
            x = x.to(device)
            _, r, _ = model.forward(x, r_hard=True)
            r_np = r.cpu().numpy()
            R_all.append(r_np)

            for i in range(num_concepts):
                t = attr[:, indices['concepts']][:, i].cpu().numpy()
                T_all[i].append(t)

    R_all = np.concatenate(R_all, axis=0)
    T_all = [np.concatenate(T, axis=0) for T in T_all]

    prop_models = []
    for i in range(num_concepts):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(R_all, T_all[i])
        prop_models.append(clf)

    return prop_models


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


def test_model(model, config, split="id", seed=0):
    attributes = config.get("attributes", ATTRIBUTES)
    task = config.get("y", TASK)
    shortcuts = _build_shortcuts(config, attributes)
    use_dinov2 = config.get("use_dinov2_embeddings", False)

    if config.get("use_synthetic_label", False):
        coefficients = config.get("coefficients", COEFFICIENTS) if split == "id" else config.get("coefficients_ood", COEFFICIENTS_OOD)
    else:
        coefficients = None

    test_loader, indices = get_dataloader(
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

    # need train loader to fit propensity models
    train_loader, _ = get_dataloader(
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

    prop_models = fit_propensity_models(model, train_loader, indices, len(attributes))

    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", default_root_dir=config["ckpt_path"])
    test_results = trainer.test(model, test_loader)[0]

    naive_loaded, pseudo_oracle_loaded = _get_baselines(config, seed)

    ate_results = {}
    for concept_idx, concept in enumerate(attributes):
        ate_results[concept] = model.compare_ates(
            test_loader,
            concept_idx,
            device="cuda",
            coeffs=coefficients,
            naive=naive_loaded,
            pseudo_oracle=pseudo_oracle_loaded,
            prop_model=prop_models[concept_idx],
            causal_concepts=CAUSAL_CONCEPTS,
            causal_concept_indices=[test_loader.dataset.attr_names.index(c) for c in CAUSAL_CONCEPTS],
        )

    return {
        "ate_results": ate_results,
        "test_results": test_results,
    }
