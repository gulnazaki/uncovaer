import os
import pickle
import torch
from pytorch_lightning import Trainer
from torchvision.transforms import RandomHorizontalFlip
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from datasets.celeba.dataset import get_dataloader
from datasets.celeba.config import ATTRIBUTES, CAUSAL_CONCEPTS, TASK, COEFFICIENTS, COEFFICIENTS_OOD
from models.cace_celeba import CaCECelebA
from models.utils import (
    generate_checkpoint_callback,
    generate_early_stopping_callback,
    generate_ema_callback,
    compute_latent_confounder_metrics,
)
from models.aipw_utils import aipw_crossfit

try:
    import joblib
except ImportError:
    joblib = None


def get_callbacks(config, name):
    callbacks = [
        generate_checkpoint_callback(name, config['ckpt_path'], monitor="val_loss", mode='min'),
        generate_early_stopping_callback(patience=config['patience'], monitor="val_loss", mode='min', min_delta=1e-4)
    ]
    if config.get('ema', False):
        callbacks.append(generate_ema_callback(decay=0.999))
    return callbacks


def _build_run_name(config, seed):
    return f"{config['y']}_cace_zci_{config['latent_per_concept']}_zc_{config['shared_latent_dim']}_zs_{config['style_latent_dim']}_{seed}"


def train_model(config, seed):
    name = _build_run_name(config, seed)
    attributes = config.get("attributes", ATTRIBUTES)
    task = config.get("y", TASK)
    shortcuts = config.get("shortcuts", [c for c in CAUSAL_CONCEPTS if c not in attributes])

    transforms = RandomHorizontalFlip(0.5) if config.get('use_augmentation', False) else None

    coefficients = config.get("coefficients", COEFFICIENTS) if config.get('use_synthetic_label', False) else None

    train_loader, indices = get_dataloader(
        batch_size=config['batch_size'],
        split='train',
        attributes=attributes,
        transforms=transforms,
        task=task,
        data_dir=config['data_dir'],
        shortcuts=shortcuts,
        use_dinov2_embeddings=config.get('use_dinov2_embeddings', False),
        use_cached_images=config.get('use_cached_images', False),
        coefficients=coefficients,
        seed=seed
    )

    val_loader, _ = get_dataloader(
        batch_size=config['batch_size'],
        split='valid',
        attributes=attributes,
        task=task,
        data_dir=config['data_dir'],
        shortcuts=shortcuts,
        use_dinov2_embeddings=config.get('use_dinov2_embeddings', False),
        use_cached_images=config.get('use_cached_images', False),
        coefficients=coefficients,
        seed=seed
    )

    os.makedirs(config['ckpt_path'], exist_ok=True)
    existing = [f for f in os.listdir(config['ckpt_path']) if name in f and f.endswith('.ckpt')]
    if len(existing) > 1:
        raise RuntimeError(f"Multiple checkpoints matching run name: {existing}")
    if len(existing) == 1:
        model = CaCECelebA.load_from_checkpoint(
            os.path.join(config['ckpt_path'], existing[0]),
            num_concepts=len(attributes),
            feat_dim=config['feat_dim'],
            shared_latent_dim=config['shared_latent_dim'],
            latent_per_concept=config['latent_per_concept'],
            style_latent_dim=config['style_latent_dim'],
            hidden_dim=config['hidden_dim'],
            lr=float(config['lr']),
            kl_anneal_start=config['kl_anneal_start'],
            kl_anneal_end=config['kl_anneal_end'],
            use_aux=config['use_aux'],
            aux_weight_c=config.get('aux_weight_c', 1.0),
            aux_weight_y=config.get('aux_weight_y', 1.0),
            indices=indices,
            use_dinov2_embeddings=config.get('use_dinov2_embeddings', False)
        )
        return model
    else:
        model = CaCECelebA(
            num_concepts=len(attributes),
            feat_dim=config['feat_dim'],
            shared_latent_dim=config['shared_latent_dim'],
            latent_per_concept=config['latent_per_concept'],
            style_latent_dim=config['style_latent_dim'],
            hidden_dim=config['hidden_dim'],
            lr=float(config['lr']),
            kl_anneal_start=config['kl_anneal_start'],
            kl_anneal_end=config['kl_anneal_end'],
            use_aux=config['use_aux'],
            aux_weight_c=config.get('aux_weight_c', 1.0),
            aux_weight_y=config.get('aux_weight_y', 1.0),
            indices=indices,
            use_dinov2_embeddings=config.get('use_dinov2_embeddings', False)
        )

    with open(os.path.join(config['ckpt_path'], "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        callbacks=get_callbacks(config, name),
        default_root_dir=config['ckpt_path'],
        max_epochs=config['max_epochs'],
    )
    trainer.fit(model, train_loader, val_loader)

    return model


def _estimate_ates(model, dataloader, concepts, config, seed, coefficients):
    results = {concept: {} for concept in concepts}

    if config.get('use_aipw', False):
        for concept_idx, concept in enumerate(concepts):
            tau, se, ci = aipw_crossfit(
                model=model,
                dataloader=dataloader,
                concept_idx=concept_idx,
                n_splits=5
            )
            results[concept]["aipw"] = float(tau)

    baselines_dir = config.get("baseline_path", os.path.join(os.path.dirname(config["ckpt_path"]), "baselines"))
    naive = pseudo_oracle = None
    if baselines_dir and joblib:
        naive_path = os.path.join(baselines_dir, f"logreg_naive_{seed}.pkl")
        pseudo_path = os.path.join(baselines_dir, f"logreg_pseudo_oracle_{seed}.pkl")
        if os.path.isfile(naive_path) and os.path.isfile(pseudo_path):
            try:
                naive = joblib.load(naive_path)
                pseudo_oracle = joblib.load(pseudo_path)
            except Exception:
                naive = pseudo_oracle = None
    num_samples = int(config.get('ate_num_samples', 100))
    sensitivity = float(config.get('ate_sensitivity', 0.05))
    
    for concept_idx, concept in enumerate(concepts):
        ate_results = model.compare_ates(
            dataloader,
            concept_idx,
            num_samples=num_samples,
            naive=naive,
            pseudo_oracle=pseudo_oracle,
            device="cuda",
            coeffs=coefficients,
            causal_concepts=CAUSAL_CONCEPTS,
            causal_concept_indices=[dataloader.dataset.attr_names.index(c) for c in CAUSAL_CONCEPTS],
        )
        results[concept].update(ate_results)
    return results


def _compare_latents(trainer, model, dataloader, config, attributes, shortcut_names, z_type="z_c"):
    """
    Analyze latent representations by computing ROC-AUC and NMI with confounders.
    
    Args:
        z_type: Which latent to analyze - "z_c" or "z_x"
    """
    predictions = trainer.predict(model, dataloader)

    concepts = torch.cat([pred["c"] for pred in predictions], dim=0)
    y = torch.cat([pred["y"] for pred in predictions], dim=0)
    shortcuts = torch.cat([pred["shortcuts"] for pred in predictions], dim=0)

    if z_type == "z_c":
        if config["latent_per_concept"] > 0:
            num_concepts = concepts.shape[1]
            z_list = [[] for _ in range(num_concepts)]
            for pred in predictions:
                for i in range(num_concepts):
                    z_list[i].append(pred["z_chunks"][i])
            z = [torch.cat(z_list[i], dim=0) for i in range(num_concepts)]
        else:
            z = torch.cat([pred["z_c"] for pred in predictions], dim=0)
    elif z_type == "z_x":
        z = torch.cat([pred["z_s"] for pred in predictions], dim=0)
    else:
        raise ValueError(f"Unknown z_type: {z_type}")
    
    # Compute ROC-AUC and NMI for latents vs confounders (shortcuts)
    confounder_metrics = compute_latent_confounder_metrics(
        latents=z,
        confounders=shortcuts,
        confounder_names=shortcut_names,
        concept_names=attributes if isinstance(z, list) else None,
        max_samples=1000
    )
    
    return confounder_metrics


def test_model(model, config, split="id", seed=0):
    attributes = config.get("attributes", ATTRIBUTES)
    task = config.get("y", TASK)
    shortcuts = config.get("shortcuts", [c for c in CAUSAL_CONCEPTS if c not in attributes])

    if config.get('use_synthetic_label', False):
        coefficients = config.get("coefficients", COEFFICIENTS) if split == "id" else config.get("coefficients_ood", COEFFICIENTS_OOD)
    else:
        coefficients = None

    test_loader, _ = get_dataloader(
        batch_size=config['batch_size'],
        split='test',
        attributes=attributes,
        task=task,
        data_dir=config['data_dir'],
        shortcuts=shortcuts,
        use_dinov2_embeddings=config.get('use_dinov2_embeddings', False),
        use_cached_images=config.get('use_cached_images', False),
        coefficients=coefficients,
        seed=seed
    )

    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", default_root_dir=config['ckpt_path'])
    test_results_raw = trainer.test(model, test_loader)[0]
    test_results = {
        "accuracy_aux": test_results_raw['test aux y accuracy']
    }

    ate_results = _estimate_ates(model, test_loader, attributes, config=config, seed=seed, coefficients=coefficients)
    
    # Analyze latents (z_c and z_x)
    latent_results = {}
    for z_type, dim_key in [("z_c", None), ("z_x", "style_latent_dim")]:
        if z_type == "z_c":
            dim = config.get("latent_per_concept", 0) * len(attributes) + config.get("shared_latent_dim", 0)
        else:
            dim = config.get(dim_key, 0)
        if dim > 0:
            try:
                latent_results[z_type] = _compare_latents(trainer, model, test_loader, config, attributes, shortcut_names=shortcuts, z_type=z_type)
            except Exception as e:
                print(f"{z_type} analysis failed: {e}")

    if config.get('out_dir', None) and not config.get('use_dinov2_embeddings', False):
        out_dir = os.path.join(config['out_dir'], f'counterfactuals_{split}_{seed}')
        os.makedirs(out_dir, exist_ok=True)
        if hasattr(model, 'create_counterfactuals'):
            print("\nGenerating Counterfactuals...")
            model.create_counterfactuals(test_loader, out_dir, concept_names=attributes, device="cuda")

    return {
        "ate_results": ate_results,
        "test_results": test_results,
        "latent_results": latent_results
    }
