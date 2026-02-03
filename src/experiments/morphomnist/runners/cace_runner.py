import os
import pickle
import torch
from pytorch_lightning import Trainer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from datasets.morphomnist.dataset import get_dataloader
from datasets.morphomnist.config import TASK, SHORTCUT, COEFFICIENTS
from models.cace import CaCE
from models.utils import (
    generate_checkpoint_callback,
    generate_early_stopping_callback,
    generate_ema_callback,
    compute_latent_confounder_metrics,
)
from models.aipw_utils import aipw_crossfit
import joblib


def get_callbacks(config, name):
    callbacks = [
        generate_checkpoint_callback(name, config['ckpt_path'], monitor="val_loss", mode='min'),
        generate_early_stopping_callback(patience=config['patience'], monitor="val_loss", mode='min', min_delta=1e-4)
    ]
    if config.get('ema', False):
        callbacks.append(generate_ema_callback(decay=0.999))
    return callbacks


def train_model(config, seed):
    name = f"{TASK}_uncovaer_zci_{config['latent_per_concept']}_zc_{config['shared_latent_dim']}_zs_{config['style_latent_dim']}_{seed}"

    os.makedirs(config['ckpt_path'], exist_ok=True)
    paths = [f for f in os.listdir(config['ckpt_path']) if name in f]
    if len(paths) > 1:
        exit(f"WARNING: more than one checkpoint in: {paths}")

    train_loader, val_loader, indices = get_dataloader(
        batch_size=config['batch_size'],
        split='train',
        attributes=config['attributes'],
        transforms=config['transforms'],
        binarize=config['binarize'],
        data_dir=config.get('data_dir', None) + f'_{seed}',
        shortcuts=config.get('shortcuts', [SHORTCUT]),
        coefficients=config.get("coefficients", COEFFICIENTS),
        seed=seed
    )

    if len(paths) == 1:
        path = os.path.join(config['ckpt_path'], paths[0])
        model = CaCE.load_from_checkpoint(
            path,
            num_concepts=len(config['attributes']),
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
            tau_anneal_start=config.get('tau_anneal_start', 1.0),
            tau_anneal_min=config.get('tau_anneal_min', 0.1),
            tau_anneal_decay=config.get('tau_anneal_decay', 0.05)
        )
        return model
    else:
        model = CaCE(
            num_concepts=len(config['attributes']),
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
            tau_anneal_start=config.get('tau_anneal_start', 1.0),
            tau_anneal_min=config.get('tau_anneal_min', 0.1),
            tau_anneal_decay=config.get('tau_anneal_decay', 0.05)
        )

    with open(os.path.join(config['ckpt_path'], "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        callbacks=get_callbacks(config, name),
        default_root_dir=config['ckpt_path'],
        max_epochs=config['max_epochs']
    )
    trainer.fit(model, train_loader, val_loader)

    return model


def _estimate_ates(model, dataloader, concepts, config, seed):
    results = {}

    for concept_idx, concept in enumerate(concepts):
        results[concept] = {}

    # # AIPW estimates
    # for concept_idx, concept in enumerate(concepts):
    #     tau, se, ci = aipw_crossfit(
    #         model=model,
    #         dataloader=dataloader,
    #         concept_idx=concept_idx,
    #         n_splits=5
    #     )
    #     results[concept]["aipw"] = float(tau)

    # MC sampling (model's internal compare_ates)
    model_dir = config.get("baseline_path", os.path.join(os.path.dirname(config["ckpt_path"]), "baselines"))
    naive_loaded = joblib.load(f"{model_dir}/logreg_naive_{seed}.pkl")
    pseudo_oracle_loaded = joblib.load(f"{model_dir}/logreg_pseudo_oracle_{seed}.pkl")

    coefficients = config.get("coefficients", COEFFICIENTS)

    for concept_idx, concept in enumerate(concepts):
        ate_results = model.compare_ates(dataloader, concept_idx, num_samples=100, device="cuda", coeffs=coefficients,
                                         naive=naive_loaded, pseudo_oracle=pseudo_oracle_loaded)
        results[concept].update(ate_results)

    return results


def _compare_latents(trainer, model, dataloader, config, z_type="z_c"):
    """
    Analyze latent representations by computing ROC-AUC and NMI with confounders.
    
    Args:
        z_type: Which latent to analyze - "z_c" or "z_x"
    """
    predictions = trainer.predict(model, dataloader)

    concepts = torch.cat([pred["c"] for pred in predictions], dim=0)
    y = torch.cat([pred["y"] for pred in predictions], dim=0)
    shortcuts = torch.cat([pred["shortcuts"] for pred in predictions], dim=0)

    shortcut_names = config.get('shortcuts', [SHORTCUT])
    attributes = config['attributes']

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
        concept_names=attributes if isinstance(z, list) else None
    )
    
    return confounder_metrics


def test_model(model, config, split="id", seed=0):
    test_loader, indices = get_dataloader(
        batch_size=config['batch_size'],
        split='test',
        attributes=config['attributes'],
        transforms=config['transforms'],
        binarize=config['binarize'],
        test_ood=(split == "ood"),
        data_dir=config.get('data_dir', None) + f'_{seed}',
        shortcuts=config.get('shortcuts', [SHORTCUT]),
        coefficients=config.get("coefficients", COEFFICIENTS),
        seed=seed
    )

    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", default_root_dir=config['ckpt_path'])
    test_results = trainer.test(model, test_loader)[0]
    test_results = {
        "accuracy_aux": test_results['test aux y accuracy'],
        "accuracy_concepts_aux": test_results['test aux c accuracy']
    }

    ate_results = _estimate_ates(model, test_loader, config['attributes'], config=config, seed=seed)
    
    # Analyze latents (z_c and z_x)
    latent_results = {}
    for z_type, dim_key in [("z_c", None), ("z_x", "style_latent_dim")]:
        if z_type == "z_c":
            dim = config.get("latent_per_concept", 0) * len(config['attributes']) + config.get("shared_latent_dim", 0)
        else:
            dim = config.get(dim_key, 0)
        if dim > 0:
            try:
                latent_results[z_type] = _compare_latents(trainer, model, test_loader, config, z_type=z_type)
            except Exception as e:
                print(f"{z_type} analysis failed: {e}")

    print("\nGenerating Counterfactuals...")
    out_dir = os.path.join(config['out_dir'], f'counterfactuals_{split}_{seed}')
    os.makedirs(out_dir, exist_ok=True)
    model.create_counterfactuals(test_loader, out_dir, concept_names=config['attributes'], device="cuda")

    return {
        "ate_results": ate_results,
        "test_results": test_results,
        "latent_results": latent_results        
    }
