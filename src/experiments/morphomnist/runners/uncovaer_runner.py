import os
import pickle
import torch
from pytorch_lightning import Trainer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from datasets.morphomnist.dataset import get_dataloader
from datasets.morphomnist.config import TASK, SHORTCUT, COEFFICIENTS
from models.uncovaer import UnCoVAEr
from models.utils import (
    generate_checkpoint_callback,
    generate_early_stopping_callback,
    generate_ema_callback,
    compute_latent_confounder_metrics
)
from models.aipw_utils import aipw_crossfit, collect_latents_from_dataloader
import joblib


def get_callbacks(config, name):
    callbacks = [
        generate_checkpoint_callback(name, config['ckpt_path'], monitor="val_loss", mode='min'),
        generate_early_stopping_callback(patience=config['patience'], monitor="val_loss", mode='min', min_delta=1e-4)
    ]
    if config.get('ema', False):
        callbacks.append(generate_ema_callback(decay=0.999))
    return callbacks


def _build_run_name(config, seed):
    """Build run name including all latent dimensions for clarity."""
    return (
        f"{TASK}_uncovaer_zci_{config['latent_per_concept']}"
        f"_zc_{config['shared_latent_dim']}"
        f"_zt_{config.get('t_latent_dim', 0)}"
        f"_zy_{config.get('y_latent_dim', 0)}"
        f"_zx_{config['style_latent_dim']}_{seed}"
    )


def _get_model_kwargs(config, indices):
    """Extract model kwargs from config to avoid duplication."""
    return dict(
        num_concepts=len(config['attributes']),
        channels=config.get('channels', 1),
        feat_dim=config['feat_dim'],
        shared_latent_dim=config['shared_latent_dim'],
        latent_per_concept=config['latent_per_concept'],
        t_latent_dim=config.get('t_latent_dim', 0),
        y_latent_dim=config.get('y_latent_dim', 0),
        style_latent_dim=config['style_latent_dim'],
        hidden_dim=config['hidden_dim'],
        lr=float(config['lr']),
        kl_anneal_start=config['kl_anneal_start'],
        kl_anneal_end=config['kl_anneal_end'],
        use_aux=config['use_aux'],
        aux_weight_c=config.get('aux_weight_c', 1.0),
        aux_weight_y=config.get('aux_weight_y', 1.0),
        indices=indices,
        mim_weight=config.get('mim_weight', 0.0),
        use_adversarial_independence=config.get('use_adversarial_independence', False),
        no_X=config.get('no_X', False),
        no_C=config.get('no_C', False),
        conditional_prior=config.get('conditional_prior', False),
        pure_idvae=config.get('pure_idvae', False),
        x_on_c=config.get('x_on_c', True),
        x_on_y=config.get('x_on_y', False),
        z_on_c=config.get('z_on_c', True),
        z_on_y=config.get('z_on_y', True),
        separate_encoders=config.get('separate_encoders', False),
        separate_prior_encoders=config.get('separate_prior_encoders', True),
        causal_parents=config.get('causal_parents', None),
        marginalize_c=config.get('marginalize_c', True),
        beta=config.get('beta', 1.0),
        mse_loss=config.get('mse_loss', False),
        conditional_prior_type=config.get('conditional_prior_type', 'idvae'),
    )


def train_model(config, seed):
    name = _build_run_name(config, seed)

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

    model_kwargs = _get_model_kwargs(config, indices)

    if len(paths) == 1:
        path = os.path.join(config['ckpt_path'], paths[0])
        model = UnCoVAEr.load_from_checkpoint(path, **model_kwargs)
        return model
    else:
        model = UnCoVAEr(**model_kwargs)
        if config.get('mfcvae', False):
            model.initialize_prior_from_gmm(train_loader)

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


def _estimate_ates(model, test_loader, concepts, config, seed):
    results = {}

    num_samples = int(config.get('ate_num_samples', 100))

    for concept_idx, concept in enumerate(concepts):
        results[concept] = {}

    if config.get('use_aipw', False):
        # Load train loader so AIPW can be trained on train and evaluated on test
        train_loader, _, _ = get_dataloader(
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
        
        train_data = collect_latents_from_dataloader(model, train_loader, device="cuda", latent_keys=["z_c", "z_y"], mc_samples=num_samples)
        test_data = collect_latents_from_dataloader(model, test_loader, device="cuda", latent_keys=["z_c", "z_y"], mc_samples=num_samples)  

        for concept_idx, concept in enumerate(concepts):
            res = aipw_crossfit(
                train_data=train_data,
                test_data=test_data,
                concept_idx=concept_idx,
                n_splits=1,
                random_state=seed
            )
            # res is a dict with keys 'ipw','adjustment','aipw', optionally 'double_ml' mapping to (tau,se,ci)
            for key in res:
                results[concept][key] = float(res[key][0])

    # MC sampling (model's internal compare_ates)
    model_dir = config.get("baseline_path", os.path.join(os.path.dirname(config["ckpt_path"]), "baselines"))
    naive_loaded = joblib.load(f"{model_dir}/logreg_naive_{seed}.pkl")
    pseudo_oracle_loaded = joblib.load(f"{model_dir}/logreg_pseudo_oracle_{seed}.pkl")

    coefficients = config.get("coefficients", COEFFICIENTS)

    for concept_idx, concept in enumerate(concepts):
        ate_results = model.compare_ates(
            test_loader, concept_idx, num_samples=num_samples, device="cuda", coeffs=coefficients,
            naive=naive_loaded, pseudo_oracle=pseudo_oracle_loaded
        )
        results[concept].update(ate_results)

    return results


def _compare_latents(trainer, model, dataloader, config, z_type="z_c"):
    """
    Analyze latent representations by computing ROC-AUC and NMI with confounders.
    
    Args:
        z_type: Which latent to analyze - "z_c", "z_t", "z_y", or "z_x"
    """
    predictions = trainer.predict(model, dataloader)

    concepts = torch.cat([pred["c"] for pred in predictions], dim=0)
    y = torch.cat([pred["y"] for pred in predictions], dim=0)
    shortcuts = torch.cat([pred["shortcuts"] for pred in predictions], dim=0)

    shortcut_names = config.get('shortcuts', [SHORTCUT])
    attributes = config['attributes']

    if config["latent_per_concept"] > 0 and z_type in ["z_c", "z_t"]:
        num_concepts = concepts.shape[1]
        z_list = [[] for _ in range(num_concepts)]

        for pred in predictions:
            chunks_key = f"{z_type}_chunks"
            for i in range(num_concepts):
                z_list[i].append(pred[chunks_key][i])

        z = [torch.cat(z_list[i], dim=0) for i in range(num_concepts)]
    else:
        z = torch.cat([pred[z_type] for pred in predictions], dim=0)
    
    # Compute ROC-AUC and NMI for latents vs confounders (shortcuts)
    # If z is a list, results will be per-concept
    confounder_metrics = compute_latent_confounder_metrics(
        latents=z,
        confounders=shortcuts,
        confounder_names=shortcut_names,
        concept_names=attributes if isinstance(z, list) else None,
        include_nmi=True,
        per_dim=False
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
    test_results_raw = trainer.test(model, test_loader)[0]
    test_results = {
        "accuracy": test_results_raw.get('test y accuracy'),
        "accuracy_aux": test_results_raw.get('test aux y accuracy'),
        "accuracy_concepts": test_results_raw.get('test c accuracy'),
        "accuracy_concepts_aux": test_results_raw.get('test aux c accuracy')
    }

    ate_results = _estimate_ates(model, test_loader, config['attributes'], config=config, seed=seed)
    
    # Analyze latents
    latent_results = {}
    for z_type, dim_key in [("z_c", None), ("z_t", "t_latent_dim"), 
                             ("z_y", "y_latent_dim"), ("z_x", "style_latent_dim")]:
        if z_type == "z_c":
            dim = config.get("latent_per_concept", 0) * len(config['attributes']) + config.get("shared_latent_dim", 0)
        else:
            dim = config.get(dim_key, 0)
        if dim > 0:
            latent_results[z_type] = _compare_latents(trainer, model, test_loader, config, z_type=z_type)


    return {
        "ate_results": ate_results,
        "test_results": test_results,
        "latent_results": latent_results        
    }
