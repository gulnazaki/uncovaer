import os
import pickle
from pytorch_lightning import Trainer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from datasets.morphomnist.dataset import get_dataloader
from datasets.morphomnist.config import TASK, SHORTCUT, COEFFICIENTS
from models.ipw import PropensityModel
from models.utils import (
    generate_checkpoint_callback,
    generate_early_stopping_callback,
    generate_ema_callback
)
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
    name = f"{TASK}_ipw_{config['feat_dim']}_{seed}"

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
        model = PropensityModel.load_from_checkpoint(
            path,
            num_concepts=len(config['attributes']),
            feat_dim=config['feat_dim'],
            lr=float(config['lr']),
            indices=indices
        )
        return model
    else:
        model = PropensityModel(
            num_concepts=len(config['attributes']),
            feat_dim=config['feat_dim'],
            lr=float(config['lr']),
            indices=indices
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

    model_dir = config.get("baseline_path", os.path.join(os.path.dirname(config["ckpt_path"]), "baselines"))
    naive_loaded = joblib.load(f"{model_dir}/logreg_naive_{seed}.pkl")
    pseudo_oracle_loaded = joblib.load(f"{model_dir}/logreg_pseudo_oracle_{seed}.pkl")

    coefficients = config.get("coefficients", COEFFICIENTS)
    
    for concept_idx, concept in enumerate(concepts):
        ate_results = model.compare_ates(dataloader, concept_idx, device="cuda", coeffs=coefficients,
                                         naive=naive_loaded, pseudo_oracle=pseudo_oracle_loaded)
        results[concept].update(ate_results)

    return results


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

    ate_results = _estimate_ates(model, test_loader, config['attributes'], config=config, seed=seed)

    return {
        "ate_results": ate_results,
        "test_results": test_results,
    }
