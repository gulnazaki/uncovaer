import os
import pickle
from pytorch_lightning import Trainer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from datasets.morphomnist.dataset import get_dataloader
from datasets.morphomnist.config import TASK, SHORTCUT, COEFFICIENTS
from models.residual_cbm import ResidualCBM
from models.ipw import PropensityModel
from models.utils import (
    generate_checkpoint_callback,
    generate_early_stopping_callback,
    generate_ema_callback
)
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch


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

    concept_model_dir = os.path.join(os.path.dirname(config['ckpt_path']), "ipw")
    concept_model_path = os.path.join(concept_model_dir, [f for f in os.listdir(concept_model_dir) if f'{seed}-epoch=' in f][-1])
    concept_model = PropensityModel.load_from_checkpoint(
            concept_model_path,
            num_concepts=len(config['attributes']),
            feat_dim=config['feat_dim'],
            lr=float(config['lr']),
            indices=indices
        )

    if len(paths) == 1:
        path = os.path.join(config['ckpt_path'], paths[0])
        model = ResidualCBM.load_from_checkpoint(
            path,
            concept_predictor=concept_model,
            residual_dim=config['residual_dim'],
            num_concepts=len(config['attributes']),
            feat_dim=config['feat_dim'],
            lr=float(config['lr']),
            indices=indices,
            kl_r=config.get('kl_r', False)
        )
        return model
    else:
        model = ResidualCBM(
            concept_predictor=concept_model,
            residual_dim=config['residual_dim'],
            num_concepts=len(config['attributes']),
            feat_dim=config['feat_dim'],
            lr=float(config['lr']),
            indices=indices,
            kl_r=config.get('kl_r', False)
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
            # # get residual features (before sampling)
            # r_logits = model.r_cnn(x)                # (B, residual_dim)
            # r_np = r_logits.cpu().numpy()
            _, r, _ = model.forward(x, r_hard=True)
            r_np = r.cpu().numpy()
            R_all.append(r_np)

            # collect treatment labels
            for i in range(num_concepts):
                t = attr[:, indices['concepts']][:, i].cpu().numpy()
                T_all[i].append(t)

    R_all = np.concatenate(R_all, axis=0)              # (N, residual_dim)
    T_all = [np.concatenate(T, axis=0) for T in T_all]

    prop_models = []
    for i in range(num_concepts):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(R_all, T_all[i])
        prop_models.append(clf)

    return prop_models


def test_model(model, config, split="id", seed=0):
    # get test loader
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

    # also need train loader to fit propensity
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
    # fit propensity models on residual features
    prop_models = fit_propensity_models(model, train_loader, indices, len(config['attributes']))

    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto", default_root_dir=config['ckpt_path'])
    test_results = trainer.test(model, test_loader)[0]

    # load baseline S-learners
    model_dir = os.path.join(os.path.dirname(config['ckpt_path']), "baselines")
    naive_loaded = joblib.load(f"{model_dir}/logreg_naive_{seed}.pkl")
    pseudo_oracle_loaded = joblib.load(f"{model_dir}/logreg_pseudo_oracle_{seed}.pkl")

    coefficients = config.get("coefficients", COEFFICIENTS)

    ate_results = {}
    for concept_idx, concept in enumerate(config['attributes']):
        ate_results[concept] = model.compare_ates(
            test_loader,
            concept_idx,
            device="cuda",
            coeffs=coefficients,
            naive=naive_loaded,
            pseudo_oracle=pseudo_oracle_loaded,
            prop_model=prop_models[concept_idx]
        )

    return {
        "ate_results": ate_results,
        "test_results": test_results,
    }
