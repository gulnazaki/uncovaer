import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import yaml

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from experiments.utils.seed_utils import set_global_seed

# --- Helpers
def _to_long_tensor_ids(ids, device="cpu"):
    if ids is None:
        return None
    if isinstance(ids, torch.Tensor):
        return ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(ids, dtype=torch.long, device=device)

def _select_cols(tensor, col_ids):
    if col_ids is None:
        return None
    ids_t = _to_long_tensor_ids(col_ids, device=tensor.device)
    return tensor.index_select(1, ids_t)

# --- Aggregate data from dataloader
def aggregate_data(loader, concept_ids, shortcut_ids, task_id):
    X_concepts, X_all, y = [], [], []
    for _, attr in loader:
        c_t = _select_cols(attr, concept_ids)
        c = c_t.numpy()

        shortcut_ids = [s for s in shortcut_ids if s not in concept_ids] if shortcut_ids is not None else None
        s_t = _select_cols(attr, shortcut_ids) if shortcut_ids is not None else None
        if s_t is not None:
            s = s_t.numpy()
            X_all.append(np.concatenate([c, s], axis=1))
        else:
            X_all.append(c)

        t = attr[:, task_id].numpy()
        X_concepts.append(c)
        y.append(t)

    X_concepts = np.vstack(X_concepts)
    X_all = np.vstack(X_all)
    y = np.concatenate(y)
    return X_concepts, X_all, y

def get_train_and_val_loaders_and_indices(cfg, seed):
    dataset = cfg["dataset"].lower()
    if dataset == "morphomnist":
        from datasets.morphomnist.dataset import get_dataloader as get_morpho
        from datasets.morphomnist.config import COEFFICIENTS
        
        train_loader, val_loader, indices = get_morpho(
            cfg["batch_size"],
            split="train",
            attributes=cfg["attributes"],
            transforms=cfg.get("transforms"),
            binarize=cfg.get("binarize", True),
            test_ood=False,
            data_dir=f"{cfg['data_dir']}_{seed}",
            shortcuts=cfg.get("shortcuts"),
            coefficients=cfg.get("coefficients", COEFFICIENTS),
            seed=seed
        )
        return train_loader, val_loader, indices
    elif dataset == "celeba":
        from datasets.celeba.dataset import get_dataloader as get_celeba
        from datasets.celeba.config import ATTRIBUTES, CAUSAL_CONCEPTS, TASK, COEFFICIENTS
        
        attributes = cfg.get("attributes", ATTRIBUTES)
        task = cfg.get("y", TASK)
        shortcuts = cfg.get("shortcuts", [c for c in CAUSAL_CONCEPTS if c not in attributes])
        
        coefficients = cfg.get("coefficients", COEFFICIENTS) if cfg.get("use_synthetic_label", False) else None
        kwargs = {
            'batch_size': cfg["batch_size"],
            'split': "train",
            'attributes': attributes,
            'transforms': cfg.get("transforms"),
            'data_dir': cfg["data_dir"],
            'use_dinov2_embeddings': cfg.get("use_dinov2_embeddings", False),
            'balance': cfg.get("balance", False),
            'shortcuts': shortcuts,
            'task': task,
            'only_attr': True,
            'coefficients': coefficients,
            'seed': seed
        }

        # Train loader
        train_loader, indices = get_celeba(**kwargs)

        # Validation loader (CelebA supports 'valid')
        kwargs['split'] = "valid"
        val_loader, _ = get_celeba(**kwargs)

        return train_loader, val_loader, indices
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def evaluate_models(naive_clf, pseudo_oracle_clf, X_concepts_val, X_all_val, y_val):
    results = {}

    # Naive
    y_pred = naive_clf.predict(X_concepts_val)
    acc = accuracy_score(y_val, y_pred)
    auc = None
    if hasattr(naive_clf, "predict_proba"):
        try:
            proba = naive_clf.predict_proba(X_concepts_val)
            auc = roc_auc_score(y_val, proba[:, 1] if proba.ndim == 2 else proba)
        except ValueError:
            auc = None
    results["naive"] = {"accuracy": float(acc), "roc_auc": (float(auc) if auc is not None else None)}

    # Pseudo-oracle
    y_pred = pseudo_oracle_clf.predict(X_all_val)
    acc = accuracy_score(y_val, y_pred)
    auc = None
    if hasattr(pseudo_oracle_clf, "predict_proba"):
        try:
            proba = pseudo_oracle_clf.predict_proba(X_all_val)
            auc = roc_auc_score(y_val, proba[:, 1] if proba.ndim == 2 else proba)
        except ValueError:
            auc = None
    results["pseudo_oracle"] = {"accuracy": float(acc), "roc_auc": (float(auc) if auc is not None else None)}

    return results

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_dir = config.get("baseline_path", os.path.join(os.path.dirname(config["ckpt_path"]), "baselines"))
    os.makedirs(model_dir, exist_ok=True)
    results_path = os.path.join(model_dir, "baselines_results.json")

    per_seed_results = []

    for seed in config.get("seeds", [0]):
        set_global_seed(seed)

        model_naive_path = os.path.join(model_dir, f"logreg_naive_{seed}.pkl")
        model_pseudo_path = os.path.join(model_dir, f"logreg_pseudo_oracle_{seed}.pkl")

        if os.path.isfile(model_naive_path) and os.path.isfile(model_pseudo_path):
            print(f"[seed {seed}] existing baselines found; skip")
            return

        # --- Select dataloaders based on dataset
        train_loader, val_loader, indices = get_train_and_val_loaders_and_indices(config, seed)

        # Normalize indices across datasets
        concept_ids = indices["concepts"]
        shortcut_ids = indices.get("shortcut")
        task_id = indices["task"]

        # --- Aggregate training and validation data
        Xc_tr, Xa_tr, y_tr = aggregate_data(train_loader, concept_ids, shortcut_ids, task_id)
        Xc_va, Xa_va, y_va = aggregate_data(val_loader, concept_ids, shortcut_ids, task_id)

        # --- Fit or load naive logistic regression (concepts only)
        naive_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed, penalty='l2', solver='liblinear'))
        naive_clf.fit(Xc_tr, y_tr)

        # --- Fit pseudo-oracle logistic regression (concepts + shortcuts)
        pseudo_oracle_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed, penalty='l2', solver='liblinear'))
        pseudo_oracle_clf.fit(Xa_tr, y_tr)

        # --- Save models
        joblib.dump(naive_clf, model_naive_path)
        joblib.dump(pseudo_oracle_clf, model_pseudo_path)

        # --- Evaluate on validation
        metrics = evaluate_models(naive_clf, pseudo_oracle_clf, Xc_va, Xa_va, y_va)
        per_seed_results.append({"seed": seed, **metrics})

        # --- Print per-seed metrics
        print(f"[seed {seed}] naive: acc={metrics['naive']['accuracy']:.4f}, auc={metrics['naive']['roc_auc']}")
        print(f"[seed {seed}] pseudo_oracle: acc={metrics['pseudo_oracle']['accuracy']:.4f}, auc={metrics['pseudo_oracle']['roc_auc']}")

    # --- Aggregate over seeds
    def _agg(key):
        vals = [r[key]["accuracy"] for r in per_seed_results]
        aucs = [r[key]["roc_auc"] for r in per_seed_results if r[key]["roc_auc"] is not None]
        return {
            "accuracy_mean": float(np.mean(vals)) if len(vals) else None,
            "accuracy_std": float(np.std(vals)) if len(vals) else None,
            "roc_auc_mean": float(np.mean(aucs)) if len(aucs) else None,
            "roc_auc_std": float(np.std(aucs)) if len(aucs) else None,
        }

    aggregate = {"naive": _agg("naive"), "pseudo_oracle": _agg("pseudo_oracle")}
    summary = {"per_seed": per_seed_results, "aggregate": aggregate}

    # Print aggregate
    print("Aggregate (validation) metrics:")
    print(summary)

    # Save results
    import json
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results written to {results_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_baselines.py <config_path>")
        sys.exit(1)

    cfg_path = sys.argv[1]
    main(cfg_path)
