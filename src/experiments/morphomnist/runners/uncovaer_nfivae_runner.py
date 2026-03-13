import os
import pickle
import torch
import numpy as np
from pytorch_lightning import Trainer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from datasets.morphomnist.dataset import get_dataloader
from datasets.morphomnist.config import TASK, SHORTCUT, COEFFICIENTS
from models.uncovaer_nfivae import NFiVAE
from models.utils import (
    generate_checkpoint_callback,
    generate_early_stopping_callback,
    generate_ema_callback,
    compute_latent_confounder_metrics,
)
from models.aipw_utils import aipw_crossfit
import joblib
import json


def get_callbacks(config, name):
    callbacks = [
        generate_checkpoint_callback(name, config['ckpt_path'], monitor="val_loss", mode='min'),
        generate_early_stopping_callback(patience=config['patience'], monitor="val_loss", mode='min', min_delta=1e-4)
    ]
    if config.get('ema', False):
        callbacks.append(generate_ema_callback(decay=0.999))
    return callbacks


def _build_run_name(config, seed):
    """Build run name for NFiVAE."""
    return (
        f"{TASK}_nfivae_z_{config['latent_dim']}"
        f"_nf_{config.get('nf_dim', 32)}"
        f"_pred_{int(config.get('use_predictors', False))}_{seed}"
    )


def _get_model_kwargs(config, indices):
    """Extract model kwargs from config for NFiVAE."""
    return dict(
        num_concepts=len(config['attributes']),
        channels=config.get('channels', 1),
        feat_dim=config['feat_dim'],
        hidden_dim=config['hidden_dim'],
        latent_dim=config['latent_dim'],
        lr=float(config['lr']),
        indices=indices,
        # Annealing
        kl_anneal_start=config['kl_anneal_start'],
        kl_anneal_end=config['kl_anneal_end'],
        # Loss weights
        recon_weight=config.get('recon_weight', 1.0),
        # Auxiliary classifiers q(c|x), q(y|c,x)
        use_aux=config.get('use_aux', True),
        aux_weight_c=config.get('aux_weight_c', 1.0),
        aux_weight_y=config.get('aux_weight_y', 1.0),
        # Optional predictors p(c|z), p(y|c,z)
        use_predictors=config.get('use_predictors', False),
        predictor_weight_c=config.get('predictor_weight_c', 1.0),
        predictor_weight_y=config.get('predictor_weight_y', 1.0),
        # Causal structure
        causal_parents=config.get('causal_parents', None),
        use_factorized_prior=config.get('use_factorized_prior', False),
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
        model = NFiVAE.load_from_checkpoint(path, **model_kwargs)
        return model
    else:
        model = NFiVAE(**model_kwargs)

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
        for concept_idx, concept in enumerate(concepts):
            res = aipw_crossfit(
                model=model,
                dataloader=test_loader,
                train_dataloader=train_loader,
                concept_idx=concept_idx,
                n_splits=1,
                random_state=seed
            )
            # res is a dict with keys 'ipw','adjustment','aipw' mapping to (tau,se,ci)
            results[concept]["aipw"] = float(res['aipw'][0])
            results[concept]["ipw"] = float(res['ipw'][0])
            results[concept]["adjustment"] = float(res['adjustment'][0])

    # MC sampling (model's internal compare_ates)
    model_dir = config.get("baseline_path", os.path.join(os.path.dirname(config["ckpt_path"]), "baselines"))
    naive_loaded = joblib.load(f"{model_dir}/logreg_naive_{seed}.pkl")
    pseudo_oracle_loaded = joblib.load(f"{model_dir}/logreg_pseudo_oracle_{seed}.pkl")

    coefficients = config.get("coefficients", COEFFICIENTS)
    num_samples = int(config.get('ate_num_samples', 100))

    for concept_idx, concept in enumerate(concepts):
        ate_results = model.compare_ates(
            test_loader, concept_idx, num_samples=num_samples, device="cuda", coeffs=coefficients,
            naive=naive_loaded, pseudo_oracle=pseudo_oracle_loaded
        )
        results[concept].update(ate_results)

    return results


def _compare_latents(trainer, model, dataloader, config):
    """
    Analyze latent representations by computing ROC-AUC and NMI with confounders.
    NFiVAE has a single unified latent Z.
    """
    predictions = trainer.predict(model, dataloader)

    concepts = torch.cat([pred["c"] for pred in predictions], dim=0)
    y = torch.cat([pred["y"] for pred in predictions], dim=0)
    shortcuts = torch.cat([pred["shortcuts"] for pred in predictions], dim=0)

    shortcut_names = config.get('shortcuts', [SHORTCUT])
    attributes = config['attributes']

    # NFiVAE has a single unified latent z
    z = torch.cat([pred["z"] for pred in predictions], dim=0)
    
    # Compute ROC-AUC and NMI for latents vs confounders (shortcuts)
    confounder_metrics = compute_latent_confounder_metrics(
        latents=z,
        confounders=shortcuts,
        confounder_names=shortcut_names,
        concept_names=None,  # Single unified latent, not per-concept
        per_dim=True
    )
    
    return confounder_metrics


def _discover_confounders(model, dataloader, config, seed, out_dir=None):
    """
    Phase 2: Discover which latent dimensions are confounders for each concept.
    
    Runs multiple confounder discovery methods:
    1. Sparse regression (Lasso/ElasticNet)
    2. Conditional independence tests (partial correlation / KCI)
    3. Causal discovery (PC algorithm)
    
    Args:
        model: Trained NFiVAE model
        dataloader: Test dataloader
        config: Configuration dictionary
        seed: Random seed
        out_dir: Directory to save visualizations
        
    Returns:
        Dictionary with confounder discovery results for each concept and method
    """
    concepts = config['attributes']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Configuration for confounder discovery
    sparse_method = config.get('confounder_method', 'lasso')  # 'lasso' or 'elasticnet'
    sparse_regularization = config.get('confounder_regularization', 1.0)  # L1 regularization strength
    ci_method = config.get('ci_test_method', 'partial_corr')  # 'partial_corr' or 'kci'
    ci_n_confounders = config.get('ci_test_n_confounders', None)
    ci_max_latent_dims_pairwise = config.get('ci_test_max_latent_dims_pairwise', 32)
    ci_pairwise_pval_margin = config.get('ci_test_pairwise_pval_margin', 0.0)
    cd_method = config.get('causal_discovery_method', 'pc')  # 'pc', 'fci', or 'ges'
    alpha = config.get('confounder_alpha', 0.05)
    run_sparse = config.get('run_sparse_discovery', True)
    run_ci_test = config.get('run_ci_test_discovery', True)
    run_causal_discovery = config.get('run_causal_discovery', True)
    
    results = {}
    
    print("\n" + "=" * 70)
    print("Phase 2: Confounder Discovery")
    print("=" * 70)
    print(f"  Methods enabled:")
    print(f"    - Sparse regression ({sparse_method}, C={sparse_regularization}): {run_sparse}")
    print(f"    - Conditional independence test ({ci_method}): {run_ci_test}")
    print(f"    - Causal discovery ({cd_method}): {run_causal_discovery}")
    print(f"  Alpha: {alpha}")
    
    # Analyze each concept
    for concept_idx, concept_name in enumerate(concepts):
        print(f"\n{'='*50}")
        print(f"Analyzing concept '{concept_name}' (idx={concept_idx})")
        print(f"{'='*50}")
        
        results[concept_name] = {}
        
        # ========== Method 1: Sparse Regression ==========
        if run_sparse:
            try:
                print(f"\n--- Method 1: Sparse Regression ({sparse_method}) ---")
                sparse_results = model.discover_confounders_sparse(
                    dataloader,
                    concept_idx=concept_idx,
                    device=device,
                    method=sparse_method,
                    regularization=sparse_regularization,
                )
                
                results[concept_name]["sparse"] = {
                    "confounder_dims": sparse_results["confounder_dims"],
                    "n_confounders": sparse_results["n_confounders"],
                    "instrument_dims": sparse_results.get("instrument_dims", []),
                    "outcome_only_dims": sparse_results.get("outcome_only_dims", []),
                    "ranked_dims": sparse_results["ranked_dims"][:10],
                    "confounder_scores": sparse_results["confounder_score"].tolist(),
                    "coef_treatment": sparse_results["coef_treatment"].tolist(),
                    "coef_outcome": sparse_results["coef_outcome"].tolist(),
                    "no_confounders_detected": sparse_results.get("no_confounders_detected", False),
                    "model_c_score": sparse_results.get("model_c_score"),
                    "model_y_score": sparse_results.get("model_y_score"),
                }
                
                print(f"  Sparse regression results:")
                print(f"    Confounders (C∩Y): {sparse_results['confounder_dims']}")
                print(f"    Instruments (C only): {sparse_results.get('instrument_dims', [])}")
                print(f"    Outcome-only (Y only): {sparse_results.get('outcome_only_dims', [])}")
                
                # Save visualization
                if out_dir:
                    viz_path = os.path.join(out_dir, f"sparse_{concept_name}_{seed}.png")
                    try:
                        model.visualize_confounder_analysis(sparse_results, save_path=viz_path)
                    except Exception as e:
                        print(f"    Warning: Failed to save visualization: {e}")
                        
            except Exception as e:
                print(f"  Sparse regression failed: {e}")
                import traceback
                traceback.print_exc()
                results[concept_name]["sparse"] = {"error": str(e)}
        
        # ========== Method 2: Conditional Independence Test ==========
        if run_ci_test:
            try:
                print(f"\n--- Method 2: Conditional Independence Test ({ci_method}) ---")
                ci_results = model.discover_confounders_ci_test(
                    dataloader,
                    concept_idx=concept_idx,
                    device=device,
                    alpha=alpha,
                    method=ci_method,
                    n_confounders=ci_n_confounders,
                    max_latent_dims_pairwise=ci_max_latent_dims_pairwise,
                    pairwise_pval_margin=ci_pairwise_pval_margin,
                )

                # Support both legacy per-dimension CI-test outputs and CiVAE-style pairwise KCI outputs.
                ci_payload = {
                    "method": ci_method,
                    "alpha": alpha,
                    "confounder_dims": ci_results.get("confounder_dims", []),
                    "n_confounders": ci_results.get("n_confounders", len(ci_results.get("confounder_dims", []))),
                    "no_confounders_detected": ci_results.get("no_confounders_detected", False),
                }

                if "pval_pairwise_before" in ci_results and "pval_pairwise_after" in ci_results:
                    ci_payload.update({
                        "dims_used": ci_results.get("dims_used", []),
                        "confounder_pairs": ci_results.get("confounder_pairs", []),
                        "pairwise_pval_margin": ci_results.get("pairwise_pval_margin", 0.0),
                        "pval_pairwise_before": np.asarray(ci_results["pval_pairwise_before"]).tolist(),
                        "pval_pairwise_after": np.asarray(ci_results["pval_pairwise_after"]).tolist(),
                    })
                    results[concept_name]["ci_test"] = ci_payload
                    print(f"  CI test results (CiVAE pairwise KCI):")
                    print(f"    Confounder dims: {ci_results.get('confounder_dims', [])}")
                    print(f"    Confounder pairs: {len(ci_results.get('confounder_pairs', []))}")
                else:
                    ci_payload.update({
                        "instrument_dims": ci_results.get("instrument_dims", []),
                        "outcome_only_dims": ci_results.get("outcome_only_dims", []),
                        "ranked_dims": ci_results.get("ranked_dims", [])[:10],
                        "confounder_scores": np.asarray(ci_results.get("confounder_score", [])).tolist(),
                        "pval_z_indep_c": np.asarray(ci_results.get("pval_z_indep_c", [])).tolist(),
                        "pval_z_indep_y_given_c": np.asarray(ci_results.get("pval_z_indep_y_given_c", [])).tolist(),
                    })
                    results[concept_name]["ci_test"] = ci_payload

                    print(f"  CI test results:")
                    print(f"    Confounders (dep on C AND Y|C): {ci_results.get('confounder_dims', [])}")
                    print(f"    Instruments (dep on C only): {ci_results.get('instrument_dims', [])}")
                    print(f"    Outcome-only (dep on Y|C only): {ci_results.get('outcome_only_dims', [])}")
                
                # Save visualization
                if out_dir:
                    viz_path = os.path.join(out_dir, f"ci_test_{concept_name}_{seed}.png")
                    try:
                        model.visualize_ci_test_results(ci_results, save_path=viz_path)
                    except Exception as e:
                        print(f"    Warning: Failed to save CI test visualization: {e}")
                        
            except Exception as e:
                print(f"  CI test failed: {e}")
                import traceback
                traceback.print_exc()
                results[concept_name]["ci_test"] = {"error": str(e)}
        
        # ========== Method 3: Causal Discovery ==========
        if run_causal_discovery:
            try:
                print(f"\n--- Method 3: Causal Discovery ({cd_method}) ---")
                cd_results = model.discover_confounders_causal_discovery(
                    dataloader,
                    concept_idx=concept_idx,
                    device=device,
                    alpha=alpha,
                    method=cd_method,
                )
                
                # Store results (convert numpy arrays to lists for JSON)
                results[concept_name]["causal_discovery"] = {
                    "method": cd_method,
                    "alpha": alpha,
                    "confounder_dims": cd_results["confounder_dims"],
                    "n_confounders": cd_results["n_confounders"],
                    "mediator_dims": cd_results.get("mediator_dims", []),
                    "instrument_dims": cd_results.get("instrument_dims", []),
                    "outcome_only_dims": cd_results.get("outcome_only_dims", []),
                    "ranked_dims": cd_results["ranked_dims"][:10],
                    "var_names": cd_results.get("var_names", []),
                    "selected_dims": cd_results.get("selected_dims", []),
                    "c_causes_y": cd_results.get("c_causes_y", False),
                    "y_causes_c": cd_results.get("y_causes_c", False),
                    "c_y_undirected": cd_results.get("c_y_undirected", False),
                    "no_confounders_detected": cd_results.get("no_confounders_detected", False),
                }
                
                print(f"  Causal discovery results:")
                print(f"    Confounders (Z→C and Z→Y): {cd_results['confounder_dims']}")
                print(f"    Mediators (C→Z→Y): {cd_results.get('mediator_dims', [])}")
                print(f"    Instruments (Z→C only): {cd_results.get('instrument_dims', [])}")
                print(f"    C→Y edge detected: {cd_results.get('c_causes_y', False)}")
                
                # Save adjacency matrix visualization
                if out_dir and "adjacency_matrix" in cd_results:
                    try:
                        import matplotlib.pyplot as plt
                        
                        adj = cd_results["adjacency_matrix"]
                        var_names = cd_results.get("var_names", [f"V{i}" for i in range(adj.shape[0])])
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(adj, cmap='RdBu', vmin=-1, vmax=1)
                        ax.set_xticks(range(len(var_names)))
                        ax.set_yticks(range(len(var_names)))
                        ax.set_xticklabels(var_names, rotation=45, ha='right', fontsize=8)
                        ax.set_yticklabels(var_names, fontsize=8)
                        ax.set_title(f"Learned Causal Structure ({cd_method})\nConcept: {concept_name}")
                        plt.colorbar(im, ax=ax, label="Edge (1=directed, 0=none)")
                        
                        # Highlight C and Y
                        n_z = len(var_names) - 2
                        ax.axhline(y=n_z - 0.5, color='green', linestyle='--', linewidth=2)
                        ax.axvline(x=n_z - 0.5, color='green', linestyle='--', linewidth=2)
                        
                        plt.tight_layout()
                        adj_path = os.path.join(out_dir, f"causal_graph_{concept_name}_{seed}.png")
                        plt.savefig(adj_path, dpi=150)
                        plt.close()
                        print(f"    Saved adjacency matrix to {adj_path}")
                    except Exception as e:
                        print(f"    Warning: Failed to save adjacency visualization: {e}")
                        
            except Exception as e:
                print(f"  Causal discovery failed: {e}")
                import traceback
                traceback.print_exc()
                results[concept_name]["causal_discovery"] = {"error": str(e)}
        
        # ========== Consensus and ATE Estimation ==========
        print(f"\n--- Consensus and ATE Estimation ---")
        
        # Collect confounders from all methods
        all_confounders = []
        method_confounders = {}
        
        for method_name in ["sparse", "ci_test", "causal_discovery"]:
            if method_name in results[concept_name] and "confounder_dims" in results[concept_name][method_name]:
                dims = results[concept_name][method_name]["confounder_dims"]
                method_confounders[method_name] = set(dims)
                all_confounders.extend(dims)
        
        # Consensus: intersection (conservative) or union (liberal)
        if len(method_confounders) > 1:
            consensus_intersection = set.intersection(*method_confounders.values()) if method_confounders else set()
            consensus_union = set.union(*method_confounders.values()) if method_confounders else set()
        elif len(method_confounders) == 1:
            consensus_intersection = list(method_confounders.values())[0]
            consensus_union = list(method_confounders.values())[0]
        else:
            consensus_intersection = set()
            consensus_union = set()
        
        # Majority vote: dimensions appearing in >50% of methods
        from collections import Counter
        dim_counts = Counter(all_confounders)
        n_methods = len(method_confounders)
        consensus_majority = [d for d, count in dim_counts.items() if count > n_methods / 2] if n_methods > 0 else []
        
        results[concept_name]["consensus"] = {
            "intersection": list(consensus_intersection),
            "union": list(consensus_union),
            "majority_vote": consensus_majority,
            "per_method": {k: list(v) for k, v in method_confounders.items()},
            "n_methods": n_methods,
        }
        
        print(f"  Consensus confounders:")
        print(f"    Intersection (all methods agree): {list(consensus_intersection)}")
        print(f"    Union (any method): {list(consensus_union)}")
        print(f"    Majority vote (>50%): {consensus_majority}")
        
        # Estimate ATE with different adjustment sets
        print(f"\n  Estimating ATE with different adjustment sets...")
        
        ate_estimates = {}
        
        # ATE with no adjustment (naive)
        try:
            ate_naive = model.estimate_ate_with_adjustment(
                dataloader, concept_idx, confounder_dims=[], device=device
            )
            ate_estimates["naive"] = ate_naive["ate_naive"]
        except Exception as e:
            print(f"    Naive ATE failed: {e}")
            ate_estimates["naive"] = None
        
        # ATE with each method's confounders
        for method_name, dims in method_confounders.items():
            dims_list = list(dims) if dims else []
            if len(dims_list) > 0:
                try:
                    ate_result = model.estimate_ate_with_adjustment(
                        dataloader, concept_idx, confounder_dims=dims_list, device=device
                    )
                    ate_estimates[f"{method_name}_ipw"] = ate_result["ate_ipw"]
                    ate_estimates[f"{method_name}_aipw"] = ate_result["ate_aipw"]
                except Exception as e:
                    print(f"    ATE with {method_name} failed: {e}")
        
        # ATE with consensus confounders
        for consensus_name, consensus_dims in [
            ("intersection", list(consensus_intersection)),
            ("union", list(consensus_union)),
            ("majority", consensus_majority)
        ]:
            if len(consensus_dims) > 0:
                try:
                    ate_result = model.estimate_ate_with_adjustment(
                        dataloader, concept_idx, confounder_dims=consensus_dims, device=device
                    )
                    ate_estimates[f"consensus_{consensus_name}_ipw"] = ate_result["ate_ipw"]
                    ate_estimates[f"consensus_{consensus_name}_aipw"] = ate_result["ate_aipw"]
                except Exception as e:
                    print(f"    ATE with {consensus_name} consensus failed: {e}")
        
        results[concept_name]["ate_estimates"] = ate_estimates
        
        print(f"\n  ATE Estimates:")
        for est_name, est_value in ate_estimates.items():
            if est_value is not None:
                print(f"    {est_name}: {est_value:.4f}")
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("Confounder Discovery Summary")
    print("=" * 70)
    
    for concept_name in concepts:
        print(f"\n{concept_name}:")
        
        # Show confounders from each method
        for method_name in ["sparse", "ci_test", "causal_discovery"]:
            if method_name in results[concept_name] and "confounder_dims" in results[concept_name][method_name]:
                n = results[concept_name][method_name]["n_confounders"]
                dims = results[concept_name][method_name]["confounder_dims"]
                print(f"  {method_name}: {n} confounders {dims[:5]}{'...' if len(dims) > 5 else ''}")
        
        # Show consensus
        if "consensus" in results[concept_name]:
            consensus = results[concept_name]["consensus"]
            print(f"  Consensus (intersection): {consensus['intersection']}")
            print(f"  Consensus (majority): {consensus['majority_vote']}")
        
        # Show ATE comparison
        if "ate_estimates" in results[concept_name]:
            ates = results[concept_name]["ate_estimates"]
            naive = ates.get("naive", 0)
            
            # Find best adjusted ATE (using majority consensus if available)
            adj_ate = ates.get("consensus_majority_aipw", 
                              ates.get("consensus_intersection_aipw",
                              ates.get("sparse_aipw", naive)))
            
            if naive is not None and adj_ate is not None:
                bias = abs(naive - adj_ate)
                print(f"  ATE: {naive:.4f} (naive) -> {adj_ate:.4f} (adjusted), Δ={bias:.4f}")
    
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
    test_results_raw = trainer.test(model, test_loader)[0]
    test_results = {
        "accuracy": test_results_raw.get('test_y_accuracy'),
        "accuracy_aux": test_results_raw.get('test_y_accuracy'),  # Same for NFiVAE
        "accuracy_concepts": test_results_raw.get('test_c_accuracy'),
        "accuracy_concepts_aux": test_results_raw.get('test_c_accuracy'),
    }

    ate_results = _estimate_ates(model, test_loader, config['attributes'], config=config, seed=seed)
    
    # Analyze unified latent z
    latent_results = {}
    try:
        latent_results["z"] = _compare_latents(trainer, model, test_loader, config)
    except Exception as e:
        print(f"Latent analysis failed: {e}")

    # Generate counterfactuals
    if config.get('out_dir', None):
        print("\nGenerating Counterfactuals...")
        out_dir = os.path.join(config['out_dir'], f'counterfactuals_{split}_{seed}')
        os.makedirs(out_dir, exist_ok=True)
        if hasattr(model, 'create_counterfactuals'):
            model.create_counterfactuals(test_loader, out_dir, concept_names=config['attributes'], device="cuda")

    # Phase 2: Confounder Discovery
    confounder_results = {}
    if config.get('run_confounder_discovery', True):
        try:
            cf_out_dir = None
            if config.get('out_dir', None):
                cf_out_dir = os.path.join(config['out_dir'], f'confounder_analysis_{split}')
                os.makedirs(cf_out_dir, exist_ok=True)
            
            confounder_results = _discover_confounders(
                model, test_loader, config, seed, out_dir=cf_out_dir
            )
            
            # Save confounder results to JSON
            if cf_out_dir:
                results_path = os.path.join(cf_out_dir, f"confounder_results_{seed}.json")
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {}
                for k, v in confounder_results.items():
                    serializable_results[k] = {}
                    for k2, v2 in v.items():
                        if isinstance(v2, dict):
                            serializable_results[k][k2] = {
                                k3: (v3.tolist() if hasattr(v3, 'tolist') else v3)
                                for k3, v3 in v2.items()
                            }
                        else:
                            serializable_results[k][k2] = v2.tolist() if hasattr(v2, 'tolist') else v2
                
                with open(results_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                print(f"\nSaved confounder results to {results_path}")
                
        except Exception as e:
            print(f"Confounder discovery failed: {e}")
            import traceback
            traceback.print_exc()

    return {
        "ate_results": ate_results,
        "test_results": test_results,
        "latent_results": latent_results,
        "confounder_results": confounder_results,
    }
