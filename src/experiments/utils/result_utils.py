from scipy import stats
import numpy as np
import sys
import json


def aggregate_results(results_list):
    agg = {}
    for split in ["id", "ood"]:
        split_dict = {"ate_results": {}, "test_results": {}, "latent_results": {}}

        if results_list[0][split] == {}:
            agg[split] = {}
            continue
        # --- ATE results ---
        ate_keys = results_list[0][split]["ate_results"].keys()
        for k in ate_keys:
            split_dict["ate_results"][k] = {}
            inner_keys = results_list[0][split]["ate_results"][k].keys()
            for ik in inner_keys:
                vals = [res[split]["ate_results"][k][ik] for res in results_list]
                split_dict["ate_results"][k][ik] = (np.mean(vals), np.std(vals)) if all([v is not None for v in vals]) else (np.nan, np.nan)

        # Aggregate ATE error across all concepts (using per-concept mean/std)
        if split_dict["ate_results"]:
            for est_name in ['ate_error', 'ate_error_cbm', 'ate_error_ipw', 'ate_error_naive', 'ate_error_pseudo_oracle', 'ate_error_adj']:
                mu_est, std_est = aggregate_ate_error_over_concepts(split_dict["ate_results"], name=est_name)
                if not np.isnan(mu_est) and not np.isnan(std_est):
                    split_dict["ate_results"][f"all_concepts_{est_name}"] = (mu_est, std_est)

            # # Aggregate AIPW/IPW/Adjustment estimates across concepts when present
            # for est_name in ['aipw', 'ipw', 'adjustment', 'double_ml', 'adj']:
            #     mu_est, std_est = aggregate_ate_error_over_concepts(split_dict["ate_results"], name=est_name)
            #     if not np.isnan(mu_est) and not np.isnan(std_est):
            #         split_dict["ate_results"][f"all_concepts_{est_name}"] = (mu_est, std_est)

        # --- Test results ---
        test_keys = results_list[0][split]["test_results"].keys()
        for k in test_keys:
            vals = [res[split]["test_results"][k] for res in results_list]
            split_dict["test_results"][k] = (np.mean(vals), np.std(vals))

        # --- Latent results (confounder metrics: ROC-AUC and NMI) ---
        if "latent_results" in results_list[0][split]:
            latent_keys = results_list[0][split]["latent_results"].keys()
            for lk in latent_keys:
                if lk.startswith("pcf"):
                    continue  # PCF handled separately
                
                first_latent = results_list[0][split]["latent_results"][lk]
                
                # Check if it's nested (per-concept) or flat structure
                # Nested: {concept_name: {confounder_name: {roc_auc, nmi}}}
                # Flat: {confounder_name: {roc_auc, nmi}}
                first_value = next(iter(first_latent.values()))
                is_nested = isinstance(first_value, dict) and "roc_auc" not in first_value
                
                if is_nested:
                    # Per-concept structure
                    split_dict["latent_results"][lk] = {}
                    for concept_name in first_latent.keys():
                        split_dict["latent_results"][lk][concept_name] = {}
                        for conf_name in first_latent[concept_name].keys():
                            split_dict["latent_results"][lk][concept_name][conf_name] = {}
                            # Aggregate roc_auc, nmi, dcor, and mmd separately
                            for metric in first_latent[concept_name][conf_name].keys():
                                vals = [
                                    res[split]["latent_results"][lk][concept_name][conf_name].get(metric)
                                    for res in results_list
                                    if res[split]["latent_results"][lk][concept_name][conf_name].get(metric) is not None
                                ]
                                if vals:
                                    split_dict["latent_results"][lk][concept_name][conf_name][metric] = (np.mean(vals), np.std(vals))
                else:
                    # Flat structure: {confounder_name: {roc_auc, nmi, dcor, mmd}}
                    split_dict["latent_results"][lk] = {}
                    for conf_name in first_latent.keys():
                        for metric in first_latent[conf_name].keys():
                            vals = [
                                res[split]["latent_results"][lk][conf_name].get(metric)
                                for res in results_list
                                if res[split]["latent_results"][lk][conf_name].get(metric) is not None
                            ]
                            if vals:
                                split_dict["latent_results"][lk][f"{conf_name}_{metric}"] = (np.mean(vals), np.std(vals))

        agg[split] = split_dict

    return agg

def compare_models(results_a, results_b, metric, split="id"):
    """Two-sided t-test between two models' per-seed results."""
    vals_a = [r[split][metric] for r in results_a]
    vals_b = [r[split][metric] for r in results_b]
    t, p = stats.ttest_ind(vals_a, vals_b, equal_var=False)
    return t, p

def aggregate_mean_std(pairs):
    """
    pairs: list of [mu, std] entries
    returns: aggregated mean, aggregated std
    """
    mus = np.array([p[0] for p in pairs])
    stds = np.array([p[1] for p in pairs])
    mean_total = np.mean(mus)
    var_total = np.mean(stds**2 + (mus - mean_total)**2)
    return mean_total, np.sqrt(var_total)

def aggregate_ate_error_over_concepts(ate_results_dict, name='ate_error'):
    """
    Aggregate ate_error across concepts given an aggregated ate_results dict.

    ate_results_dict: dict mapping concept -> dict of metrics, where
      each metric (including 'ate_error') is a pair (mean, std) across seeds.

    Returns: (mu_all, std_all) as floats. If no valid pairs found, returns (np.nan, np.nan).
    """
    pairs = []
    for concept_key, metrics in (ate_results_dict or {}).items():
        if not isinstance(metrics, dict):
            continue
        pair = metrics.get(name, None)
        if pair is None:
            continue
        # validate pair shape (mean, std)
        try:
            mu, sd = float(pair[0]), float(pair[1])
            if not (np.isnan(mu) or np.isnan(sd)):
                pairs.append((mu, sd))
        except Exception:
            continue

    if not pairs:
        return np.nan, np.nan
    return aggregate_mean_std(pairs)

def main(results_file):
    with open(results_file, "r") as f:
        data = json.load(f)

    aggregate = data["aggregate"]

    for split_name, split in aggregate.items():
        print(f"\n=== {split_name.upper()} ===")
        ate_results = split.get("ate_results", {})
        mu, std = aggregate_ate_error_over_concepts(ate_results)
        mu = str(round(mu, 3))[1:]
        std = str(round(std, 2))[1:]
        print(f"{mu} $\pm$ {std}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python result_utils.py results.json")
    else:
        main(sys.argv[1])
