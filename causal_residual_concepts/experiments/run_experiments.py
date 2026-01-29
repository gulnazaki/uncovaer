import yaml, os, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from experiments.utils.seed_utils import set_global_seed
from experiments.utils.result_utils import aggregate_results
from experiments.morphomnist.runners import uncovaer_runner as uncovaer_morpho, uncovaer_nfivae_runner as nfivae_morpho, cace_runner as cace_morpho, ipw_runner as ipw_morpho, image_adjust_runner as adj_morpho, residual_cbm_runner as rescbm_morpho
from experiments.celeba.runners import uncovaer_runner as uncovaer_celeba, cace_runner as cace_celeba, ipw_runner as ipw_celeba, image_adjust_runner as adj_celeba, residual_cbm_runner as rescbm_celeba

RUNNERS = {
    "morphomnist": {
        "uncovaer": uncovaer_morpho,
        "uncovaer_nfivae": nfivae_morpho,
        "cace": cace_morpho,
        "ipw": ipw_morpho,
        "image_adjust": adj_morpho,
        "residual_cbm": rescbm_morpho
    },
    "celeba": {
        "uncovaer": uncovaer_celeba,
        "cace": cace_celeba,
        "ipw": ipw_celeba,
        "image_adjust": adj_celeba,
        "residual_cbm": rescbm_celeba
    }
}

def run_single_experiment(config, model_name, seed):
    set_global_seed(seed)
    dataset = config["dataset"].lower()
    runner = RUNNERS[dataset][model_name]

    model = runner.train_model(config, seed)
    results = {}
    for split in ["id", "ood"]:
        res = runner.test_model(model, config, split, seed)
        results[split] = res
    return results

def main(config):
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]
    out_dir = config.get("out_dir", f"results/{model_name}")
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    for seed in config.get("seeds", [0,1,2,3,4]):
        res = run_single_experiment(config, model_name, seed)
        all_results.append(res)

    agg = aggregate_results(all_results)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"per_seed": all_results, "aggregate": agg}, f, indent=2)

    print(f"Finished {model_name}")
    # print(json.dumps(agg, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_experiments.py <config_path>")
        sys.exit(1)

    config = sys.argv[1]  # the first argument after script name
    main(config)
