# uncovaer

### **UnCoVAEr:** Identifiable Estimation of Causal Concept Effects under Visual Latent Confounding

**UnCoVAEr (Unobserved Confounding Variational AutoEncoder)** provides a principled framework for estimating causal effects of human-interpretable concepts on image-based model predictions in the presence of latent (unobserved) confounders.

Abstract:

*Estimating the causal effect of human-interpretable visual concepts on outcomes is essential for auditing classifiers and assessing bias in image datasets. However, existing estimators typically assume unconfoundedness, a condition rarely met in practice, as concept annotations are seldom exhaustive. We formalize the problem of visual latent confounding, where unannotated factors manifest as high-dimensional visual signatures that jointly influence observed concepts and outcomes. We present UnCoVAEr (Unobserved Confounding Variational AutoEncoder), a latent-variable model that learns identifiable confounder representations from images. By leveraging observed concepts and outcomes as auxiliary variables, we prove that UnCoVAEr identifies representations sufficient for backdoor adjustment under standard assumptions. Empirically, UnCoVAEr achieves lower causal effect estimation bias on MorphoMNIST and CelebA benchmarks, outperforming feature-adjustment, counterfactual, and latent-variable baselines.*

## Repository Structure
```
uncovaer
├── README.md                                       # (This file)
├── requirements.txt                                # requirements
└── src
    ├── datasets
    │   ├── morphomnist
    │   │   ├── data_confounded_i_by_t_<DIGIT>      # dataset for Single Confounder case (DIGIT: 0-4)
    │   │   ├── data_confounded_i_by_ts_or<DIGIT>   # dataset for Multiple Confounders
    │   │   ├── data_confounded_is_by_t_<DIGIT>     # dataset for Common Confounder
    │   │   ├── data_confounded_s_by_it_<DIGIT>     # dataset for Causal Confounder
    │   │   ├── mnist                               # original mnist
    │   │   ├── morphomnist                         # morphomnist original repo
    │   │   ├── create_dataset.py                   # script to generate confounded datasets
    │   │   └── dataset.py                          # dataloaders
    │   └── celeba
    │       ├── dataset.py                          # dataloaders   
    ├── experiments
    │   ├── morphomnist
    │   │   ├── configs
    │   |   └── runners                             # train_model, test_model, estimate_ate for all models
    ├── celeba
    │   │   ├── configs
    │   |   └─── runners                            # train_model, test_model, estimate_ate for all models
    │   ├─ utils                                    # helper scripts for results gathering and plotting
    │   ├─ run_baselines.py                         # run naive and oracle baselines   
    │   └── run_experiments.py                      # entry-point for our experiments                                   #
    └── models
        ├── cace.py                                 # CaCE
        ├── ipw.py                                  # Image-Adjustment + CBM
        ├── mutual_information.py                   # MI estimator model
        ├── residual_cbm.py                         # Res-CBM
        ├── uncovaer.py                             # UnCoVAEr variants
        └── utils.py                                # helper functions for models
```

## How to Run

### Setup 
```
python3.10 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### Reproduce Experiments
For the Single Confounder case, use the configs under `configs/confounded_i_by_t/`, change accordingly. 

First, train the naive and oracle estimators that will be used by all models to test the confounding criterion and for comparison:
```
python3 run_baselines.py configs/confounded_i_by_t/uncovaer.yaml
```

Next, train, test and estimate CaCEs for all our models.

For each model, the script runs five experiments using random seeds 0–4. In each run, it:
1. Sets the random seed to ensure reproducibility and selects the dataset corresponding to that seed’s digit.
2. Trains the model on the chosen dataset.
3. Evaluates the model by testing its predictions for the target variable $Y$
4. Estimates the Average Treatment Effect (ATE) for the concepts of interest.
5. Computes the correlation and mutual information between the learned confounder representation $Z_C$ and the known latent confounder *(UnCoVAEr only)*

*Training is skipped if a model checkpoint exists in the directory already.*
All hyperparameters can be found in the configuration files.

After all five runs, the script aggregates the results and reports the mean and standard deviation across seeds.
