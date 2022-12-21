# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

SEED = 1
MODEL_PATH = "../models/homecredit_{}.pkl"
RESULTS_PATH = "../out/homecredit_submission"
LOAD_MODELS = True
FORCE = True

# !mkdir -p {RESULTS_PATH}

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import os
import re
import sys
import time
import pickle
import typing

import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from tqdm import notebook as tqdm
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

sys.path.append("..")
# -

from src.search import generalized_best_first_search
from src.transformations import NumFeature, TransformationGenerator
from src.utils.counter import ExpansionCounter, CounterLimitExceededError
from src.utils.hash import fast_hash
from exp.framework import ExperimentSuite
from exp import settings
from exp import utils
from exp import plot_params

data_train = settings.get_dataset("home_credit", "../data", mode="train")
data_test = settings.get_dataset("home_credit", "../data", mode="test")
X_train, y_train = data_train.X_train, data_train.y_train
X_test, y_test = data_test.X_test, data_test.y_test

# +
pd.options.display.max_rows = 999

import torch
print(torch.cuda.is_available())

model_zoo = {
#     "lr": LogisticRegression(C=0.04641589),
#     "xgbt": XGBClassifier(n_estimators=500, random_state=SEED, enable_categorical=True),
    "tabnet_clean": utils.load_torch_model(
        "../models/homecredit/clean.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
    "tabnet_cb_100": utils.load_torch_model(
        "../models/homecredit/l1_100.0.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
    "tabnet_cb_300": utils.load_torch_model(
        "../models/homecredit/l1_300.0.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
    "tabnet_cb_1000": utils.load_torch_model(
        "../models/homecredit/l1_1000.0.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
    "tabnet_cb_3000": utils.load_torch_model(
        "../models/homecredit/l1_3000.0.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
    "tabnet_cb_10000": utils.load_torch_model(
        "../models/homecredit/l1_10000.0.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
    "tabnet_cb_100000": utils.load_torch_model(
        "../models/homecredit/l1_100000.0.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
    "tabnet_ub_300000": utils.load_torch_model(
        "../models/homecredit/ub_300000.0.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
    "tabnet_ub_400000": utils.load_torch_model(
        "../models/homecredit/ub_400000.0.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
    "tabnet_ub_500000": utils.load_torch_model(
        "../models/homecredit/ub_500000.0.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
    "tabnet_ub_600000": utils.load_torch_model(
        "../models/homecredit/ub_600000.0.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
    "tabnet_ub_800000": utils.load_torch_model(
        "../models/homecredit/ub_800000.0.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
}
# -


for model_name, clf in tqdm.tqdm(model_zoo.items()):
    if isinstance(clf, utils.TorchWrapper):
        continue

    model_filepath = MODEL_PATH.format(model_name)
    if not LOAD_MODELS or not os.path.exists(model_filepath):
        clf.fit(X_train, y_train)
        with open(model_filepath, "wb") as f:
            pickle.dump(clf, f)
    else:
        with open(model_filepath, "rb") as f:
            model_zoo[model_name] = pickle.load(f)

model_accuracies = {}
for model_name, clf in model_zoo.items():
    #print(X_test.shape)
    acc = clf.score(X_test, y_test)
    model_accuracies[model_name] = acc
    print(f"{model_name}: {acc:.2%}")

eval_settings = settings.setup_dataset_eval("home_credit", "../data", seed=0)
core_bounds = [1, 10, 100, 1000, 10000]
taus = [10000, 300000, 400000, 500000, 600000, 800000]
gain_bounds = [f"AMT_CREDIT-{tau}" for tau in taus]
cost_bounds = core_bounds + gain_bounds

# +
FORCE = False

for model_name, clf in model_zoo.items():
    print(f"=== Model: {model_name} ===")

    for cost_bound in cost_bounds:
        exp_suite = ExperimentSuite(
            clf,
            eval_settings.working_datasets.X_test,
            eval_settings.working_datasets.y_test,
            target_class=eval_settings.target_class,
            dataset=eval_settings.working_datasets.dataset,
            cost_bound=cost_bound,
            spec=eval_settings.spec,
            iter_lim=100,
        )

        for experiment in eval_settings.experiments:
            if experiment.name != "greedy_delta":
                continue

#             if experiment.name in ["pgd", "pgd_1k"] and model_name != "tabnet_clean":
#                 continue

            experiment_path = settings.get_experiment_path(
                RESULTS_PATH, model_name, experiment.name, cost_bound
            )
            if os.path.isfile(experiment_path) and not FORCE:
                # "Already exists. Skipping attack...
                continue

            print(f"{experiment.name} {cost_bound=}")
            print(experiment_path)

            results = exp_suite.run(experiment)
            results.to_pickle(experiment_path)
# -

results_dict = {}

for model_name, clf in model_zoo.items():
    for experiment in eval_settings.experiments:
        for cost_bound in cost_bounds:
            experiment_path = settings.get_experiment_path(
                RESULTS_PATH, model_name, experiment.name, cost_bound,
            )
            if experiment_path in results_dict:
                print(f"{experiment_path} (done)")
                continue

            # Fill out the cost bounds for UCS.
            try:
                if experiment.name == "ucs" and cost_bound is not None:
                    base_experiment_path = settings.get_experiment_path(
                        RESULTS_PATH, model_name, experiment.name, cost_bound=None,
                    )
                    df = pd.read_pickle(base_experiment_path)

                    if not isinstance(cost_bound, str):
                        success_idx = df.cost <= cost_bound
                    else:
                        success_idx = df.cost <= df.apply(
                            lambda row: data_test.cost_orig_df.loc[row.orig_index],
                            axis=1,
                        )

                    df.loc[~success_idx, "adv_x"] = None
                    df.loc[~success_idx, "cost"] = None

                else:
                    print(experiment_path)
                    df = pd.read_pickle(experiment_path)
                    success_idx = ~df.adv_x.isna()

            except FileNotFoundError:
                continue

            print(f"{experiment_path}...")

            # Annotate success.
            target_class = eval_settings.target_class
            df["success"] = success_idx

            # Compute adversarial utility.
            df["adv_util"] = df.apply(
                lambda row: utils.get_adv_utility(
                    target_class,
                    clf,
                    row.x,
                    row.adv_x,
                    row.cost,
                    data_test.cost_orig_df.loc[row.orig_index],
                ),
                axis=1,
            )

            # Annotate costs.
#             df.loc[df.success, "l1"] = df.loc[df.success].apply(
#                 lambda row: (row.x - row.adv_x).abs().sum(), axis=1
#             )
            #             try:
            #                 df.loc[df.success, "actual_cost"] = df.loc[df.success].apply(
            #                     lambda row: utils.recompute_cost(
            #                         eval_settings.spec, row.x, row.adv_x), axis=1
            #                 )
            #             except TypeError:
            #                 print("invalid transformation")
            #                 continue

            # Annotate confidences.
            adv_x_data = np.array([adv_x.values for adv_x in df.loc[df.success].adv_x])
            if df.success.mean() > 0:
                df.loc[df.success, "adv_conf"] = clf.predict_proba(adv_x_data)[
                    :, target_class
                ]
            else:
                df.loc[:, "adv_conf"] = None

            # Annotate accuracy.
            df["acc"] = model_accuracies.get(model_name)

            results_dict[experiment_path] = df.assign(
                model=model_name,
                method=experiment.name,
                cost_bound=cost_bound or "none",
            )

results = pd.concat(results_dict.values())
# +
renaming_dict = {
    "tabnet_clean": f"Clean (Acc: {model_accuracies.get('tabnet_clean'):0.2f})",
#     "tabnet_cb_1_wo": f"CB $\\varepsilon=1$ (Acc: {model_accuracies.get('tabnet_cb_1_wo'):0.2f})",
#     "tabnet_cb_3_wo": f"CB $\\varepsilon=3$ (Acc: {model_accuracies.get('tabnet_cb_3_wo'):0.2f})",
#     "tabnet_cb_10_wo": f"CB $\\varepsilon=10$ (Acc: {model_accuracies.get('tabnet_cb_10_wo'):0.2f})",
#     "tabnet_cb_30_wo": f"CB $\\varepsilon=30$ (Acc: {model_accuracies.get('tabnet_cb_30_wo'):0.2f})",
    "tabnet_ub_300000": f"UB $\\tau = 300K$ (Acc: {model_accuracies.get('tabnet_ub_300000'):0.2f})",
    "tabnet_ub_400000": f"UB $\\tau = 400K$ (Acc: {model_accuracies.get('tabnet_ub_400000'):0.2f})",
    "tabnet_ub_600000": f"UB $\\tau = 600K$ (Acc: {model_accuracies.get('tabnet_ub_600000'):0.2f})",
    "tabnet_ub_800000": f"UB $\\tau = 800K$ (Acc: {model_accuracies.get('tabnet_ub_800000'):0.2f})",
    "random": "Random search",
    "hc": "Hill climbing",
    "greedy": "Benefit-Cost",
    "greedy_delta": "Delta Benefit-Cost",
    "greedy_delta_beam10": "Beam Delta Benefit-Cost (10)",
    "greedy_delta_beam100": "Beam Delta Benefit-Cost (100)",
    "astar_subopt_beam1": "Greedy A*",
    "astar_subopt_beam10": "Beam A* (10)",
    "astar_subopt_beam100": "Beam A* (100)",
    "ucs": "UCS",
    "astar_subopt": "A* (sub.)",
    "astar_opt": "A* (opt.)",
    "ps_subopt_beam1": "Greedy PS",
    "ps_subopt_beam10": "Beam PS (10)",
    "ps_subopt_beam100": "Beam PS (100)",
    "ps_subopt": "PS (sub.)",
    "ps_opt": "PS (opt.)",
    "pgd": "PGD (100 steps)",
    "pgd_1k": "PGD (1,000 steps)",
    "cost_bound": "Cost bound",
    "success": "Adv. success",
    "eff_index_inv": "Success/time ratio",
    "cost": "Cost",
    "method": "Method",
    "attack": "Attack",
    "attack_type": "Scoring func.",
    "beam_size": "Beam size",
    "model": "Model",
    "time": "Time",
    "TransactionAmt": "Gain",
    "adv_util": "Adv. utility",
    "acc": "Test acc.",
}


def mean(col):
    return col.dropna().mean()


def std(col):
    return col.dropna().std()


# -

pd.set_option("display.float_format", "{:.2f}".format)

# +
comp_models = [
    "tabnet_clean",
    "tabnet_cb_1_wo",
    "tabnet_cb_3_wo",
    "tabnet_cb_10_wo",
    "tabnet_cb_30_wo",
]

plot_df = (
    results.query("cost != 0")
    #     .query("cost_bound != 'value'")
    .query(f"model in {comp_models}")
    .query("method in ['greedy_delta']")
    .rename(columns=renaming_dict)
)

plot_df = plot_df.melt(
    id_vars=["Model", "Method", "Cost bound"],
    value_vars=["Adv. success", "Cost"],
    value_name="Value",
    var_name="Metric",
).replace(renaming_dict)

g = sns.catplot(
    data=plot_df,
    x="Cost bound",
    y="Value",
    col="Metric",
    hue="Model",
    palette="rocket",
    kind="point",
    height=7,
    dodge=0.15,
    sharey=False,
    order=[10, 30, "Gain", "none"],
)

g.set_xticklabels(["10", "30", "Gain", "$\\infty$"])

# +
plot_df = (
    results.query("cost != 0")
    #     .query(f"cost_bound == {repr(eval_settings.gain_col)}")
    .query(f"model in {comp_models}")
    .query("method in ['greedy_delta']")
    .rename(columns=renaming_dict)
)


def model_name_to_defence_type(model_name):
    if "_cb" in model_name:
        return "CB"
    elif "_ub" in model_name:
        return "UB"
    else:
        return "None"


plot_df["Defence type"] = plot_df.apply(
    lambda row: model_name_to_defence_type(row.Model), axis=1
)

plot_df = plot_df.melt(
    id_vars=["Model", "Method", "Cost bound", "Defence type", "Test acc."],
    value_vars=["Adv. success", "Cost"],
    value_name="Value",
    var_name="Metric",
).replace(renaming_dict)

plot_df

# +
comp_models = [
    "tabnet_clean",
    "tabnet_cb_1_wo",
    "tabnet_cb_3_wo",
    "tabnet_cb_10_wo",
    "tabnet_cb_30_wo",
    "tabnet_ub_0",
    "tabnet_ub_100",
    "tabnet_ub_200",
    "tabnet_ub_500",
]

plot_df = (
    results.query("cost != 0")
    #     .query(f"cost_bound == {repr(eval_settings.gain_col)}")
    .query(f"model in {comp_models}")
    .query("method in ['greedy_delta']")
    .rename(columns=renaming_dict)
)


def model_name_to_defence_type(model_name):
    if "_cb" in model_name:
        return "CB"
    elif "_ub" in model_name:
        return "UB"
    else:
        return "None"


plot_df["Defence type"] = plot_df.apply(
    lambda row: model_name_to_defence_type(row.Model), axis=1
)

plot_df = plot_df.melt(
    id_vars=["Model", "Method", "Cost bound", "Defence type", "Test acc."],
    value_vars=["Adv. success", "Cost"],
    value_name="Value",
    var_name="Metric",
).replace(renaming_dict)
plot_df["Value"] = plot_df["Value"].astype(np.float)

sns.relplot(
    data=plot_df,
    x="Test acc.",
    y="Value",
    hue="Defence type",
    col="Cost bound",
    row="Metric",
    kind="line",
    height=7,
)
# -

model_to_ub_params = {}
for model in model_zoo.keys():
    config_match = re.search("tabnet_ub_(\d+)", model)
    if config_match is not None:
        param = int(config_match.group(1))
    elif model == "tabnet_clean":
        param = 1000
    else:
        param = None
    model_to_ub_params[model] = param

# +
comp_models = [
    "tabnet_ub_300000",
    "tabnet_ub_400000",
    "tabnet_ub_600000",
    "tabnet_ub_800000",
    "tabnet_clean",
]
plot_df = (
    results
    .query("cost != 0")
    .query(f"model in {comp_models}")
    .query("method in ['greedy_delta']")
    .rename(columns=renaming_dict)
)

plot_df = plot_df.melt(
    id_vars=["Model", "Method", "Cost bound"],
    value_vars=["Adv. utility", "Adv. success", "Cost"],
    value_name="Value",
    var_name="Metric",
).replace(renaming_dict)

plot_df.Value = plot_df.Value.astype(float)

g = sns.catplot(
    data=plot_df,
    x="Cost bound",
    y="Value",
    col="Metric",
    hue="Model",
    palette="rocket",
    kind="point",
    height=10,
    dodge=0.15,
    sharey=False,
    order=core_bounds,
    hue_order=[renaming_dict[model_name] for model_name in comp_models],
)

g.set_xlabels(r"Cost bound $\varepsilon$")
# g.set_xticklabels(["10", "30", r"Gain ($\tau = 0)$", "$\\infty$"])
# plt.savefig("../images/homecredit_ub_vs_cb_cost.pdf", bbox_extra_artists=(g.legend,), bbox_inches='tight')

# +
comp_models = [
    "tabnet_ub_300000",
    "tabnet_ub_400000",
    "tabnet_ub_600000",
    "tabnet_ub_800000",
    "tabnet_clean",
]

plot_df = (
    results
    .query("cost != 0")
#     .query("success == 1")
#     .query("cost_bound != 'value'")
    .query(f"model in {comp_models}")
    .query("method in ['greedy_delta']")
    .rename(columns=renaming_dict)
)

plot_df = plot_df.melt(
    id_vars=["Model", "Method", "Cost bound"],
    value_vars=["Adv. utility", "Adv. success", "Cost"],
    value_name="Value",
    var_name="Metric",
).replace(renaming_dict)

plot_df.Value = plot_df.Value.astype(float)

g = sns.catplot(
    data=plot_df,
    x="Cost bound",
    y="Value",
    col="Metric",
    hue="Model",
    palette="rocket",
    height=5,
    kind="point",
    dodge=0.15,
    sharey=False,
    order=gain_bounds,
    hue_order=[renaming_dict[model_name] for model_name in comp_models],
)

g.set_xlabels(r"Margin $\tau$")
g.set_xticklabels(taus)
# plt.savefig("../images/homecredit_ub_vs_ub_cost.pdf", bbox_extra_artists=(g.legend,), bbox_inches='tight')
# -


