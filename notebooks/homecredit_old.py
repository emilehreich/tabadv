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
FORCE = False

# !mkdir -p {RESULTS_PATH}

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import os
import sys
import time
import pickle
import typing

import pandas as pd
import numpy as np
import seaborn as sns

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

data_train = settings.get_dataset("home_credit", "../data", mode="train")
data_test = settings.get_dataset("home_credit", "../data", mode="test")
X_train, y_train = data_train.X_train, data_train.y_train
X_test, y_test = data_test.X_test, data_test.y_test


model_zoo = {
    # 5-fold CV using default LogisticRegressionCV
    #     "lr": LogisticRegression(C=0.04641589),
    #     "xgbt": XGBClassifier(n_estimators=500, random_state=SEED,),
    "clean": utils.load_torch_model(
        "../models/home_credit/clean.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
    "tabnet_cb_10": utils.load_torch_model(
        "../models/home_credit/l1_10.0.pt",
        dataset="home_credit",
        model_label="tabnet_homecredit",
        device="cuda",
    ),
    #     "tabnet_cb_200": utils.load_torch_model(
    #         "../models/home_credit/l1_200.0.pt",
    #         dataset="home_credit",
    #         model_label="tabnet_homecredit",
    #         device="cuda",
    #     ),
    #     "tabnet_cb_1000": utils.load_torch_model(
    #         "../models/home_credit/l1_1000.0.pt",
    #         dataset="home_credit",
    #         model_label="tabnet_homecredit",
    #         device="cuda",
    #     ),
    #     "tabnet_cb_2000": utils.load_torch_model(
    #         "../models/home_credit/l1_2000.0.pt",
    #         dataset="home_credit",
    #         model_label="tabnet_homecredit",
    #         device="cuda",
    #     ),
    #     "tabnet_ub_200K": utils.load_torch_model(
    #         "../models/home_credit/ub_200000.0.pt",
    #         dataset="home_credit",
    #         model_label="tabnet_homecredit",
    #         device="cuda",
    #     ),
    #     "tabnet_ub_500K": utils.load_torch_model(
    #         "../models/home_credit/ub_500000.0.pt",
    #         dataset="home_credit",
    #         model_label="tabnet_homecredit",
    #         device="cuda",
    #     ),
    #     "tabnet_ub_1M": utils.load_torch_model(
    #         "../models/home_credit/ub_1000000.0.pt",
    #         dataset="home_credit",
    #         model_label="tabnet_homecredit",
    #         device="cuda",
    #     ),
}

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
    acc = clf.score(X_test, y_test)
    model_accuracies[model_name] = acc
    print(f"{model_name}: {acc:.2%}")

eval_settings = settings.setup_dataset_eval("home_credit", "../data", seed=0)
cost_bounds = ["AMT_CREDIT", None, 10, 20]

for model_name, clf in model_zoo.items():
    print(f"=== Model: {model_name} ===")

    for cost_bound in cost_bounds:
        exp_suite = ExperimentSuite(
            clf,
            eval_settings.working_datasets.X_test,
            eval_settings.working_datasets.y_test,
            target_class=eval_settings.target_class,
            cost_bound=cost_bound,
            spec=eval_settings.spec,
        )

        for experiment in eval_settings.experiments:
            if experiment.name not in ["greedy_delta"]:
                continue

            # We only need to run UCS without the cost bound.
            if experiment.name == "ucs" and cost_bound is not None:
                continue

            experiment_path = settings.get_experiment_path(
                RESULTS_PATH, model_name, experiment.name, cost_bound
            )
            if os.path.isfile(experiment_path) and not FORCE:
                # "Already exists. Skipping attack...
                continue

            if ("_opt" in experiment.name and model_name != "lr") or (
                "ps_" in experiment.name and cost_bound is None
            ):
                # Incompatible algorithm. Skipping...
                continue

            print(f"{experiment.name} {cost_bound=}")
            print(experiment_path)

            results = exp_suite.run(experiment)
            results.to_pickle(experiment_path)

results_dict = {}

for model_name, clf in model_zoo.items():
    for experiment in eval_settings.experiments:
        for cost_bound in cost_bounds:
            experiment_path = settings.get_experiment_path(
                RESULTS_PATH, model_name, experiment.name, cost_bound,
            )
            if experiment_path in results_dict:
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
                    df = pd.read_pickle(experiment_path)
                    success_idx = ~df.adv_x.isna()

            except FileNotFoundError:
                continue

            print(experiment_path)

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
    #     "lr": f"LR (Acc: {model_accuracies['lr']:0.2f})",
    #     "xgbt": f"XGBT (Acc: {model_accuracies.get('xgbt'):0.2f})",
    "random": "Random search",
    "hc": "Hill climbing",
    "greedy": "Benefit-Cost",
    "greedy_delta": "Delta Benefit-Cost",
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
}


def mean(col):
    return col.dropna().mean()


def std(col):
    return col.dropna().std()


# +
def compute_eff_indices(df):
    result = df.copy()
    result["eff_index"] = df.time.mean() / df.success.mean()
    result["eff_index_inv"] = df.success.mean() / df.time.mean()
    return result.reset_index().drop(columns=["model", "cost_bound", "method"])


attack_summary = (
    results.query("cost != 0")
    .query(
        "model in ['tabnet_cb_1', 'tabnet_cb_10', 'tabnet_cb_100', 'tabnet_cb_300', 'tabnet_cb_1000']"
    )
    .groupby(["cost_bound", "method", "model"])
    .apply(compute_eff_indices)
    .reset_index()
)

(
    attack_summary.groupby(["cost_bound", "method", "model"])
    .agg({"cost": mean, "time": mean, "success": mean, "eff_index_inv": mean})
    .unstack(level=0)
    .loc[:, ["time", "success", "eff_index_inv"]]
)

# +
order = ["greedy", "greedy_delta", "ps_subopt_beam1", "astar_subopt_beam1"]
g = sns.catplot(
    data=(
        attack_summary.melt(
            id_vars=["method", "model", "cost_bound"],
            value_vars=["success", "eff_index_inv"],
        )
    ),
    x="cost_bound",
    y="value",
    hue="method",
    col="variable",
    hue_order=order,
    #     hue_order=[renaming_dict[algo] for algo in order],
    markers=["^", "s", "o", "v"],
    kind="point",
    sharey=False,
    order=["AMT_CREDIT", 10, 20, "none"],
)

g.set(xticklabels=["Gain", "$10", "$20", "Unbounded"])
g.set(xlabel="Cost bound")

# +
comp_models = ["tabnet_cb_1", "tabnet_cb_10", "tabnet_cb_100"]

plot_df = (
    results.query("cost != 0")
    .query("cost_bound != 'value'")
    .query(f"model in {comp_models}")
    .query("method in ['ucs']")
)

plot_df = plot_df.melt(
    id_vars=["model", "method", "cost_bound"], value_vars=["success", "cost", "acc"]
)
sns.catplot(
    data=plot_df,
    x="cost_bound",
    y="value",
    col="variable",
    hue="model",
    order=[10, 20, "none"],
    color_pallette="rocket",
    kind="point",
    dodge=0.15,
    sharey=False,
)
# -

clf = model_zoo["rf"]
feature_importances = dict(
    sorted(zip(X_train.columns, clf.feature_importances_), key=lambda t: -t[1])
)
