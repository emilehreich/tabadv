import argparse
import os
import pickle
import sys
import time
import typing
#import ckwrap
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import lightgbm as lgb
from scipy import stats

sys.path.append("..")
sys.path.append(".")

from src.utils.data import one_hot_encode
from src.utils.data import diff
from src.utils.hash import fast_hash
from src.utils.counter import ExpansionCounter, CounterLimitExceededError
from src.transformations import TransformationGenerator
from src.transformations import CategoricalFeature, NumFeature
from src.search import a_star_search as generalized_a_star_search

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import autonotebook as tqdm

from utils import *
from train import default_model_dict as model_dict

from loaders import shape_dict
from exp.framework import ExperimentSuite
from exp.utils import TorchWrapper, EmbWrapper
from exp import settings
from exp.settings import get_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--results_dir", default="../out", type=str)
    parser.add_argument(
        "--dataset",
        default="ieeecis",
        choices=["ieeecis", "twitter_bot", "home_credit", "syn"],
        type=str,
    )
    parser.add_argument("--attack", default="greedy", type=str)
    parser.add_argument("--embs", default="", type=str)
    parser.add_argument("--cost_bound", default=None, type=float)
    parser.add_argument("--tr", default=0.0, type=float)
    parser.add_argument("--model_path", default="../models/default.pt", type=str)
    parser.add_argument(
        "--utility_type",
        default="success_rate",
        choices=["maximum", "satisficing", "cost-restrictred", "average-attack-cost", "success_rate"],
        type=str,
    )
    parser.add_argument(
        "--satisfaction-value", default=-1, type=float, help="Value for satisfaction"
    )
    parser.add_argument(
        "--max-cost-value", default=-1, type=float, help="Max-cost value"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--test-lim", default=10000, type=int)
    parser.add_argument("--noise", default="0", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")

    return parser.parse_args()

def cluster_vec(vec, t):
    idx = np.argsort(vec)
    svec = vec.copy()
    mns = []
    for i, a in enumerate(idx):
        if mns == []:
            mns.append(vec[a])
            continue
        elif vec[a] - mns[-1] < t:
            svec[a] = mns[-1]
            continue
        else:
            mns.append(vec[a])
    return svec
    

def prone_embs(embs, t=0.01):
    for emb in embs:
        w = emb.weight.detach().cpu().numpy()
        #print(w.shape)
        if w.shape[0] == 1:
            continue # do not treat numeric yet
        for i in range(w.shape[0]):
            w[i] = cluster_vec(w[i], t)
            #print(emb.weight[i], w[i])
        emb.weight = torch.nn.Parameter(torch.Tensor(w).to(emb.weight.device))

def get_embs_dists(embs):
    dists = []
    for emb in embs:
        w = emb.weight.detach().cpu().numpy()
        if w.shape[0] == 1:
            continue # do not treat numeric yet
        for i in range(w.shape[1]):
            for j in range(w.shape[1]):
                for k in range(w.shape[0]):
                    if i != j:
                        d = np.abs(w[k,i] - w[k,j])
                        dists.append(d)
    return np.array(dists)

            
def dump_costs_embs(costs, embs, n=50):
    for i in range(n):
        name = "cat" + str(i)
        print("Cost matrix for ", name)
        print(costs[name])
        print("Embeddings for ", name)
        print(embs[i].weight.T)
    
    for i in range(n):
        name = "num" + str(i)
        print("Cost matrix for ", name)
        print(costs[name])
        print("Embeddings for ", name)
        print(embs[n+i].weight.T)

def get_utility(results, cost_orig, X_test, y, mode="maximum", cost=-1, t_value=-1):
    total_ut = 0
    divider = 0
    for i, r in results.iterrows():
        #print(i, r.cost, cost_orig.iloc[int(r.orig_index)], r.orig_index)
        #if y.iloc[int(r.orig_index)] == 0: No need for it, only y = target is evaluated here
        #    continue
        if r.cost is None:
            if mode == "success_rate":
                divider += 1
            continue
        else:
            if mode == "maximum":
                total_ut += max(cost_orig.iloc[int(r.orig_index)] - r.cost, 0)
            elif mode == "satisficing":
                if cost_orig.iloc[int(r.orig_index)] - r.cost > t_value:
                    total_ut += max(cost_orig.iloc[int(r.orig_index)] - r.cost, 0)
            elif mode == "cost-restrictred":
                if r.cost < cost:
                    total_ut += max(cost_orig.iloc[int(r.orig_index)] - r.cost, 0)
            elif mode == "average-attack-cost":
                if r.cost > 0.001:
                    total_ut += r.cost
                    divider += 1
            elif mode == "success_rate":
                if r.cost > 0.001:
                    total_ut += 1
                divider += 1
    
    if mode == "average-attack-cost" or mode == "success_rate":
        return total_ut / divider
    else:
        return total_ut / len(X_test)


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # print("Cuda Device Available")
        # print("Name of the Cuda Device: ", torch.cuda.get_device_name())
        # print("GPU Computational Capablity: ", torch.cuda.get_device_capability())
    else:
        device = torch.device("cpu")

    eval_settings = settings.setup_dataset_eval(args.dataset, args.data_dir, seed=0, noise = args.noise)
    eval_settings.working_datasets.X_test = eval_settings.working_datasets.X_test[:args.test_lim]
    eval_settings.working_datasets.y_test = eval_settings.working_datasets.y_test[:args.test_lim]
    experiment_path = settings.get_experiment_path(
        args.results_dir, args.model_path, args.attack, args.cost_bound, embs=args.embs, tr=args.tr
    )
    print(experiment_path)
    if os.path.isfile(experiment_path) and not args.force:
        print(f"{experiment_path} already exists. Skipping attack...")
    else:
        if args.model_path == "lgbm":
                data_train = get_dataset(
                    args.dataset, args.data_dir, mode="train", cat_map=True, noise=args.noise
                )
                data_test = get_dataset(
                    args.dataset, args.data_dir, mode="test", cat_map=True, noise=args.noise
                )
                clf = lgb.LGBMClassifier()
                if args.embs != "":
                    emb_model = model_dict[args.dataset](inp_dim=shape_dict[args.dataset], cat_map=data_train.cat_map).to(device)
                    emb_model.load_state_dict(torch.load(args.embs))
                    np.set_printoptions(precision=4)
                    #dump_costs_embs(data_test.costs, emb_model.emb_layers)
                    dsts = get_embs_dists(emb_model.emb_layers)
                    tr = stats.scoreatpercentile(dsts, args.tr * 100)
                    print("Threshold: " + str(tr))
                    prone_embs(emb_model.emb_layers, t=tr)
                    clf = EmbWrapper(clf, emb_model.emb_layers, emb_model.cats, device)

                clf.fit(data_train.X_train, data_train.y_train)
                y_pred=clf.predict(data_test.X_test)
                print(classification_report(data_test.y_test, y_pred))
        else:
            net = model_dict[args.dataset](inp_dim=shape_dict[args.dataset], cat_map=eval_settings.working_datasets.cat_map).to(device)
            net.load_state_dict(torch.load(args.model_path))
            net.eval()
            clf = TorchWrapper(net, device)

        if args.attack == "Ballet":
            eval_settings.working_datasets.orig_df["isFraud"] = \
                eval_settings.working_datasets.orig_df["isFraud"].astype('float')
            cr = eval_settings.working_datasets.orig_df.corr()
            corr_vec = cr["isFraud"]
            corr_vec.drop(index='isFraud')
            corr_vec = corr_vec.to_numpy(dtype=np.float32)
            for i, w in enumerate(eval_settings.working_datasets.w):
                if w < 10000:
                    eval_settings.working_datasets.w[0,i] = np.abs(corr_vec[i]) / (np.norm(corr_vec[i]) ** 2)

        exp_suite = ExperimentSuite(
            clf,
            eval_settings.working_datasets.X_test,
            eval_settings.working_datasets.y_test,
            target_class=eval_settings.target_class,
            cost_bound=args.cost_bound,
            spec=eval_settings.spec,
            gain_col=eval_settings.gain_col,
            dataset=eval_settings.working_datasets.dataset,
            iter_lim=100
        )
        preds = clf.predict(eval_settings.working_datasets.X_test)
        #print(preds[0], eval_settings.working_datasets.X_test.head())
        print("Acc: ", sum(preds == eval_settings.working_datasets.y_test) / len(eval_settings.working_datasets.X_test))

        attack_config = {a.name: a for a in eval_settings.experiments}[args.attack]

        results = exp_suite.run(attack_config)
        #results['x']
        results.to_pickle(experiment_path)

    result = pd.read_pickle(experiment_path)
    ut = get_utility(
            result,
            eval_settings.working_datasets.orig_cost,
            eval_settings.working_datasets.X_test,
            eval_settings.working_datasets.orig_y,
            mode=args.utility_type,
            cost=args.max_cost_value,
            t_value=args.satisfaction_value,
        )
    print(ut)
    if (args.utility_type == "success_rate"):
        print("Rob Acc: ", 1 - ut)

if __name__ == "__main__":
    main()
