# ---
# jupyter:
#   jupytext:
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
MODEL_PATH = "../models/ieeecis_{}.pkl"
RESULTS_PATH = "../out/ieeecis_{}.pkl"
LOAD_MODELS = False

# %load_ext autoreload
# %autoreload 2

# +
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

sys.path.append("..")
# -

from loaders.ieeecis import load_ieeecis
from src.utils.data import one_hot_encode
from src.utils.data import diff

df = load_ieeecis()

df = one_hot_encode(df, binary_vars=["isFraud"], standardize=True)

df.TransactionAmt

X = df.drop(columns="isFraud")
y = df["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=3000, random_state=SEED
)

del df

X_resampled, y_resampled = RandomUnderSampler(random_state=SEED).fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=3000, random_state=SEED
)

print(X_train.shape)

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
print(dummy_clf.score(X_test, y_test))

model_zoo = {
    # hidden_layer_sizes=[[1000], [1000, 1000], [2000], [2000, 2000]],
    "nn": MLPClassifier(hidden_layer_sizes=[6], random_state=SEED,),
}

for model_name, clf in tqdm.tqdm(model_zoo.items()):
    model_filepath = MODEL_PATH.format(model_name)
    if not LOAD_MODELS:
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
    print(f"{model_name}: {acc}")
