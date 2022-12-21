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
MODEL_PATH = "../models/kddcup99_{}.pkl"
LOAD_MODELS = False

# %load_ext autoreload
# %autoreload 2

# +
import sys
import pickle

import pandas as pd

from tqdm import notebook as tqdm
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

sys.path.append("..")
# -

from loaders.kddcup99 import load_kddcup99

df = load_kddcup99()
df

cat_cols = df.select_dtypes(["category"]).columns
df = pd.concat([df.drop(columns=cat_cols), pd.get_dummies(df[cat_cols])], axis=1)

X = df.drop(columns="label")
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=3000, random_state=SEED
)

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_test, y_test)

model_zoo = {
    # 5-fold CV using default LogisticRegressionCV
    "lr": LogisticRegression(C=1291),
    # Based on OOB scores using the grid:
    # num_estimators=[10, 100, 500]
    # max_features=["sqrt", "log2"]
    "rf": RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_split=min_samples_split,
        oob_score=True,
    ),
    # hidden_layer_sizes=[[100], [100, 100], [100, 100, 100]],
    "nn": MLPClassifier(hidden_layer_sizes=[100],),
}

if not LOAD_MODELS:
    for model_name, clf in tqdm.tqdm(model_zoo.items()):
        clf.fit(X_train, y_train)
        with open(MODEL_PATH.format(model_name), "wb") as f:
            pickle.dump(clf, f)

for model_name, clf in model_zoo.items():
    print(f"{model_name}: {clf.score(X_test, y_test)}")
