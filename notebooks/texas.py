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
# %load_ext autoreload
# %autoreload 2
# +
import sys

sys.path.append("..")

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

# -

from loaders.texas import load_texas

df = load_texas()

df.head()

clf = LogisticRegressionCV()
clf.fit(df)
