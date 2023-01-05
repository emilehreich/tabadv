import pickle
import sys
import time
import typing
import warnings
import random

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
from tqdm import notebook as tqdm

sys.path.append("..")
# -

from src.utils.data import one_hot_encode
from src.utils.data import diff
from src.utils.hash import fast_hash

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import ast


def get_w_vec(df, weights, one_hot=True, sep="_", num_list=[]):
    col_names = df.columns.values
    w = torch.zeros(len(col_names))
    for i, c in enumerate(col_names):
        if sep in c and one_hot == True:
            if c in num_list:
                name = c
            else:
                name = sep.join((c.split(sep)[:-1]))
        else:
            name = c
        w[i] = weights.get(name, 100000.0)
    return w

def get_w_rep(X, costs, cat_map, sep="_", num_list=[], w_max=0.0):

    w_rep = np.zeros_like(X)
    for i, num in enumerate(num_list):
        w_rep[:, i] = costs[num]

    for name, cost in costs.items():
        if name not in num_list:
            i, j = cat_map[name]
            if w_max != 0.0:
                cost[cost >= w_max] = 100000.0
            w_rep[:, i:j+1] = np.matmul(X[:, i:j+1], cost)
            #print(name, w_rep[0, i:j+1],X[0, i:j+1], cost)
        else:
            pass
    return w_rep

def get_cat_map(df, cat_var, bin_var=[], sep="_"):
    col_names = df.columns.values
    c_map = {}
    for i, c in enumerate(col_names):
        if sep in c:
            if c in bin_var:
                c_map[c] = [i, i]
                continue
            name = sep.join((c.split(sep)[:-1]))
            if name in cat_var:
                if name in c_map:
                    c_map[name][1] = i
                else:
                    c_map[name] = [i, i]
    return c_map

class CreditCard(Dataset):
    def __init__(
        self,
        folder_path,
        mode="train",
        seed=0,
        balanced=False,
        discrete_one_hot=False,
        same_cost=False,
        cat_map=False,
        ballet=False
    ):
        self.max_eps = 1000.0
        self.mode = mode
        self.same_cost = same_cost
        df = pd.read_csv(f"{folder_path}/credit_sim/credit_card_transactions-balanced-v2.csv", index_col=0)
        self.gain_col = "Amount"
        numerical_columns = [
            "Month",
            "Day",
            "Hour",
            "Minutes",
            "Amount"
        ]
        categorical_columns = [
            "Use Chip",
            "MCC_City",
            "Errors",
            "card_brand",
            "card_type",
            "Fraud",
        ]
        weights = {
            "Use Chip": 20,
            "card_brand": 20,
            "card_type": 20,
            "Month": 0.1,
            "Day": 0.1,
            "Hour": 0.1,
            "Minutes": 0.1,
            "MCC_City": 100,
            "Errors": 10,
        }
        df = df[numerical_columns + categorical_columns]
        #for categorical_column in categorical_columns:
        #    df[categorical_column].fillna("NULL", inplace=True)
        #    df[categorical_column] = df[categorical_column].astype('category',copy=False)

        df = one_hot_encode(
            df,
            cat_cols=categorical_columns,
            num_cols=numerical_columns,
            binary_vars=["Fraud"],
            standardize=False,
            prefix_sep="_",
        )

        self.cat_map = get_cat_map(df, categorical_columns, sep='_')
        w_vector = get_w_vec(
            df.drop(columns=["Fraud"]), weights, sep="_", num_list=(numerical_columns)
        ).unsqueeze(0)
        self.w = w_vector

        df["Fraud"] = np.where(
            df["Fraud"].str.contains("Y"), "1", "0"
        )
        y = df["Fraud"]
        X = df.drop(columns=["Fraud"])
        X_train, X_test, y_train, y_test = train_test_split( # ALREADY BALANCED DATA
                X, y, test_size=3000, random_state=seed
            )

        del X
        del y
        del df

        if mode == "train":
            self.X_train = X_train
            self.y_train = y_train
            self.inp = X_train.to_numpy(dtype=np.float32)
            self.oup = y_train.to_numpy(dtype=np.float32)
            #self.cost_orig = cost_orig.iloc[X_train.index].to_numpy(dtype=np.float32)
            self.cost_orig = X_train[self.gain_col].to_numpy(dtype=np.float32)
            #print(self.inp[2], self.cost_orig[2])

        elif mode == "test":
            self.X_test = X_test
            self.y_test = y_test
            self.inp = X_test.to_numpy(dtype=np.float32)
            self.oup = y_test.to_numpy(dtype=np.float32)
            self.cost_orig = X_test[self.gain_col].to_numpy(dtype=np.float32)

        else:
            raise ValueError

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inpt = torch.Tensor(self.inp[idx])
        oupt = torch.Tensor([self.oup[idx]])
        if self.same_cost:
            cost = torch.Tensor([1.0])
        else:
            cost = torch.Tensor([self.cost_orig[idx]])
        return (inpt, oupt, cost)

class IEEECISDataset(Dataset):
    def __init__(
        self,
        folder_path,
        mode="train",
        seed=0,
        balanced=False,
        discrete_one_hot=False,
        same_cost=False,
        cat_map=False,
        ballet=False
    ):
        self.max_eps = 20.4
        self.mode = mode
        self.same_cost = same_cost
        train_identity = pd.read_csv(f"{folder_path}/ieeecis/train_identity.csv")
        train_transaction = pd.read_csv(f"{folder_path}/ieeecis/train_transaction.csv")

        df = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")
        cost_orig = df["TransactionAmt"]
        self.gain_col = "TransactionAmt"
        self.cost_orig_df = cost_orig
        if discrete_one_hot:
            num_cols = [
                # "TransactionID",
                # "TransactionDT",
                "TransactionAmt",
                "card1",
                "card2",
                "card3",
                "card5",
                "addr1",
                "addr2",
                # "dist1",
                # "dist2",
            ]

            cat_cols = [
                "ProductCD",
                "card4",
                "card6",
                "P_emaildomain",
                "R_emaildomain",
                "DeviceType",
                # "DeviceInfo",
                "isFraud",
            ]

            weights = {
                # Do not need it anymore
                #"TransactionAmt": 1000000,
                #"card1": 1000000,  # 2,
                #"card2": 1000000,  # 3,
                #"card3": 1000000,  # 4,
                #"card5": 1000000,  # 5,
                #"addr1": 1000000,  # 6, 
                #"addr2": 1000000,  # 7,
                #"ProductCD": 1,
                "card_type": 20,
                # "card6":1 ,
                "P_emaildomain": 0.2,
                # "R_emaildomain": 0.2,
                "DeviceType": 0.1,
            }

            one_hot_vars = ["card_type", "P_emaildomain", "DeviceType"]
            # let's combine the data and work with the whole dataset
            df = df[num_cols + cat_cols]
            df[cat_cols] = df[cat_cols].astype("category")

            df = df[~df[num_cols].isna().any(axis=1)]

            num_cols = df.select_dtypes(include=np.number).columns.tolist()

            df["card_type"] = (
                df["card4"].astype("string") + "-" + df["card6"].astype("string")
            ).astype("category")
            df = df.drop(columns=["card4", "card6"])
            df = one_hot_encode(
                df, binary_vars=["isFraud"], standardize=False, prefix_sep="_"
            )
            if cat_map:
                self.cat_map = get_cat_map(df, one_hot_vars, sep='_')
            else:
                self.cat_map = None
        else:
            df = (
                df.select_dtypes(exclude=["object"])
                .dropna(axis=1, how="any")
                .drop(columns=["TransactionID", "TransactionDT"])
            )

        X = df.drop(columns="isFraud")
        y = df["isFraud"]
        self.orig_df = df
        w_vector = get_w_vec(X, weights, one_hot=True, sep="_").unsqueeze(0)
        self.w = w_vector
        if balanced:
            X_resampled, self.y_resampled = RandomUnderSampler(
                random_state=seed
            ).fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, self.y_resampled, test_size=3000, random_state=seed
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=3000, random_state=seed
            )

        del X
        del y
        del df

        if mode == "train":
            self.X_train = X_train
            self.y_train = y_train
            self.inp = X_train.to_numpy(dtype=np.float32)
            self.oup = y_train.to_numpy(dtype=np.float32)
            #self.cost_orig = cost_orig.iloc[X_train.index].to_numpy(dtype=np.float32)
            self.cost_orig = X_train[self.gain_col].to_numpy(dtype=np.float32)
            #print(self.inp[2], self.cost_orig[2])

        elif mode == "test":
            self.X_test = X_test
            self.y_test = y_test
            self.inp = X_test.to_numpy(dtype=np.float32)
            self.oup = y_test.to_numpy(dtype=np.float32)
            self.cost_orig = X_test[self.gain_col].to_numpy(dtype=np.float32)

        else:
            raise ValueError

        self.mean = 1  # np.mean(self.cost_orig)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inpt = torch.Tensor(self.inp[idx])
        oupt = torch.Tensor([self.oup[idx]])
        if self.same_cost:
            cost = torch.Tensor([1.0])
        else:
            cost = torch.Tensor([self.cost_orig[idx] / self.mean])
        return (inpt, oupt, cost)


def _transform_source_identity(X_k, sources_count=7):
    """
    Helper to transform the source_identity field.
    """
    X_k = X_k.apply(lambda x: x.replace(";", ","))
    X_k = X_k.apply(ast.literal_eval)

    N, K = X_k.shape[0], sources_count * 2
    X_k_transformed = np.zeros((N, K), dtype="intc")

    # Set (1, 0) if the source is present for the user and (0, 1) if absent.
    for i in range(N):
        for j in range(sources_count):
            if j in X_k[i]:
                X_k_transformed[i, j * 2] = 1
            else:
                X_k_transformed[i, j * 2 + 1] = 1

    return X_k_transformed


class TwitterBotDataset(Dataset):
    def __init__(
        self,
        folder_path,
        mode="train",
        seed=0,
        balanced=False,
        discrete_one_hot=False,
        same_cost=False,
        drop_features=None,
        cat_map=False,
    ):
        one_hot_vars = []
        self.max_eps = 0.0
        # human_datasets = [f"{folder_path}/twitter_bots/humans/humans.{num}.csv" for num in ['1k', '100k', '1M', '10M']]
        # bot_datasets = [f"{folder_path}/twitter_bots/bots/bots.{num}.csv" for num in ['1k', '100k', '1M', '10M']]
        self.mode = mode
        self.same_cost = same_cost
        if drop_features is None:
            # Default features that will be removed.
            drop_features = [
                "follower_friend_ratio",
                "tweet_frequency",
                "favourite_tweet_ratio",
                # "user_tweeted",
                # "user_replied",
                # "likes_per_tweet",
                # "retweets_per_tweet",
                # "urls_count",
                # "cdn_content_in_kb",
                # "sources_count",
                # "user_retweeted",
                # "age_of_account_in_days",
                # "source_identity_news",
            ]

        weights = {
            "user_tweeted": 2,
            "user_replied": 2,
            "likes_per_tweet": 0.025,
            "retweets_per_tweet": 0.025,
            # "source_identity_mobile": 100,
        }

        # For robust baseline
        #for f, w in weights.items():
        #    weights[f] = -1
        #nrint(weights)
        values = {"1k": 1, "100k": 100, "1M": 1000, "10M": 10000}
        # Load data for humans.
        df = pd.DataFrame()
        for num in ["1k", "100k", "1M", "10M"]:
            # for num in ['100k']:
            human_dataset = f"{folder_path}/twitter_bots/humans/humans.{num}.csv"
            df1 = pd.read_csv(human_dataset)
            df1 = df1.drop("screen_name", axis=1)  # remove screen_name column
            df1 = df1.assign(is_bot=0)
            df1 = df1.assign(value=values[num])
            df = df.append(df1, ignore_index=True)

            # Load data for bots.
            bot_dataset = f"{folder_path}/twitter_bots/bots/bots.{num}.csv"
            df2 = pd.read_csv(bot_dataset)
            df2 = df2.drop("screen_name", axis=1)  # remove screen_name column
            df2 = df2.assign(is_bot=1)
            df2 = df2.assign(value=values[num])
            df = df.append(df2, ignore_index=True)

        # Concatenate dataframes.

        # Drop unwanted features.
        self.mean = 1

        for column in df:

            # Source identity and is_bot are not quantizable.
            if column == "source_identity" or column == "is_bot" or column == "value":
                continue

            # Drop feature if there is only 1 distinct value.
            if np.unique(df[column]).size == 1:
                warnings.warn(
                    "Dropping feature because only one unique value: %s" % column
                )
                df = df.drop(column, axis=1)
                continue

        # Encode 'source_identity' field by setting '1's if source is present.
        transformed = _transform_source_identity(df.loc[:, "source_identity"])

        df = df.drop("source_identity", axis=1)
        df["source_identity_other"] = transformed[:, 0]
        df["source_identity_browser"] = transformed[:, 2]
        df["source_identity_mobile"] = transformed[:, 4]
        df["source_identity_osn"] = transformed[:, 6]
        df["source_identity_automation"] = transformed[:, 8]
        df["source_identity_marketing"] = transformed[:, 10]
        df["source_identity_news"] = transformed[:, 12]

        df = df.drop(drop_features, axis=1)
        # df = one_hot_encode(df, binary_vars=["is_bot"], standardize=False, prefix_sep='?')
        self.orig_df = df
        if cat_map:
            self.cat_map = get_cat_map(df, one_hot_vars, sep='_')
        else:
            self.cat_map = None
        X = df.drop(columns="is_bot")
        w_vector = get_w_vec(X, weights, one_hot=False, sep="?").unsqueeze(0)
        self.w = w_vector
        y = df["is_bot"]
        cost_orig = df["value"]
        self.cost_orig_df = cost_orig
        self.gain_col = "value"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        if mode == "train":
            self.X_train = X_train
            self.y_train = y_train
            self.inp = X_train.to_numpy(dtype=np.float32)
            self.oup = y_train.to_numpy(dtype=np.float32)
            self.cost_orig = X_train[self.gain_col].to_numpy(dtype=np.float32)

        elif mode == "test":
            self.X_test = X_test
            self.y_test = y_test
            self.inp = X_test.to_numpy(dtype=np.float32)
            self.oup = y_test.to_numpy(dtype=np.float32)
            self.cost_orig = X_test[self.gain_col].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inpt = torch.Tensor(self.inp[idx])
        oupt = torch.Tensor([self.oup[idx]])
        if self.same_cost:
            cost = torch.Tensor([1.0])
        else:
            cost = torch.Tensor([self.cost_orig[idx] / self.mean])
        # (oupt.shape, oupt)
        return (inpt, oupt, cost)


class TexasDataset(Dataset):
    def __init__(
        self,
        folder_path,
        mode="train",
        seed=0,
        balanced=False,
        discrete_one_hot=False,
        same_cost=False,
        drop_features=None,
        bins=None,
        cat_map=False,
    ):
        """Load Texas Hospital Discharge dataset.

        Code is based on a script by Theresa Stadler.
        """
        cat_cols = [
            "TYPE_OF_ADMISSION",
            "PAT_STATE",
            "PAT_ZIP",
            "SEX_CODE",
            "RACE",
            "ETHNICITY",
            "ILLNESS_SEVERITY",
            "ADMITTING_DIAGNOSIS",
            "PRINC_DIAG_CODE",
            "RISK_MORTALITY",
            # "PAT_STATUS",
            # "ADMIT_WEEKDAY",
            # "DISCHARGE",
        ]
        num_cols = [
            "LENGTH_OF_STAY",
            "PAT_AGE",
            "TOTAL_CHARGES",
            "TOTAL_NON_COV_CHARGES",
            # "TOTAL_CHARGES_ACCOMM",
            # "TOTAL_NON_COV_CHARGES_ACCOMM",
            # "TOTAL_CHARGES_ANCIL",
            # "TOTAL_NON_COV_CHARGES_ANCIL",
        ]

        dtype = {col: "category" for col in cat_cols}
        q1 = pd.read_csv(
            folder_path + "PUDF_base1_%dq2013_tab.txt" % 1,
            common_delimiter="\t",
            dtype=dtype,
        )
        q2 = pd.read_csv(
            folder_path + "PUDF_base1_%dq2013_tab.txt" % 2, delimiter="\t", dtype=dtype
        )
        q3 = pd.read_csv(
            folder_path + "PUDF_base1_%dq2013_tab.txt" % 3, delimiter="\t", dtype=dtype
        )
        q4 = pd.read_csv(
            folder_path + "PUDF_base1_%dq2013_tab.txt" % 4, delimiter="\t", dtype=dtype
        )

        q1_clean = q1[cat_cols + num_cols]
        q2_clean = q2[cat_cols + num_cols]
        q3_clean = q3[cat_cols + num_cols]
        q4_clean = q4[cat_cols + num_cols]
        combined_df = pd.concat([q1_clean, q2_clean, q3_clean, q4_clean])

        for c in cat_cols + num_cols:
            combined_df.loc[combined_df[c].astype(str) == "`", c] = pd.NA

        c = "TYPE_OF_ADMISSION"
        combined_df.loc[combined_df[c].notnull(), c] = (
            combined_df.loc[combined_df[c].notnull(), c]
            .astype(str)
            .apply(lambda x: x.split(".")[0])
        )

        # Patient status
        # def strip_zeros(x):
        #     s = x.split("0")
        #     if len(s) > 1:
        #         if s[0] == "":
        #             return s[1]
        #         else:
        #             return x
        #     else:
        #         return x

        # c = "PAT_STATUS"
        # combined_df.loc[combined_df[c].notnull(), c] = (
        #     combined_df.loc[combined_df[c].notnull(), c]
        #     .astype("str")
        #     .apply(lambda x: x.split(".")[0])
        #     .apply(lambda x: strip_zeros(x))
        #     .astype("category")
        # )

        c = "RACE"
        combined_df.loc[combined_df[c].notnull(), c] = (
            combined_df.loc[combined_df[c].notnull(), c]
            .astype("str")
            .apply(lambda x: x.split(".")[0])
        )

        c = "ETHNICITY"
        combined_df.loc[combined_df[c].notnull(), c] = (
            combined_df.loc[combined_df[c].notnull(), c]
            .astype("str")
            .apply(lambda x: x.split(".")[0])
        )

        c = "PAT_AGE"
        combined_df.loc[combined_df[c].notnull(), c] = combined_df.loc[
            combined_df[c].notnull(), c
        ].astype(np.float32)

        combined_df[cat_cols] = combined_df[cat_cols].astype("category")

        self.orig_df = combined_df


class Synthetic(Dataset):
    def __init__(
        self,
        folder_path,
        mode="train",
        seed=13,
        same_cost=False,
        cat_map=False, # TODO remove
        noise='0',
        mat_mode=True,
        w_max=0.0
    ):
        self.mode = mode
        self.same_cost = same_cost
        if noise == '0':
            df = pd.read_csv(folder_path + "/syn.csv")
        else:
            df = pd.read_csv(folder_path + "/syn_" + noise + ".csv")

        if mat_mode:
            costs = np.load(folder_path + "/syn_" + noise + "_costs.npy", allow_pickle=True)[()]
            self.costs = costs

        categorical_columns = ["cat" + str(_) for _ in range(50)]
        numerical_columns = ["num" + str(_) for _ in range(50)]

        y = df["target"]
        df["gain"] = 1
        cost_orig = df["gain"]
        self.gain_col = "gain"
        self.cost_orig_df = cost_orig
        self.orig_df = df

        random.seed(seed)

        weights = {f: random.choice([0.1, 1.0, 10.0, 100.0]) for f in df.columns}


        for categorical_column in categorical_columns:
            df[categorical_column].fillna("NULL", inplace=True)
            df[categorical_column] = df[categorical_column].astype('category',copy=False)

        df = one_hot_encode(
            df,
            cat_cols=categorical_columns,
            num_cols=numerical_columns,
            binary_vars=["target"],
            standardize=False,
            prefix_sep="_",
        )

        X = df.drop(columns=["target"])


        self.max_eps = 100000.0
        self.cat_map = get_cat_map(df, categorical_columns, sep='_')
        if mat_mode:
            pass
        else:
            w_vector = get_w_vec(df.drop(columns=["target"]), weights, sep="_", one_hot=True).unsqueeze(0)
            self.w = w_vector
            print(self.w)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=10000, random_state=seed
        )
        if mode == "train":
            self.X_train = X_train
            self.y_train = y_train
            self.inp = X_train.to_numpy(dtype=np.float32)
            self.oup = y_train.to_numpy(dtype=np.float32)
            self.cost_orig = X_train[self.gain_col].to_numpy(dtype=np.float32)
            if mat_mode:
                self.w_rep = get_w_rep(self.inp, costs, self.cat_map, num_list=numerical_columns, w_max=w_max)

        elif mode == "test":
            self.X_test = X_test
            self.y_test = y_test
            self.inp = X_test.to_numpy(dtype=np.float32)
            self.oup = y_test.to_numpy(dtype=np.float32)
            self.cost_orig = X_test[self.gain_col].to_numpy(dtype=np.float32)
            if mat_mode:
                self.w_rep = get_w_rep(self.inp, costs, self.cat_map, num_list=numerical_columns, w_max=w_max)

        self.mat_mode = mat_mode

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inpt = torch.Tensor(self.inp[idx])
        oupt = torch.Tensor([self.oup[idx]])
        if self.same_cost:
            cost = torch.Tensor([1.0])
        else:
            cost = torch.Tensor([self.cost_orig[idx] / 1])
        if self.mat_mode:
            w_rep = torch.Tensor(self.w_rep[idx])

        return (inpt, oupt, cost, w_rep)

class HomeCreditDataset(Dataset):
    def __init__(
        self,
        folder_path,
        mode="train",
        seed=0,
        balanced=True,
        discrete_one_hot=False,
        same_cost=False,
        drop_features=None,
        cat_map=False,
        ballet=False
    ):
        self.mode = mode
        self.same_cost = same_cost
        
        self.max_eps = 30000.0
        all_weights = {
            "NAME_CONTRACT_TYPE": 0.1,
            "FLAG_OWN_CAR": 100,
            "FLAG_OWN_REALTY": 100,
            "AMT_INCOME_TOTAL": 1,
            "NAME_TYPE_SUITE": 0.1,
            "NAME_INCOME_TYPE": 100,
            "has_children": 1000,
            "house_variables_sum_isnull": 100,
            "cluster_days_employed": 100,
            "NAME_EDUCATION_TYPE": 1000,
            "NAME_FAMILY_STATUS": 1000,
            "NAME_HOUSING_TYPE": 100,
            "REGION_RATING_CLIENT": 100,
            "REG_REGION_NOT_LIVE_REGION": 100,
            "REG_REGION_NOT_WORK_REGION": 100,
            "LIVE_REGION_NOT_WORK_REGION": 100,
            "REG_CITY_NOT_LIVE_CITY": 100,
            "REG_CITY_NOT_WORK_CITY": 100,
            "LIVE_CITY_NOT_WORK_CITY": 100,
            "FLAG_MOBIL": 10,
            "FLAG_EMP_PHONE": 10,
            "FLAG_WORK_PHONE": 10,
            "FLAG_CONT_MOBILE": 10,
            "FLAG_PHONE": 10,
            "FLAG_EMAIL": 0.1,
            "WEEKDAY_APPR_PROCESS_START": 0.1,
            "HOUR_APPR_PROCESS_START": 0.1,
            "OCCUPATION_TYPE": 100,
            "ORGANIZATION_TYPE": 100,
            "EXT_SOURCE_1":10000,
            "EXT_SOURCE_2":10000,
            "EXT_SOURCE_3":10000,
        }

        cons_weights = {
            "NAME_CONTRACT_TYPE": -1,
            "FLAG_OWN_CAR": -1,
            "FLAG_OWN_REALTY": -1,
            "AMT_INCOME_TOTAL": -1,
            "NAME_TYPE_SUITE":  -1,
            "NAME_INCOME_TYPE": -1,
            "NAME_EDUCATION_TYPE": -1,
            "NAME_FAMILY_STATUS": -1,
            "NAME_HOUSING_TYPE": -1,
            "FLAG_MOBIL": -1,
            "cluster_days_employed": -1,
            "FLAG_EMP_PHONE": -1,
            "FLAG_WORK_PHONE": -1,
            "FLAG_CONT_MOBILE": -1,
            "FLAG_PHONE": -1,
            "FLAG_EMAIL": -1,
            "WEEKDAY_APPR_PROCESS_START": -1,
            "HOUR_APPR_PROCESS_START": -1,
            "OCCUPATION_TYPE": -1,
            "ORGANIZATION_TYPE": -1,

        }

        weights = all_weights

        # For robust baseline
        #for f, w in weights.items():
        #    weights[f] = -1
        #print(weights)

        application_train_df = pd.read_csv(
            folder_path + "/home-credit-default-risk/application_train.csv"
        ).sample(frac=1, random_state=seed)
        application_test_df = pd.read_csv(
            folder_path + "/home-credit-default-risk/application_test.csv"
        )
        previous_application_df = pd.read_csv(
            folder_path + "/home-credit-default-risk/previous_application.csv"
        )

        application_train_df["CSV_SOURCE"] = "application_train.csv"
        application_test_df["CSV_SOURCE"] = "application_test.csv"
        df = pd.concat([application_train_df, application_test_df])

        # MANAGE previous_applications.csv
        temp_previous_df = previous_application_df.groupby(
            "SK_ID_CURR", as_index=False
        ).agg({"NAME_CONTRACT_STATUS": lambda x: ",".join(set(",".join(x).split(",")))})
        temp_previous_df["has_only_approved"] = np.where(
            temp_previous_df["NAME_CONTRACT_STATUS"] == "Approved", "1", "0"
        )
        temp_previous_df["has_been_rejected"] = np.where(
            temp_previous_df["NAME_CONTRACT_STATUS"].str.contains("Refused"), "1", "0"
        )

        # JOIN DATA
        df = pd.merge(df, temp_previous_df, on="SK_ID_CURR", how="left")

        # CREATE CUSTOM COLUMNS
        #################################################### total_amt_req_credit_bureau
        df["total_amt_req_credit_bureau"] = (
            df["AMT_REQ_CREDIT_BUREAU_YEAR"] * 1
            + df["AMT_REQ_CREDIT_BUREAU_QRT"] * 2
            + df["AMT_REQ_CREDIT_BUREAU_MON"] * 8
            + df["AMT_REQ_CREDIT_BUREAU_WEEK"] * 16
            + df["AMT_REQ_CREDIT_BUREAU_DAY"] * 32
            + df["AMT_REQ_CREDIT_BUREAU_HOUR"] * 64
        )
        df["total_amt_req_credit_bureau_isnull"] = np.where(
            df["total_amt_req_credit_bureau"].isnull(), "1", "0"
        )
        df["total_amt_req_credit_bureau"].fillna(0, inplace=True)

        #######################################################################  has_job
        #df["has_job"] = np.where(
        #    df["NAME_INCOME_TYPE"].isin(["Pensioner", "Student", "Unemployed"]),
        #    "1",
        #    "0",
        #)

        #######################################################################  has_children
        df["has_children"] = np.where(df["CNT_CHILDREN"] > 0, "1", "0")

        ####################################################### clusterise_days_employed
        def clusterise_days_employed(x):
            days = x["DAYS_EMPLOYED"]
            if days > 0:
                return "not available"
            else:
                days = abs(days)
                if days < 30:
                    return "less 1 month"
                elif days < 180:
                    return "less 6 months"
                elif days < 365:
                    return "less 1 year"
                elif days < 1095:
                    return "less 3 years"
                elif days < 1825:
                    return "less 5 years"
                elif days < 3600:
                    return "less 10 years"
                elif days < 7200:
                    return "less 20 years"
                elif days >= 7200:
                    return "more 20 years"
                else:
                    return "not available"

        df["cluster_days_employed"] = df.apply(clusterise_days_employed, axis=1)

        #######################################################################  custom_ext_source_3
        def clusterise_ext_source(x):
            if str(x) == "nan":
                return "not available"
            else:
                if x < 0.1:
                    return "less 0.1"
                elif x < 0.2:
                    return "less 0.2"
                elif x < 0.3:
                    return "less 0.3"
                elif x < 0.4:
                    return "less 0.4"
                elif x < 0.5:
                    return "less 0.5"
                elif x < 0.6:
                    return "less 0.6"
                elif x < 0.7:
                    return "less 0.7"
                elif x < 0.8:
                    return "less 0.8"
                elif x < 0.9:
                    return "less 0.9"
                elif x <= 1:
                    return "less 1"

        df["clusterise_ext_source_1"] = df["EXT_SOURCE_1"].apply(
            lambda x: clusterise_ext_source(x)
        )
        df["clusterise_ext_source_2"] = df["EXT_SOURCE_2"].apply(
            lambda x: clusterise_ext_source(x)
        )
        df["clusterise_ext_source_3"] = df["EXT_SOURCE_3"].apply(
            lambda x: clusterise_ext_source(x)
        )

        #######################################################################  house_variables_sum
        house_vars = [
            "APARTMENTS_AVG",
            "APARTMENTS_MEDI",
            "APARTMENTS_MODE",
            "BASEMENTAREA_AVG",
            "BASEMENTAREA_MEDI",
            "BASEMENTAREA_MODE",
            "COMMONAREA_AVG",
            "COMMONAREA_MEDI",
            "COMMONAREA_MODE",
            "ELEVATORS_AVG",
            "ELEVATORS_MEDI",
            "ELEVATORS_MODE",
            "EMERGENCYSTATE_MODE",
            "ENTRANCES_AVG",
            "ENTRANCES_MEDI",
            "ENTRANCES_MODE",
            "FLOORSMAX_AVG",
            "FLOORSMAX_MEDI",
            "FLOORSMAX_MODE",
            "FLOORSMIN_AVG",
            "FLOORSMIN_MEDI",
            "FLOORSMIN_MODE",
            "FONDKAPREMONT_MODE",
            "HOUSETYPE_MODE",
            "LANDAREA_AVG",
            "LANDAREA_MEDI",
            "LANDAREA_MODE",
            "LIVINGAPARTMENTS_AVG",
            "LIVINGAPARTMENTS_MEDI",
            "LIVINGAPARTMENTS_MODE",
            "LIVINGAREA_AVG",
            "LIVINGAREA_MEDI",
            "LIVINGAREA_MODE",
            "NONLIVINGAPARTMENTS_AVG",
            "NONLIVINGAPARTMENTS_MEDI",
            "NONLIVINGAPARTMENTS_MODE",
            "NONLIVINGAREA_AVG",
            "NONLIVINGAREA_MEDI",
            "NONLIVINGAREA_MODE",
            "TOTALAREA_MODE",
            "WALLSMATERIAL_MODE",
            "YEARS_BEGINEXPLUATATION_AVG",
            "YEARS_BEGINEXPLUATATION_MEDI",
            "YEARS_BEGINEXPLUATATION_MODE",
            "YEARS_BUILD_AVG",
            "YEARS_BUILD_MEDI",
            "YEARS_BUILD_MODE",
        ]
        df["house_variables_sum"] = df[house_vars].sum(axis=1)
        df["house_variables_sum_isnull"] = np.where(
            df["house_variables_sum"].isnull(), "1", "0"
        )
        df["house_variables_sum"].fillna(
            value=df["house_variables_sum"].median(), inplace=True
        )

        # SELECT COLUMNS
        numerical_columns = [
            "AMT_ANNUITY",
            "AMT_CREDIT",
            #"AMT_GOODS_PRICE",
            "AMT_INCOME_TOTAL",
            #"REGION_POPULATION_RELATIVE",
            #"DAYS_BIRTH",
            #"DAYS_ID_PUBLISH",
            #"DAYS_REGISTRATION",
            #CNT_CHILDREN",
            #"CNT_FAM_MEMBERS",
            #"DAYS_EMPLOYED",
            #"DAYS_LAST_PHONE_CHANGE",
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            #"total_amt_req_credit_bureau",
            #"house_variables_sum",
        ]
        categorical_columns = [
            "CODE_GENDER",
            "CSV_SOURCE",
            "NAME_EDUCATION_TYPE",
            #"CNT_CHILDREN",
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE",
            "NAME_CONTRACT_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "NAME_INCOME_TYPE",
            "NAME_TYPE_SUITE",
            "WEEKDAY_APPR_PROCESS_START",
            "HOUR_APPR_PROCESS_START",
            "REGION_RATING_CLIENT",
            #"has_only_approved",
            #"has_been_rejected",
            #"has_job",
            "cluster_days_employed",
            #"clusterise_ext_source_1",
            #"clusterise_ext_source_2",
            #"clusterise_ext_source_3",
            #"total_amt_req_credit_bureau_isnull",
            "house_variables_sum_isnull",
        ]

        binary_columns = [
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_EMAIL",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "has_children",
            "REG_REGION_NOT_LIVE_REGION",
            "REG_REGION_NOT_WORK_REGION",
            "LIVE_REGION_NOT_WORK_REGION",
            "REG_CITY_NOT_LIVE_CITY",
            "REG_CITY_NOT_WORK_CITY",
            "LIVE_CITY_NOT_WORK_CITY",
        ]

        one_hot_vars = [
            "NAME_CONTRACT_TYPE",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            #"CNT_CHILDREN",
            "NAME_TYPE_SUITE",
            "NAME_INCOME_TYPE",
            "cluster_days_employed",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
            "WEEKDAY_APPR_PROCESS_START",
            "HOUR_APPR_PROCESS_START",
            "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE"
        ]

        target_column = ["TARGET"]
        df = df[numerical_columns + categorical_columns + target_column + binary_columns]
        # MANAGE MISSING VALUES
        for numerical_column in numerical_columns:
            if df[numerical_column].isnull().values.any():
                df[numerical_column + "_isnull"] = np.where(
                    df[numerical_column].isnull(), "1", "0"
                )
            df[numerical_column].fillna(
                value=df[numerical_column].median(), inplace=True
            )

        for categorical_column in categorical_columns:
            df[categorical_column].fillna("NULL", inplace=True)
            df[categorical_column] = df[categorical_column].astype('category',copy=False)

        df["FLAG_OWN_CAR"] = np.where(
            df["FLAG_OWN_CAR"].str.contains("Y"), "1", "0"
        )
        df["FLAG_OWN_REALTY"] = np.where(
            df["FLAG_OWN_CAR"].str.contains("Y"), "1", "0"
        )
        # STANDARDISE
        # min_max_scaler = preprocessing.MinMaxScaler()
        # df[numerical_columns] = pd.DataFrame(min_max_scaler.fit_transform(df[numerical_columns]))

        # CONVERT CATEGORICAL COLUMNS INTO TYPE "category"
        categorical_columns.remove("CSV_SOURCE")

        cost_orig = df["AMT_CREDIT"]
        self.gain_col = "AMT_CREDIT"
        self.orig_df = df

        #df["EXT_SOURCE_1"] = 10 * np.log10(df["EXT_SOURCE_1"])
        df["EXT_SOURCE_1"] = np.log2(1 - df["EXT_SOURCE_1"])
        #df["EXT_SOURCE_2"] = 10 * np.log10(df["EXT_SOURCE_2"])
        #df["EXT_SOURCE_3"] = 10 * np.log10(df["EXT_SOURCE_3"])
        df["EXT_SOURCE_2"] = np.log2(1 - df["EXT_SOURCE_2"])
        df["EXT_SOURCE_3"] = np.log2(1 - df["EXT_SOURCE_3"])

        df = one_hot_encode(
            df,
            cat_cols=categorical_columns,
            num_cols=numerical_columns,
            binary_vars=["TARGET", "CSV_SOURCE"],
            standardize=False,
            prefix_sep="_",
        )
        w_vector = get_w_vec(
            df.drop(columns=["TARGET", "CSV_SOURCE"]), weights, sep="_", num_list=(numerical_columns + binary_columns)
        ).unsqueeze(0)
        self.w = w_vector
        #print(w_vector)

        # SPLIT DATA INTO TRAINING vs TRAIN
        df = df[df["CSV_SOURCE"] == "application_train.csv"]
        y = df["TARGET"]
        self.cost_orig_df = cost_orig

        # REMOVE NOT USEFUL COLUMNS
        if cat_map:
            self.cat_map = get_cat_map(df, categorical_columns, bin_var=binary_columns, sep='_')
            print(self.cat_map)
        else:
            self.cat_map = None

        X = df.drop(columns=["CSV_SOURCE", "TARGET"], axis=0)
        #print(list(X.columns))
        if balanced:
            X_resampled, self.y_resampled = RandomUnderSampler(
                random_state=seed
            ).fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, self.y_resampled, test_size=3000, random_state=seed
            )

        del X
        del y
        del df

        if mode == "train":
            self.X_train = X_train
            self.y_train = y_train
            self.inp = X_train.to_numpy(dtype=np.float32)
            self.oup = y_train.to_numpy(dtype=np.float32)
            self.cost_orig = X_train[self.gain_col].to_numpy(dtype=np.float32)

        elif mode == "test":
            self.X_test = X_test
            #print(self.X_test.iloc[0])
            self.y_test = y_test
            self.inp = X_test.to_numpy(dtype=np.float32)
            self.oup = y_test.to_numpy(dtype=np.float32)
            self.cost_orig = X_test[self.gain_col].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inpt = torch.Tensor(self.inp[idx])
        oupt = torch.Tensor([self.oup[idx]])
        if self.same_cost:
            cost = torch.Tensor([1.0])
        else:
            cost = torch.Tensor([self.cost_orig[idx] / 1])
        # (oupt.shape, oupt)
        return (inpt, oupt, cost)

# ==================================================================================================
# Emile's code

def fill_mort_acc(total_acc, mort_acc, avg):
    if np.isnan(mort_acc):
        return avg[total_acc].round()
    else:
        return mort_acc

class LendingClubDataset(Dataset):

    def __init__(self, path, mode="train", balanced=True, seed=42, cat_map=False, same_cost=False):
        self.mode = mode
        self.same_cost = same_cost

        df = pd.read_csv("../data/lending_club/lending_club_loan_two.csv")

        # -----------------------------------------------------------
        # data preprocessing

        df.emp_title.nunique()
        df.drop('emp_title', axis=1, inplace=True)
        df.emp_length.unique()
        df.drop('emp_length', axis=1, inplace=True)
        df.drop('title', axis=1, inplace=True)
        df.drop_duplicates()
        total_acc_avg = df.groupby(by='total_acc').mean().mort_acc

        df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc'], total_acc_avg), axis=1)

        # for column in data.columns:
        #     if df[column].isna().sum() != 0:
        #         missing = data[column].isna().sum()
        #         portion = (missing / data.shape[0]) * 100
        #         print(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")

        df.dropna(inplace=True)

        numerical_columns = [
            "loan_amnt",
            "annual_inc",
            "dti",
            "int_rate",
            "pub_rec_bankruptcies",
            "revol_util",
            "revol_bal"
        ]
        categorical_columns = [
            "zip_code",
        ]

        df['zip_code'] = df.address.apply(lambda x: x[-5:])

        weights = {
            "loan_amnt": 0,
            "annual_inc": 100,
            "dti": 500,
            "int_rate": 3750,
            "pub_rec_bankruptcies": 500,
            "revol_util": 500,
            "revol_bal": 500,
            "zip_code": 1
        }

        df = df[categorical_columns + numerical_columns + ["loan_status"]]

        df['loan_status'] = df.loan_status.map({'Fully Paid':1, 'Charged Off':0})
        self.cost_orig = df["loan_amnt"]
        self.gain_col = "loan_amnt"

        self.cat_map = get_cat_map(df, categorical_columns, sep='_')

        w_vector = get_w_vec(
            df.drop(columns=["loan_status"]), weights, sep="_", num_list=(numerical_columns)
        ).unsqueeze(0)
        self.w = w_vector

        y = df["loan_status"]
        X = df.drop(columns=["loan_status"], axis=0)

        # balance data
        rus = RandomUnderSampler(random_state=42)
        X, y = rus.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=3000, random_state=seed
        )

        del X
        del y
        del df

        # 
        if mode == "train":
            self.X_train = X_train
            self.y_train = y_train
            self.inp = X_train.to_numpy(dtype=np.float32)
            self.oup = y_train.to_numpy(dtype=np.float32)
            self.cost_orig = X_train[self.gain_col].to_numpy(dtype=np.float32)

        elif mode == "test":
            self.X_test = X_test
            self.y_test = y_test
            self.inp = X_test.to_numpy(dtype=np.float32)
            self.oup = y_test.to_numpy(dtype=np.float32)
            self.cost_orig = X_test[self.gain_col].to_numpy(dtype=np.float32)

        else:
            raise ValueError

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inpt = torch.Tensor(self.inp[idx])
        oupt = torch.Tensor([self.oup[idx]])
        if self.same_cost:
            cost = torch.Tensor([1.0])
        else:
            cost = torch.Tensor([self.cost_orig[idx]])
        return (inpt, oupt, cost, 0)
    
    


shape_dict = {"ieeecis": 147, "twitter_bot": 19, "home_credit": 183, "syn": 100, "credit_app": 70} # TODO: check if I need it at all 