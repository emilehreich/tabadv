from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import string
import random

from sklearn.preprocessing import OneHotEncoder
def generate_dataset(
    num_features = 40,
    cat_features = 40,
    n_cat = [3, 4, 8, 16],
    n_samples = 100000,
    noise_level = 0.1,
    costs = [0.1, 1.0, 10.0, 100.0]
):
    df = pd.DataFrame()
    labels = np.random.binomial(1, 0.5, n_samples)
    #print(labels.shape)
    num_f = np.repeat(labels[..., np.newaxis], num_features, axis=1)
    #print(num_f.shape)
    noise_vec = np.random.standard_normal(size=num_f.shape) * noise_level * 10
    num_f = num_f + noise_vec
    #print(num_f[0])
    df['target'] = labels
    for i in range(num_features):
        df["num"+str(i)] = num_f[:,i]

    for i in range(cat_features):
        feature_dim = np.random.choice(n_cat)
        feature_values = list(string.ascii_lowercase[0:feature_dim])
        #print(feature_values)
        p_0 = np.zeros((feature_dim))
        p_0[0] += 1 - noise_level * ((feature_dim - 1) / feature_dim) + noise_level/feature_dim
        p_0 -= noise_level/feature_dim
        p_0 = np.abs(p_0)
        p_1 = np.zeros((feature_dim))
        p_1[1] += 1 - noise_level * ((feature_dim - 1) / feature_dim) + noise_level/feature_dim
        p_1 -= noise_level/feature_dim
        p_1 = np.abs(p_1)
        print(p_0, p_1)
        feature_0 = np.random.choice(feature_values, size=n_samples, p=p_0)
        #print(feature_0)
        feature_1 = np.random.choice(feature_values, size=n_samples, p=p_1)
        print("cat" + str(i), feature_0, feature_1)
        features = [feature_0, feature_1]
        feature = [features[l][i] for i, l in enumerate(labels)]
        df["cat" + str(i)] = feature
        #print(labels[0:10])
        #print(feature[0:10])

    print(df.head())

    return df.copy()



def main():
    df = generate_dataset(
        num_features = 50,
        cat_features = 50,
        n_cat = [3, 4, 8, 16],
        n_samples = 100000,
        noise_level = 0.85
    )

    dataset_name = "syn_85"
    costs = {}

    for col in df.columns:
        if col[0:3] == "num":
            costs[col] = random.choice([0.1, 1.0, 10.0, 100.0])
        elif col[0:3] == "cat":
            num_var = df[col].nunique()
            cost_matrix = np.zeros((num_var, num_var))
            vecs = []
            dim = 2
            for i in range(num_var):
                norm = random.choice([0.1, 1.0, 10.0, 100.0, 100000.0])
                point = np.random.uniform(0, 1, 2)
                vec = point * norm
                vecs.append(vec)
            #print(vecs)
            for i in range(num_var):
                 for j in range(num_var):
                    cost_matrix[i, j] = np.linalg.norm(vecs[i] - vecs[j])

            #     for j in range(i + 1, num_var):
            #             cost = 0
            #             if i == 0:
            #                 cost_matrix[i, j] = random.choice([0.1, 1.0, 10.0, 100.0, 1000000.0])
            #             else:
            #                 mcost = 0.1
            #                 for k in range(i):
            #                     if cost_matrix[k, j] - cost_matrix[k, i] > mcost:
            #                         mcost = cost_matrix[k, j] - cost_matrix[k, i] # to have a distance matrix

            #                 while cost < mcost:
            #                     cost = random.choice([0.1, 1.0, 10.0, 100.0, 1000000.0])
            #                 cost_matrix[i, j] = cost
            
            # for i in reversed(range(num_var)):
            #     for j in reversed(range(i)):
            #         cost = 0
            #         mcost = 0.1
            #         for k in range(i):
            #             print(i, j, k, cost_matrix[k, j], cost_matrix[k, i])
            #             if cost_matrix[k, j] > mcost + cost_matrix[k, i]: # no shortcut
            #                 mcost = cost_matrix[k, j] - cost_matrix[k, i] # to have a distance matrix
            #                 #print(mcost)
            #         for k in range(i + 1, num_var):
            #             print(i, j, k, cost_matrix[k, j], cost_matrix[k, i])
            #             if cost_matrix[k, j] > mcost + cost_matrix[k, i]: # no shortcut
            #                 mcost = cost_matrix[k, j] - cost_matrix[k, i] # to have a distance matrix
            #                 #print(mcost)
                        
            #         while cost < mcost:
            #             cost = random.choice([0.1, 1.0, 10.0, 100.0, 1000000.0])
            #         cost_matrix[i, j] = cost
            #         print((cost, mcost))

            costs[col] = cost_matrix
            #print(cost_matrix)
        else:
            pass
    



    y = df['target']
    print(df.head())
    X = df.drop(['target'],axis=1)
    print(X.shape)
    X = OneHotEncoder(max_categories=16).fit_transform(X)

    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)
    #ft_ind = ['feature' + str(_) for _ in range(n_ft)]
    clf = RandomForestClassifier(max_depth=4, random_state=13)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    print(acc)

    #df = pd.DataFrame(data=X, columns=ft_ind)
    #df['target'] = y
    #print(df)
    np.save("../data/" + dataset_name + "_costs", costs)
    df.to_csv("../data/" + dataset_name + ".csv",index=False)


if __name__ == "__main__":
    main()
