import sys
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import notebook as tqdm

sys.path.append("..")

from src.utils.data import one_hot_encode
from src.utils.data import diff
from src.utils.hash import fast_hash

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import wandb


def clf_pgd_attack(clf, x, y, eps, steps, dataset, goal_fn, attack_type="pgd", **kwargs):
    x_ten = torch.Tensor(np.array([x], dtype=np.float32)).to(clf.device)
    y_ten = torch.abs(
        1
        - torch.Tensor(np.array([y], dtype=np.float32)[..., np.newaxis]).to(clf.device)
    )  # we need source class here
    gains = torch.Tensor([1.0]).to(clf.device)

    pgd_alpha = 2 * (1 / steps)
    w_vec = dataset.w.to(clf.device)
    out = clf.predict([x])
    if out == y:
        return x, 0.0
    # print(x_ten, x_ten.shape)
    if attack_type == "pgd":
        #eps *= 2 # Do not need it anymore
        delta = attack_pgd_training(
            clf.model,
            x_ten,
            y_ten,
            eps,
            pgd_alpha,
            steps,
            dist="l1",
            gains=gains,
            w_vec=w_vec,
            utility_type="constant",
            eps_part=1.0,
            eps_max=0.0,
        )
    elif attack_type == "Ballet":
        delta = attack_pgd_training(
            clf.model,
            x_ten,
            y_ten,
            eps,
            0.1,
            steps,
            dist="Ballet",
            gains=gains,
            w_vec=w_vec,
            utility_type="constant",
            eps_part=1.0,
            eps_max=0.0,
        )
    adv_x = x_ten + delta
    if dataset.cat_map is not None:
        for cat in dataset.cat_map:
            i, j = dataset.cat_map[cat]
            max_ind = torch.argmax(adv_x[:, i : j + 1], 1)
            adv_x[:, i : j + 1] = 0
            adv_x[:, i : j + 1][range(x_ten.shape[0]), max_ind] = 1
    # print(delta)
    # print(x_ten - adv_x)
    cost = torch.squeeze(torch.matmul(torch.abs(x_ten - adv_x), w_vec.T))

    adv_x = pd.Series(adv_x[0].detach().cpu().numpy(), index=x.index)

    if not goal_fn(adv_x):
        return None, None

    return adv_x, cost.detach().cpu().numpy() / 2

class EmbWrapper:
    def __init__(self, model, embs, cats, device):
        self.model = model
        self.embs = embs
        self.device = device
        self.cats = cats

    def emb_transform(self, X):
        X_ten = torch.Tensor(np.array(X, dtype=np.float32)).to(self.device)
        cats = self.cats
        outs = []
        for k, emb in enumerate(self.embs):
            i, j = cats[k]
            outs.append(emb(X_ten[:, i:j+1]))
        X_emb = torch.cat(outs, dim=1)
        return X_emb.detach().cpu().numpy()


    def fit(self, X_train, y_train):
        X_train_emb = self.emb_transform(X_train)
        self.model.fit(X_train_emb, y_train)

    def predict(self, X):
        X_emb = self.emb_transform(X)
        return self.model.predict(X_emb)

    def predict_proba(self, X):
        X_emb = self.emb_transform(X)
        return self.model.predict_proba(X_emb)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

class TorchWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def predict(self, X):
        X_ten = torch.Tensor(np.array(X, dtype=np.float32)).to(self.device)
        out = self.model(X_ten)
        y = torch.round(torch.sigmoid(out))
        # print(y.detach().cpu().numpy().astype(np.int32)[:,0])
        return y.detach().cpu().numpy().astype(np.int32)[:, 0]

    def predict_proba(self, X):
        X_ten = torch.Tensor(np.array(X, dtype=np.float32)).to(self.device)
        out = self.model(X_ten)
        sigmoid_output = torch.sigmoid(out).detach().cpu().numpy()
        return np.array(list(zip(1 - sigmoid_output, sigmoid_output)))[:, :, 0]

    def score(self, X, y):
        return (self.predict(X) == y).mean()


def load_torch_model(path, dataset, device, model_label=None):
    from .loaders import shape_dict
    from .train import model_dict

    model_label = model_label or dataset
    net = model_dict[model_label](inp_dim=shape_dict[dataset]).to(device)
    net.load_state_dict(torch.load(path))
    net.eval()
    return TorchWrapper(net, device)


def project_onto_l1_ball(x, eps):
    """
    Compute Euclidean projection onto the L1 ball for a batch.

      min ||x - u||_2 s.t. ||u||_1 <= eps

    Inspired by the corresponding numpy version by Adrien Gaidon.

    Parameters
    ----------
    x: (batch_size, *) torch array
      batch of arbitrary-size tensors to project, possibly on GPU

    eps: float
      radius of l-1 ball to project onto

    Returns
    -------
    u: (batch_size, *) torch array
      batch of projected tensors, reshaped to match the original

    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.

    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)

    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)


def project_onto_l1_ball_w(x, epses):

    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < epses).float().unsqueeze(1)
    mu, ind = torch.sort(torch.abs(x), dim=1, descending=True)

    cumsum = torch.cumsum(mu, dim=1)

    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - epses.unsqueeze(1))) * arange, dim=1)

    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - epses) / rho

    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)

    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)

def check_sorted(x, ind):
    n = x.shape[1]
    checks = torch.ge(x[:,0:n-1], x[:, 1:n])
    return torch.all(checks)

def project_w_accel(x, epses, w, sigma=0.0, prev_ind=None):
    y = torch.zeros_like(x)
    x = x.view(x.shape[0], -1)

    s = torch.sign(x)
    x = torch.abs(x)

    if w.shape != x.shape:
        w = w.repeat(x.shape[0], 1)

    mask = ((x * w).sum(dim=1) < epses).float().unsqueeze(1)

    z_u = torch.div(x, w + sigma)

    z, ind = torch.sort(z_u, dim=1, descending=True)
    lambdas = torch.zeros(x.shape[0]).cuda()

    x_p = x.gather(dim=1, index=ind)
    w_p = w.gather(dim=1, index=ind)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    w_cumsums = torch.cumsum(w_p ** 2, dim=1)
    cumsum = torch.cumsum(x_p * w_p, dim=1)
    rho, _ = torch.max((z > ((cumsum - epses.unsqueeze(1)) / w_cumsums) ) * arange, dim=1)

    lambdas = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - epses) / (w_cumsums[torch.arange(x.shape[0]), rho.cpu() - 1])
    lambdas = lambdas.unsqueeze(1).repeat(1, x.shape[1]).cuda()
    y = mask * x * s + (1 - mask) * s * torch.nn.functional.relu(x  - w * lambdas)
    return y, ind

def project_w_onto_l1_ball_w(x, epses, w, sigma=0.0):
    y = torch.zeros_like(x)
    x = x.view(x.shape[0], -1)

    s = torch.sign(x)
    x = torch.abs(x)
    if w.shape != x.shape:
        w = w.repeat(x.shape[0], 1)

    z_u = torch.div(x, w + sigma)

    z, ind = torch.sort(z_u, dim=1, descending=False)
    max_j = torch.zeros(x.shape[0],).cuda().type(torch.LongTensor)
    lambdas = torch.zeros(x.shape[0]).cuda()

    x_p = x.gather(dim=1, index=ind)
    w_p = w.gather(dim=1, index=ind)
    for i in range(x.shape[1]):
        f = (-epses + torch.sum(x_p[:, i + 1 :] * w_p[:, i + 1 :], dim=1)) / torch.sum(
            w_p[:, i + 1 :] ** 2, dim=1
        )
        lambdas = torch.where(f > z[:, i], f, lambdas)

    lambdas = lambdas.unsqueeze(1).repeat(1, x.shape[1]).cuda()
    y = s * torch.nn.functional.relu(x - w * lambdas)
    return y


def project_w_onto_l1_ball_w_rec(x, epses, w):
    y = torch.zeros_like(x)
    original_shape = x.shape
    # print(x.shape)
    x = x.view(x.shape[0], -1)

    s = torch.sign(x)
    x = torch.abs(x)
    w = w.repeat(x.shape[0], 1)
    # print(x.shape, w.shape)

    z_u = torch.div(x, w)

    z, ind = torch.sort(z_u, dim=1, descending=False)
    max_j = torch.zeros(x.shape[0],).cuda().type(torch.LongTensor)
    lambdas = torch.zeros(x.shape[0]).cuda()

    # print(z_u[0], x[0], w[0])
    x_p = x.gather(dim=1, index=ind)
    w_p = w.gather(dim=1, index=ind)
    # print(z[0], x_p[0], w_p[0])
    # print(z_u.shape, x.shape, w.shape, ind.shape, x_p.shape, w_p.shape)
    for i in range(x.shape[1]):
        # print((x_p[:,0:i+1] * w_p[:,0:i+1]).shape, x_p[:,0:i+1].shape, w_p[:,0:i+1].shape)
        f = (-epses + torch.sum(x_p[:, i + 1 :] * w_p[:, i + 1 :], dim=1)) / torch.sum(
            w_p[:, i + 1 :] ** 2, dim=1
        )
        # for k in range(x.shape[0]):
        # max_j[k] = i if f[k] > z[k, i] else max_j[k]
        # lambdas[k] = f[k] if f[k] > z[k, i] else lambdas[k]
        # max_j = torch.where(f > z[:,i], i, max_j)
        lambdas = torch.where(f > z[:, i], f, lambdas)
    # for k in range(x.shape[0]):
    # for i in range(x.shape[1]):
    #    y[k, i] = s[k,i] * torch.nn.functional.relu(x[k,i] - w[k,i] * lambdas[k])
    lambdas = lambdas.unsqueeze(1).repeat(1, x.shape[1]).cuda()
    y = s * torch.nn.functional.relu(x - w * lambdas)
    return y

def project_simplex(x):
    N = x.shape[1]
    x_mins, _ = torch.min(x, 1)
    x_mins = x_mins.unsqueeze(1).expand(-1, N)
    x += 1/N
    x -= x_mins
    mask = (torch.norm(x, p=1, dim=1) < 1).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - 1)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - 1) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    y = mask * x + (1 - mask) * proj * torch.sign(x)
    return y

def project_cat(X, x, cat_map):
    y = x
    for cat in cat_map:
        i, j = cat_map[cat]
        if i == j:
            y[:, i] = torch.clamp(x[:, i], X[:,i] * -1, 1 - X[:,i])
        else:
            y[:, i : j + 1] = project_simplex(x[:, i : j + 1])
    return y

def check_cat(X, delta, cat_map):
    sums = []
    mind_b = []
    mind_a = []
    mxs = []
    with torch.no_grad():
        for k, cat in enumerate(cat_map):
            i, j = cat_map[cat]
            if i == j:
                pass
            else:
                #print(k, (X + delta)[0,i: j + 1])
                sums.append(torch.sum((X + delta)[0,i: j + 1]).cpu().numpy())
                mind_b.append(torch.argmax(X[0,i: j + 1]).cpu().numpy())
                mind_a.append(torch.argmax((X + delta)[0,i: j + 1]).cpu().numpy())
                mxs.append(torch.max((X + delta)[0,i: j + 1]).cpu().numpy())
    wandb.log({"Avg cat": sum(sums)})
    return sums

def project_intersection(X, delta, epses, w, cat_map, iter_lim=1):
    for cat in cat_map:
        i, j = cat_map[cat]
        if i == j:
            pass
        else:
            delta[:,i: j + 1] += X[:,i: j + 1]

    x = delta
    y = x
    p = torch.zeros_like(delta)
    q = torch.zeros_like(delta)
    for i in range(iter_lim):
        y = project_cat(X, x + p, cat_map)
        p = x + p - y
        x, ind = project_w_accel(y + q, epses, w, sigma=0.0001, prev_ind=None)
        q = y + q - x

    for cat in cat_map:
        i, j = cat_map[cat]
        if i == j:
            pass
        else:
            x[:,i: j + 1] -= X[:,i: j + 1]
    return x


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.train()


def util_loss(output, delta, y, costs, w_vec):
    vec_gain = torch.squeeze(
        torch.clip(torch.abs(torch.sigmoid(output) - y) * 2, 0, 1) * costs
    )
    vec_cost = torch.squeeze(torch.matmul(torch.abs(delta), w_vec.T))
    vec_util = torch.nn.functional.relu(vec_gain - vec_cost) / delta.shape[0]
    return torch.sum(vec_util)


def util_loss_non_rel(output, delta, y, costs, w_vec):
    vec_gain = torch.squeeze(
        torch.clip(torch.abs(torch.sigmoid(output) - y) * 2, 0, 2) * costs
    )
    vec_cost = torch.squeeze(torch.matmul(torch.abs(delta), w_vec.T))
    # print(torch.sigmoid(output[0]), y[0], vec_gain[0], vec_cost[0])
    vec_util = (vec_gain - vec_cost) / delta.shape[0]
    return vec_util


def max_util_delta(
    model,
    X,
    y,
    alpha,
    attack_iters,
    dist="linf",
    costs=None,
    gamma=1,
    w_vec=None,
    cat_map=None,
    eps_part=1.0,
):
    delta = torch.zeros_like(X).cuda()
    delta.requires_grad = True
    w_rep = w_vec.repeat(X.shape[0], 1)
    costs_rep = costs.repeat(1, X.shape[1])
    prev_util = torch.zeros_like(X).cuda()
    steps = torch.ones_like(X).cuda() * 1 * eps_part
    prev_delta = delta
    for i in range(attack_iters):

        delta.requires_grad = True
        output = model(X + delta)
        vec_util = util_loss_non_rel(output, delta, y, costs, w_vec)

        loss = torch.sum(vec_util)
        loss.backward()
        vec_util = vec_util[:, None].repeat(1, X.shape[1])
        grad = delta.grad.detach()
        # with torch.no_grad():
        # for i in range(X.shape[0]):
        # if vec_util[i] < prev_util[i]:
        #    steps = torch.where((vec_util - prev_util) > 0, steps / 2, steps)
        #    delta = torch.where(vec_util < prev_util, prev_delta, delta)
        #    grad = torch.where(vec_util < prev_util, grad * 0, grad)

        prev_delta.data = delta.data.detach()
        grad[torch.where(w_rep > 90000)] = 0
        grad[torch.where(costs_rep < 0.001)] = 0
        grad_norm = torch.abs(grad).sum([1], keepdim=True)
        # print(grad_norm[0], steps[0,0])
        delta.data = delta + (grad / (grad_norm + 0.001)) * steps
        prev_util = vec_util.detach()
        # delta.grad.zero_()
        if cat_map is not None:
            for cat in cat_map:
                with torch.no_grad():
                    i, j = cat_map[cat]
                    # print(delta[0, i:j + 1], X[0, i:j + 1], torch.clip((X + delta)[0, i:j + 1].data, 0, 1) - X[0, i:j + 1].data)
                    delta[:, i : j + 1] = (
                        torch.clip((X + delta)[:, i : j + 1], 0, 1) - X[:, i : j + 1]
                    )
                    # print(delta[0, i:j + 1])
    delta.requires_grad = False
    return delta.detach()


def attack_pgd_training(
    model,
    X,
    y,
    eps,
    alpha,
    attack_iters,
    dist="l1",
    gains=None,
    gamma=1,
    w_rep=None,
    utility_type="constant",
    eps_part=1.0,
    utility_max=False,
    eps_max=0,
    cat_map=None,
    iter_lim=10
):
    noise = torch.zeros_like(X).cuda()
    delta = torch.randn(X.shape).cuda() * 0.05 # to avoid catastrophic overfitting
    sigma = 0.000001
    grad_norm = torch.zeros_like(X).cuda()
    delta.requires_grad = True
    criterion = torch.nn.BCEWithLogitsLoss()

    if dist == "l-mix":
        #print(w_rep[0])
        mp_copy = cat_map.copy()
        for cat in mp_copy:
            i, j = cat_map[cat]
            if i == j:
                pass
            else:
                mask = (X[:, i: j + 1] - 1) * -1
                w_rep[:, i: j + 1] = w_rep[:, i: j + 1] * mask          
            #if w_rep[0, i] > eps or w_rep[0, j] > eps:
            #    w_rep[:, i: j + 1] = 100000
            #    cat_map.pop(cat)

    for _ in range(attack_iters):
        output = model(X + delta)

        if utility_max:
            vec_util = util_loss_non_rel(output, delta, y, gains, w_vec)
            loss = torch.sum(vec_util)
        else:
            if dist == "Ballet":
                loss = criterion(output, y) + eps * torch.sum(torch.norm(delta * w_rep, dim=1))
            else:
                loss = criterion(output, y)

        loss.backward()
        grad = delta.grad.detach()
        idx_update = torch.ones(y.shape, dtype=torch.bool)

        if dist == "l-mix":
            pass
        else:
            grad[torch.where(w_rep > 90000)] = 0
        
        if dist == "Ballet":
            grad[torch.where(w_rep > 90000)] = 0
            delta.data = delta + alpha * grad
            delta.grad.zero_()

        elif dist == "l2":
            grad_norm = (grad ** 2).sum([1], keepdim=True) ** 0.5
            delta.data = delta + alpha * grad / (grad_norm + sigma)
            delta_norms = (delta.data ** 2).sum([1], keepdim=True) ** 0.5
            delta.data = (
                eps
                * delta.data
                / torch.max(eps * torch.ones_like(delta_norms), delta_norms)
            )
            delta.grad.zero_()
        elif dist == "l-mix":
            grad_norm = torch.abs(grad).sum([1], keepdim=True)
            if utility_type == "constant":
                epses = eps * gamma * torch.ones_like(gains)
            elif utility_type == "additive":
                epses = torch.nn.functional.relu(gains - eps) * gamma
            elif utility_type == "multiplicative":
                epses = (gains * eps) * gamma
            if eps_max != 0.0:
                epses = torch.clip(epses, 0, eps_max)
            alphas = alpha * epses * gamma
            epses = epses.squeeze()
            epses = epses * eps_part
            delta.data = delta + alphas *  grad / (grad_norm + sigma)
            with torch.no_grad():
                delta.data = project_intersection(
                    X.data, delta, epses, w_rep, cat_map, iter_lim=iter_lim
                )
            #print("Torch", torch.tensordot(torch.abs(delta), w_rep, dims=([1], [1])).shape)
            delta.grad.zero_()
        elif dist == "l1":
            grad_norm = torch.abs(grad).sum([1], keepdim=True)
 
            # print(X[0], grad[0], grad_norm[0], torch.max(grad[0]))
            if gains == None:
                delta.data = project_onto_l1_ball(
                    delta + alpha * grad / (grad_norm + sigma), eps
                )
            else:
                if utility_type == "constant":
                    epses = eps * gamma * torch.ones_like(gains)
                elif utility_type == "additive":
                    epses = torch.nn.functional.relu(gains - eps) * gamma
                elif utility_type == "multiplicative":
                    epses = (gains * eps) * gamma

                if eps_max != 0.0:
                    epses = torch.clip(epses, 0, eps_max)
                alphas = alpha * epses * gamma
                epses = epses.squeeze()
                epses = epses * eps_part
                # print(epses[0])
                # print(torch.max(gains), torch.max(epses))
                if w_rep is None:
                    delta.data = project_onto_l1_ball_w(
                        delta + alphas * grad / (grad_norm + sigma), epses
                    )
                else:
                    upd_delta = delta + alphas * grad / (grad_norm + sigma)
                    if cat_map is not None and not utility_max:
                        for cat in cat_map:
                            i, j = cat_map[cat]
                            if i == j:
                                mask =  torch.ones_like(X[:, i]) - X[:, i] * 2
                                upd_delta[:, i] = mask * torch.nn.functional.relu(upd_delta[:, i] * mask)
                                #if (i == 6):
                                #    print(upd_delta[0, i], X[0, i], grad[0, i])
                            else:
                                max_ind = torch.argmax(X[:, i : j + 1], 1)
                                #print(max_ind)
                                tmp = upd_delta[:, i : j + 1][range(X.shape[0]), max_ind]
                                upd_delta[:, i : j + 1] = torch.nn.functional.relu(upd_delta[:, i : j + 1])
                                upd_delta[:, i : j + 1][range(X.shape[0]), max_ind] = -torch.nn.functional.relu(-tmp)
                                #print(upd_delta[:, i : j + 1][range(X.shape[0]), max_ind][0])

                    delta.data = project_w_onto_l1_ball_w(
                        upd_delta, epses, w_rep
                    )
                    # print(epses[0], torch.dot(torch.abs(delta.data[0]), w_vec[0]), grad_norm[0], alphas[0])
                    #print(delta.data[0,6:8], delta.data[0,34])

            delta.grad.zero_()
        else:
            delta.data[idx_update] = (delta + alpha * grad_sign)[idx_update]
            delta.data = torch.clip(delta.data, -eps, eps)
            delta.grad.zero_()
    delta.requires_grad = False
    # delta_norm = torch.abs(delta).sum([1], keepdim=True)
    if dist == "l-mix":
        wandb.log({
        "avg_delta_weight" : torch.tensordot(torch.abs(delta), w_rep, dims=([1], [1])).diagonal().mean().cpu().numpy(),
        "avg_grad_norm": grad_norm.mean().cpu().numpy()
        })
    return delta.detach() + noise.detach()


def dataset_eval(model, loader, score="f1", device=torch.device("cpu")):
    model.eval()
    epoch_loss = 0
    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for bidx, (x, y, c, _) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        predictions = model(x)
        for idx, i in enumerate(predictions):
            i = torch.round(torch.sigmoid(i))
            if i == y[idx]:
                correct += 1
                if y[idx] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if y[idx] == 1:
                    fp += 1
                else:
                    fn += 1

        if score == "acc":
            acc = correct / len(loader.dataset)

        if score == "f1":
            acc = tp / (tp + (fp + fn) / 2)

    model.train()

    return acc, 0


def dataset_eval_rob(
    model,
    loader,
    score="f1",
    eps=0.0,
    pgd_alpha=0,
    pgd_steps=0,
    w_rep=None,
    device=torch.device("cpu"),
    cat_map=None,
    utility_type="constant",
    utility_max=False,
    eps_max=0.0,
    dist="l1",
    iter_lim=10
):
    model.eval()
    epoch_loss = 0
    correct = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for bidx, (x, y, c, w_rep) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)
        w_rep = w_rep.to(device)

        #w_rep = w_vec.repeat(x.shape[0], 1)
        x[torch.where(w_rep == -1)] = 0

        if not utility_max:
            delta = attack_pgd_training(
                model,
                x,
                y,
                eps,
                pgd_alpha,
                pgd_steps,
                dist=dist,
                gains=c,
                w_rep=w_rep,
                utility_type=utility_type,
                eps_max=eps_max,
                cat_map=cat_map,
                iter_lim=iter_lim
            )
            #print(eps, delta[0])
        else:
            # delta = max_util_delta(
            #    model, x, y, pgd_alpha, pgd_steps, dist="l1", costs=c, w_vec=w_vec, cat_map=cat_map, eps_part=10.0
            # )
            delta = attack_pgd_training(
                model,
                x,
                y,
                10.0,
                pgd_alpha / 5,
                pgd_steps * 5,
                dist=dist,
                gains=c,
                w_rep=w_rep,
                utility_type="constant",
                cat_map=cat_map
            )
        x_adv = x + delta
        #if cat_map is not None and not utility_max:
        #    for cat in cat_map:
        #        i, j = cat_map[cat]
        #        max_ind = torch.argmax(x_adv[:, i : j + 1], 1)
        #        x_adv[:, i : j + 1] = 0
        #        x_adv[:, i : j + 1][range(x.shape[0]), max_ind] = 1

        
        predictions = model(x_adv)
        for idx, i in enumerate(predictions):
            i = torch.round(torch.sigmoid(i))
            if i == y[idx]:
                correct += 1
                if y[idx] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if y[idx] == 1:
                    fp += 1
                else:
                    fn += 1
                #print(delta[idx], x[idx], y[idx], torch.round(torch.sigmoid(predictions[idx])))
        if score == "acc":
            acc = correct / len(loader.dataset)

        if score == "f1":
            acc = tp / (tp + (fp + fn) / 2)

    model.train()

    return acc, 0


def recompute_cost(specs, x, adv_x):
    cost = 0
    for feature_spec in specs:
        cost += feature_spec.get_cost(x, adv_x)
    return cost


def diff(x, x_prime):
    assert x_prime is not None
    diff_index = x != x_prime
    return pd.concat(dict(orig=x, new=x_prime), axis=1).loc[diff_index]


def get_adv_utility(target_class, clf, x, adv_x, cost, gain):
    adv_x = adv_x if adv_x is not None else x
    cost = cost if cost is not None else 0
    pred_class = clf.predict(np.array([adv_x]))[0]
    return (target_class == pred_class) * max(gain - cost, 0)
