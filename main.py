#!/usr/bin/env python3

import warnings
from collections import OrderedDict
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.optimize import root_scalar
from scipy.special import xlogy
from scipy.special import expit

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

pd.set_option('display.width', 1000)
np.set_printoptions(edgeitems=10, linewidth=100000,
    precision=3, suppress=True)


TILE = 20    # Set default no dataset passes when MC estimating quantities.




# Bound solving utils.

def phi_s(s, u1, u2):
    "Calculate reparameterised version of phi."
    u3 = 1 - u1 - u2
    # Instead of using t = np.exp(s), we use following overflow-resistant variants.
    # Below is stable form of: u1 + u2/(1+2t) + u3/(1+t)
    d = u1 + u2 * expit(-s * np.log(2)) + u3 * expit(-s)
    # Below is stable form of: log(d) + u2 log(1+2t) + u3 log(1+t)
    return np.log(d) + u2 * np.logaddexp(0, s * np.log(2)) + u3 * np.logaddexp(0, s)

def phi_mu(mu, u1, u2):
    "Calculate phi(mu)."
    # Not actually used by solver.
    s = -np.log(-(mu + 1))
    return phi_s(s, u1, u2)

def solve_mu_star(u1, u2, c):
    # Should probably error check for too large a c giving overflow.
    s_star = root_scalar(lambda s: phi_s(s, u1, u2) - c, bracket=[-10, 800], method="bisect")
    return -np.exp(-s_star.root) - 1

def inv_kl(u1, u2, c):
    u3 = 1 - u1 - u2
    try:
        ms = solve_mu_star(u1, u2, c)
    except ValueError as e:
        warnings.warn(f"solve_mu_star error with inputs u1 = {u1:.4f}, u2 = {u2:.4f}, c = {c:.4f}. bracket = [{phi_s(-10, u1, u2) - c:.4f}, {phi_s(1000, u1, u2) - c:.4f}]  Default to inv_kl = 1.")
        return 1.
    ls_inv = u1/(ms + 1) + u2/(ms - 1) + u3/ms
    fs_den = u1/(ms + 1) - u2/(ms - 1)
    fs = fs_den / ls_inv
    return fs

def bernoulli_small_kl(q, p):
    return xlogy(q, q/p) + xlogy(1-q, (1-q)/(1-p))

def invert_small_kl(train_loss, rhs):
    "Get the inverted small-kl, largest p such that kl(train : p) \le rhs"
    start = np.minimum(0.999, train_loss + np.sqrt(rhs / 2.))    # start at McAllester if in range.
    try:
        res = root_scalar(lambda r: bernoulli_small_kl(train_loss, r) - rhs,
                        x0=start,
                        bracket=[train_loss, 1. - 1e-10])
    except ValueError:
        return 1.
    return res.root




# Datasets and training.

def read_dataset(name, split_seed=0, corrupt_frac=0):
    "Read preprocessed dataset exactly as in Mhammedi et al. (2019)."
    X = pd.read_csv(f"./preproc/{name}-xs.csv", sep=" ").to_numpy()
    y = pd.read_csv(f"./preproc/{name}-ys.csv", sep=" ").to_numpy().squeeze()
    flip = np.random.choice(len(y), int(corrupt_frac * len(y)), replace=False)
    y[flip] = 1 - y[flip]
    return train_test_split(X, y, test_size=0.2, random_state=split_seed)

def fit_lgr(Xs, ys, lambda_=0.01):
    "Fit a logistic regression."
    return LogisticRegression(C=1. / len(ys) / lambda_, max_iter=1000).fit(Xs, ys)

def predict_perturbed(lgr, xs, sigma):
    "Predict xs with lgr perturbed by sigma-stddev noise."
    linear = np.squeeze(np.dot(xs, lgr.coef_.T) + lgr.intercept_, axis=1)     # Make same as lgr.predict(xs) format.
    # (w + sg) x + b = wx + b + s * g * sqrt(1+|x|^2)
    norm2 = 1. + np.linalg.norm(xs, axis=1) ** 2
    perturbed_linear = np.random.normal(loc=linear, scale=sigma * np.sqrt(norm2))
    return (expit(perturbed_linear) > 0.5).astype(int)

def get_kl(lgr, sigma_prior, sigma, lgr_prior=None):
    "Get Gaussian KL divergence of lgr."
    w = np.concatenate([lgr.coef_[0], lgr.intercept_])
    if lgr_prior is not None:
        w0 = np.concatenate([lgr_prior.coef_[0], lgr_prior.intercept_])
        w = w - w0
    dim = len(w)
    ratio2 = (sigma_prior / sigma) ** 2
    kl = 0.5 * (np.linalg.norm(w) / sigma_prior) ** 2
    kl += 0.5 * dim * (np.log(ratio2) + 1/ratio2 - 1)
    return kl

def stochastic_train_err(lgr, xs, ys, sigma, tile=20):
    "Mean train error of lgr perturbed by sigma."
    xs_tiled, ys_tiled = np.tile(xs, [tile, 1]), np.tile(ys, tile)
    return np.mean((predict_perturbed(lgr, xs_tiled, sigma) != ys_tiled).astype(int))

def is_err(preds, ys):
    return (preds != ys).astype(int)


# Simple Test.
# Check that non-perturbed version of stochastic-train-err is working.
xs_, _, ys_, _ = read_dataset("haberman")
lgr_full = fit_lgr(xs_, ys_)
assert np.array_equal(lgr_full.predict(xs_), predict_perturbed(lgr_full, xs_, 0.0))



# Bounds.

def opt_bound(bound, m, delta=0.05):
    "Optimise bound wrt posterior sigma."
    # Note we use the full dataset m in opt_bound, even if using X_bnd.
    sigmas = [np.sqrt(0.5) ** j for j in range(1, int(np.ceil(np.log2(m))))]
    grid_delta = delta / np.ceil(np.log2(m))     # TODO: not sure this is needed? Bound valid for any \sigma in posterior?? Only for data-dependent bound.
    bnds = [bound(s, delta=grid_delta) for s in sigmas]
    return np.min(bnds)

def maurer_bound(lgr, xs, ys, sigma, tile=TILE, delta=0.05):
    err = stochastic_train_err(lgr, xs, ys, sigma, tile=tile)
    numerator = get_kl(lgr, SIGMA_PRIOR, sigma) + np.log(2 * np.sqrt(len(ys)) / delta)
    return invert_small_kl(err, numerator / len(ys))



def debiased_errs(lgr, xs, ys, debiases, sigma, tile=TILE):
    "Get two types of empirical excess risk with lgr on xs, ys with debiases."
    xs_tile = np.tile(xs, [tile, 1])
    ys_tile = np.tile(ys, tile)
    deb_tile = np.tile(debiases, tile)
    ex_err = is_err(predict_perturbed(lgr, xs_tile, sigma), ys_tile) - deb_tile
    err_plus = np.mean(np.maximum(ex_err, 0))
    err_minus = np.mean(np.maximum(-ex_err, 0))
    return err_plus, err_minus


def full_bound_calc(lgr, xs, ys, debias, kl, sigma, tile=TILE, delta=0.05, debias_bnd=None):
    m = len(ys)
    if debias_bnd is None:
        debias_bnd = invert_small_kl(np.mean(debias), np.log(2/delta) / m)
    err_plus, err_minus = debiased_errs(lgr, xs, ys, debias, sigma, tile)
    numerator = kl + np.log(4 * m / delta)
    # TODO: could add a debias_bnd input that can be a little tighter. (e.g. Bernoulli)
    return inv_kl(err_plus, err_minus, numerator / m) + debias_bnd


def online_debias_bound(lgr, xs, ys, debias, sigma, tile=TILE, delta=0.05, debias_bnd=None):
    "Calculate the bound using only online sequence of debias, no fancy prior/debias bound."
    kl = get_kl(lgr, SIGMA_PRIOR, sigma)
    return full_bound_calc(lgr, xs, ys, debias, kl, sigma, tile, delta, debias_bnd)


def get_step_debiases(xs, ys, start=150, skip=150):
    debiases = np.array([0] * start)
    for i in range(start, len(ys), skip):
        x_ = xs[:i]
        y_ = ys[:i]
        lgr_ = fit_lgr(x_, y_)
        errs_ = (lgr_.predict(xs[i:i+skip]) != ys[i:i+skip]).astype(int)
        debiases = np.concatenate([debiases, errs_])
    return np.array(debiases)




def c_eta(eta):
    "C_eta term in Unexpected Bernstein."
    return -1 - xlogy(1/eta, 1-eta)


def unexpected_bound(lgr, xs, ys, debias, sigma, tile=TILE, debias_bnd=None, delta=0.05):
    m = len(ys)
    if debias_bnd is None:
        debias_bnd = invert_small_kl(np.mean(debias), np.log(2/delta) / m)
    err_plus, err_minus = debiased_errs(lgr, xs, ys, debias, sigma, tile)

    v_hat = err_plus + err_minus   # Only works with misclassification 0-1 losses.
    err = stochastic_train_err(lgr, xs, ys, sigma, tile=tile)

    k_max = int(np.ceil(np.log2(0.5 * np.sqrt(m / -np.log(delta)))))
    kl_term = (get_kl(lgr, SIGMA_PRIOR, sigma) - np.log(delta / k_max)) / len(ys)

    def get_bnd(eta):
        return err + c_eta(eta) * v_hat + kl_term / eta + debias_bnd

    bnds = [get_bnd(0.5 ** k) for k in range(k_max+1)]

    return np.min(bnds)






def get_dataset_results(dataset_name, no_seeds=20, corrupted=0., no_debias=False):
    "Run all the bounds on dataset_name."
    test = []
    maurer = []
    unexpected = []
    online_ours = []

    for seed in tqdm(range(no_seeds)):
        # Load dataset and fit non-random predictor.
        X_train, X_test, y_train, y_test = read_dataset(dataset_name, seed, corrupted)
        lgr_full = fit_lgr(X_train, y_train)
        test.append(1 - lgr_full.score(X_test, y_test))

        m = len(y_train)
        m_pre = int(np.floor(m / 2.))
        X_pre, X_bnd, y_pre, y_bnd = X_train[:m_pre], X_train[m_pre:], y_train[:m_pre], y_train[m_pre:]
        h_star = fit_lgr(X_pre, y_pre)

        maurer.append(opt_bound(partial(maurer_bound, lgr_full, X_train, y_train), m))

        # Bounds using online debiasing.
        if no_debias:
            fix_debias = 0.0
            debias = fix_debias * np.ones(len(y_train))
            online_ours.append(opt_bound(partial(online_debias_bound, lgr_full, X_train, y_train, debias, debias_bnd=fix_debias), m))
            unexpected.append(opt_bound(partial(unexpected_bound, lgr_full, X_train, y_train, debias, debias_bnd=fix_debias), m))
        else:
            debias = get_step_debiases(X_train, y_train)
            online_ours.append(opt_bound(partial(online_debias_bound, lgr_full, X_train, y_train, debias), m))
            unexpected.append(opt_bound(partial(unexpected_bound, lgr_full, X_train, y_train, debias), m))

    res = OrderedDict({
        "Dataset": dataset_name,
        "Test Err": (np.mean(test), np.std(test)),
        "Maurer": (np.mean(maurer), np.std(maurer)),
        "Online Debias": (np.mean(online_ours), np.std(online_ours)),
        "Unexpected": (np.mean(unexpected), np.std(unexpected)),
    })
    print_results_summary_table(res)
    return res

def print_results_summary_table(res):
    print("-"*30)
    for k, v in res.items():
        if isinstance(v, tuple):
            print(f"{k}: \t {v[0]:.3f} \u00B1 {v[1]:.3f}")
        else:
            print(f"{k}: \t {v}")
    print("-"*30)



if __name__ == "__main__":

    SIGMA_PRIOR = np.sqrt(0.5)
    DATASET_NAMES = ("haberman", "breast-cancer", "tictactoe", "bank-notes", "kr-vs-kp", "spam", "mushroom", "adults")

    results = []
    for name in DATASET_NAMES:
        results.append(get_dataset_results(name, no_seeds=20, no_debias=False))


    print("\n\n")
    print("Dataset", "Test    ", "Maurer  ", "Bernstein", "Ours", sep=17*" ")
    for r in results:
        out = lambda field: f"{r[field][0]:.4f} \u00B1 {r[field][1]:.4f}"
        print(f"{r['Dataset']: <18}", out("Test Err"), out("Maurer"), out("Unexpected"), out("Online Debias"), sep=" \t")

    print("\n\nFor Latex:")
    for r in results:
        out = lambda field: f"{r[field][0]:.4f} \\(\\pm\\) {r[field][1]:.4f}"
        print(f"{r['Dataset']: <15}", out("Test Err"), out("Maurer"), out("Unexpected"), out("Online Debias"), sep="  &  ", end="  \\\\\n")
