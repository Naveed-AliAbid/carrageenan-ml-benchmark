"""Shared model, validation and metric functions for the synthetic benchmark."""
import json, warnings, time
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import least_squares, curve_fit
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import dgp

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42


def q2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)


# ---------------------------------------------------------------- models ----
def gpr():
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(len(dgp.FEATURES)), nu=2.5,
                                    length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-2, (1e-6, 1))
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True,
                                                                    n_restarts_optimizer=3, random_state=SEED))


def fit_predict(name, Xtr, ytr, Xte, gtr):
    """Fit one model; SVR/MLP hyperparameters tuned by inner GroupKFold (nested CV)."""
    inner = GroupKFold(n_splits=4)
    if name == "Ridge":
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    elif name == "RF":
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=2)
    elif name == "SVR":
        m = GridSearchCV(make_pipeline(StandardScaler(), SVR(kernel="rbf")),
                         {"svr__C": [1, 10, 100], "svr__gamma": ["scale", 0.05], "svr__epsilon": [0.01, 0.05]},
                         cv=inner, n_jobs=2)
    elif name == "MLP":
        m = GridSearchCV(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(8, 4), max_iter=4000,
                                                                      learning_rate_init=0.01, random_state=SEED)),
                         {"mlpregressor__alpha": [1e-3, 1e-2, 1e-1, 1.0]}, cv=inner, n_jobs=2)
    elif name == "GPR":
        m = gpr()
    if isinstance(m, GridSearchCV):
        m.fit(Xtr, ytr, groups=gtr)
    else:
        m.fit(Xtr, ytr)
    return m, m.predict(Xte)


# grey-box (physics-based) lifetime model: known functional form, unknown parameters
GB_PARAMS = ["A", "alpha", "beta", "K0"]


def greybox_predict(theta, c, R, dG):
    A, alpha, beta, K0 = theta
    p = dict(dgp.TRUE, A=A, alpha=alpha, beta=beta, K0=K0)
    return dgp.ground_truth(c, R, dG, p)["TTF50"]


def greybox_fit(df):
    c, R, dG, y = df.c.values, df.R.values, df.dG_hyd.values, np.log(df.TTF50.values)
    res = least_squares(lambda th: np.log(greybox_predict(th, c, R, dG)) - y,
                        x0=[5.0, 0.04, 0.4, 1.0], bounds=([1e-3, 0, 0, 1e-3], [1e3, 0.2, 3, 100]))
    return res.x


def cv_predict(df, target, model, cv="group", n_splits=5, seed=SEED, groups=None):
    X, y, g = df[dgp.FEATURES].values, np.log(df[target].values), df.formulation.values
    if groups is not None:
        g = np.asarray(groups)
    splitter = GroupKFold(n_splits) if cv == "group" else KFold(n_splits, shuffle=True, random_state=seed)
    yhat, ystd = np.zeros_like(y), np.full_like(y, np.nan)
    for tr, te in splitter.split(X, y, g):
        if model == "Grey-box":
            th = greybox_fit(df.iloc[tr]); t = df.iloc[te]
            yhat[te] = np.log(greybox_predict(th, t.c.values, t.R.values, t.dG_hyd.values))
        else:
            m, p = fit_predict(model, X[tr], y[tr], X[te], g[tr]); yhat[te] = p
            if model == "GPR":
                _, s = m.predict(X[te], return_std=True); ystd[te] = s
    return y, yhat, ystd


def noise_ceiling(df, target):
    """Best achievable Q2 on observed log-values = variance explained by the noise-free truth."""
    y, t = np.log(df[target].values), np.log(df["true_" + target].values)
    return q2(y, t)


