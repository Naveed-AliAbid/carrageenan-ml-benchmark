"""Additional benchmarks (all data synthetic). Run: python3 extra_analysis.py  (~15 min on 2 cores)
 A. leave-one-ion-out extrapolation        B. grey-box misspecification
 C. formulations vs replicates at fixed budget (scored against the noise-free truth)
 D. GPR interval calibration and grouped recalibration
 E. extended learning curve to N = 480 (CV = 4%)"""
import json, time
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import least_squares
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
import dgp
from models import q2, gpr, cv_predict, greybox_fit, greybox_predict

OUT = Path("results"); RES = {}; t0 = time.time()
df = pd.read_csv(OUT / "synthetic_dataset.csv")

# A. leave-one-ion-out: can a model predict a metal it has never seen?
loio = []
for m in ["Grey-box", "Ridge", "GPR", "SVR", "RF", "MLP"]:
    y, yh, _ = cv_predict(df, "TTF50", m, groups=df.ion.values, n_splits=6)
    loio.append(dict(model=m, Q2_loio=q2(y, yh), RMSE_loio=float(np.sqrt(np.mean((np.exp(y) - np.exp(yh)) ** 2)))))
loio = pd.DataFrame(loio); RES["loio"] = loio.to_dict("records"); print(loio.round(3), round(time.time() - t0))

# B. misspecified grey-box forms
def fit_form(d, form):
    c, R, dG, y = d.c.values, d.R.values, d.dG_hyd.values, np.log(d.TTF50.values)
    E = dgp.TRUE["kappa"] * np.abs(dG)
    def pred(th):
        if form == "no saturation":          # theta = R : Langmuir saturation ignored
            A, a, b = th; rho = dgp.TRUE["rho_max"] * c / 3 * R
            return np.log(A) + a * E / dgp.RT + b * np.log(rho)
        if form == "no binding term":        # alpha = 0 : ion chemistry ignored
            A, b, K0 = th; p = dict(dgp.TRUE, K0=K0)
            rho = dgp.ground_truth(c, R, dG, p)["rho"]; return np.log(A) + b * np.log(rho)
    x0 = {"no saturation": [5, .04, .4], "no binding term": [100, .4, 1]}[form]
    lb = {"no saturation": [1e-3, 0, 0], "no binding term": [1e-3, 0, 1e-3]}[form]
    ub = {"no saturation": [1e4, .2, 3], "no binding term": [1e5, 3, 100]}[form]
    th = least_squares(lambda t: pred(t) - y, x0=x0, bounds=(lb, ub)).x
    return th, pred

mis = []
for form in ["correct", "no saturation", "no binding term"]:
    y_all, yh_all = [], []
    for tr, te in GroupKFold(5).split(df, groups=df.formulation):
        dtr, dte = df.iloc[tr], df.iloc[te]
        if form == "correct":
            th = greybox_fit(dtr); yh = np.log(greybox_predict(th, dte.c.values, dte.R.values, dte.dG_hyd.values))
        else:
            th, _ = fit_form(dtr, form); _, pf = fit_form(dte, form)  # pf built on test inputs
            yh = pf(th)
        y_all += list(np.log(dte.TTF50.values)); yh_all += list(yh)
    full = greybox_fit(df) if form == "correct" else fit_form(df, form)[0]
    mis.append(dict(form=form, Q2=q2(np.array(y_all), np.array(yh_all)), params=[float(v) for v in full]))
RES["misspec"] = mis; print(pd.DataFrame(mis), round(time.time() - t0))

# C. fixed budget of 120 samples: formulations x replicates, scored against noise-free truth
budget = []
for nf, rep in [(1, 20), (2, 10), (4, 5), (5, 4), (10, 2), (20, 1)]:
    for s in range(10):
        d = dgp.make_dataset(n_form_per_ion=nf, n_rep=rep, seed=2000 + s)
        truth = np.log(d.true_TTF50.values)
        for m in ["Grey-box", "GPR", "RF"]:
            _, yh, _ = cv_predict(d, "TTF50", m)
            budget.append(dict(n_form=6 * nf, n_rep=rep, seed=s, model=m, Q2_truth=q2(truth, yh)))
    print("budget", nf, rep, round(time.time() - t0))
budget = pd.DataFrame(budget); budget.to_csv(OUT / "budget.csv", index=False)

# D. GPR calibration: raw intervals vs grouped (nested) recalibration of the predictive SD
X, y, g = df[dgp.FEATURES].values, np.log(df.TTF50.values), df.formulation.values
mu, sd, scale = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
for tr, te in GroupKFold(5).split(X, y, g):
    zin = []
    for itr, ite in GroupKFold(4).split(X[tr], y[tr], g[tr]):
        m = gpr().fit(X[tr][itr], y[tr][itr]); a, b = m.predict(X[tr][ite], return_std=True)
        zin += list(np.abs(y[tr][ite] - a) / b)
    m = gpr().fit(X[tr], y[tr]); a, b = m.predict(X[te], return_std=True)
    mu[te], sd[te] = a, b
    scale[te] = np.quantile(zin, 0.95) / 1.96          # inflate SD so 95% of inner residuals are covered
z = np.abs(y - mu) / sd
levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
qz = norm.ppf((1 + levels) / 2)
cal = dict(levels=levels.tolist(), raw=[float(np.mean(z <= q)) for q in qz],
           recal=[float(np.mean(z / scale <= q)) for q in qz], scale_mean=float(scale.mean()))
RES["calibration"] = cal; print(cal, round(time.time() - t0))

# E. extended learning curve, CV = 4%
lc = []
for nf in [2, 4, 8, 16]:
    for s in range(5):
        d = dgp.make_dataset(n_form_per_ion=nf, cv=0.04, cv_batch=0.02, seed=3000 + s)
        for m in ["Grey-box", "GPR", "Ridge", "RF", "MLP"]:
            yy, yh, _ = cv_predict(d, "TTF50", m)
            lc.append(dict(N=len(d), seed=s, model=m, Q2=q2(yy, yh), ceiling=float(q2(yy, np.log(d.true_TTF50.values)))))
    print("lc", nf, round(time.time() - t0))
lc = pd.DataFrame(lc); lc.to_csv(OUT / "learning_curve.csv", index=False)

json.dump(RES, open(OUT / "extra_results.json", "w"), indent=1, default=float)
print("done", round(time.time() - t0))
