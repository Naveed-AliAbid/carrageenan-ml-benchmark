"""Benchmark analyses on the synthetic dataset. Run:  python3 analysis.py
Writes results/results.json and results/*.csv. All data are synthetic (see dgp.py)."""
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
OUT = Path("results"); OUT.mkdir(exist_ok=True)
SEED = 42
RES = {}


from models import q2, gpr, fit_predict, GB_PARAMS, greybox_predict, greybox_fit, cv_predict, noise_ceiling


# ------------------------------------------------------- 1. model benchmark --
t0 = time.time()
df = dgp.make_dataset(seed=SEED)
df.to_csv(OUT / "synthetic_dataset.csv", index=False)
bench = []
oof = {}
for target in ["TTF50", "G"]:
    models = ["Ridge", "RF", "SVR", "MLP", "GPR"] + (["Grey-box"] if target == "TTF50" else [])
    for mname in models:
        y, yh, ys = cv_predict(df, target, mname)
        bt = np.exp(y); bh = np.exp(yh)
        row = dict(target=target, model=mname, Q2_log=q2(y, yh), Q2=q2(bt, bh),
                   RMSE=float(np.sqrt(np.mean((bt - bh) ** 2))), MAPE=float(np.mean(np.abs(bh / bt - 1)) * 100))
        if mname == "GPR":
            lo, hi = np.exp(yh - 1.96 * ys), np.exp(yh + 1.96 * ys)
            row["CI95_coverage"] = float(np.mean((bt >= lo) & (bt <= hi)))
            oof[target] = dict(y=bt.tolist(), yhat=bh.tolist(), lo=lo.tolist(), hi=hi.tolist(),
                               ion=df.ion.tolist())
        bench.append(row)
    RES[f"ceiling_{target}"] = noise_ceiling(df, target)
bench = pd.DataFrame(bench); bench.to_csv(OUT / "benchmark.csv", index=False)
RES["benchmark"] = bench.to_dict("records"); RES["oof"] = oof
th = greybox_fit(df)
RES["greybox_full"] = dict(zip(GB_PARAMS, th.tolist()))
print("benchmark done", round(time.time() - t0), "s"); print(bench.round(3).to_string())

# ------------------------------------------------------------ 2. leakage ----
leak = []
for mname in ["Ridge", "RF", "GPR", "MLP"]:
    for cv in ["random", "group"]:
        y, yh, _ = cv_predict(df, "TTF50", mname, cv=cv)
        leak.append(dict(model=mname, cv=cv, Q2=q2(y, yh)))
leak = pd.DataFrame(leak); leak.to_csv(OUT / "leakage.csv", index=False); RES["leakage"] = leak.to_dict("records")
print("leakage done", round(time.time() - t0), "s"); print(leak.round(3).to_string())

# ------------------------------------------------- 3. noise x sample size ----
sweep = []
SWEEP_DONE = (OUT / "sweep.csv").exists()          # delete results/sweep.csv to recompute (~18 min)
for cvn in ([] if SWEEP_DONE else [0.01, 0.04, 0.10, 0.20]):
    for nf in [2, 4, 8]:
        for s in range(10):
            d = dgp.make_dataset(n_form_per_ion=nf, cv=cvn, cv_batch=cvn / 2, seed=1000 + s)
            ceil = noise_ceiling(d, "TTF50")
            for mname in ["Ridge", "RF", "GPR", "Grey-box"]:
                y, yh, _ = cv_predict(d, "TTF50", mname)
                sweep.append(dict(cv=cvn, N=len(d), seed=s, model=mname, Q2_log=q2(y, yh), ceiling=ceil))
            thd = greybox_fit(d)
            sweep.append(dict(cv=cvn, N=len(d), seed=s, model="params",
                              **dict(zip(GB_PARAMS, thd))))
    print("sweep cv", cvn, round(time.time() - t0), "s")
if not SWEEP_DONE:
    pd.DataFrame(sweep).to_csv(OUT / "sweep.csv", index=False)

# ------------------------------------------------------- 4. attribution ----
X, y, g = df[dgp.FEATURES].values, np.log(df.TTF50.values), df.formulation.values
lasso = make_pipeline(StandardScaler(), LassoCV(cv=list(GroupKFold(5).split(X, y, g)), random_state=SEED))
lasso.fit(X, y)
coef = lasso[-1].coef_
perm = np.zeros((5, len(dgp.FEATURES)))
rng = np.random.default_rng(SEED)
for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
    m = gpr().fit(X[tr], y[tr]); base = np.mean((m.predict(X[te]) - y[te]) ** 2)
    for j in range(X.shape[1]):
        drops = []
        for _ in range(20):
            Xp = X[te].copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(np.mean((m.predict(Xp) - y[te]) ** 2) - base)
        perm[k, j] = np.mean(drops)
attr = pd.DataFrame(dict(feature=dgp.FEATURES, lasso_coef=coef, perm_mean=perm.mean(0), perm_sd=perm.std(0),
                         true_driver=[f in dgp.TRUE_DRIVERS for f in dgp.FEATURES]))
attr.to_csv(OUT / "attribution.csv", index=False); RES["attribution"] = attr.to_dict("records")
print("attribution done"); print(attr.round(4).to_string())

# ----------------------------------------------------------------- 5. LDA ----
F = np.column_stack([df.c, df.R, np.log(df.G), np.log(df.Q), df.sigma, np.log(df.TTF50)])
ions = df.ion.values
pred = np.empty_like(ions)
for tr, te in GroupKFold(6).split(F, ions, g):
    lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()).fit(F[tr], ions[tr])
    pred[te] = lda.predict(F[te])
order = list(dgp.IONS.ion)
cm = confusion_matrix(ions, pred, labels=order)
lda_all = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()).fit(F, ions)
Z = lda_all.transform(F)
RES["lda"] = dict(accuracy=float(np.mean(pred == ions)), labels=order, cm=cm.tolist(),
                  explained=lda_all[-1].explained_variance_ratio_[:2].tolist())
pd.DataFrame(dict(ion=ions, LD1=Z[:, 0], LD2=Z[:, 1])).to_csv(OUT / "lda_scores.csv", index=False)
print("LDA acc", RES["lda"]["accuracy"])

# ---------------------------------------------------------- 6. degradation ----
def logistic(t, G0, k, t50):
    return G0 / (1 + np.exp(k * (t - t50)))


def t50_fit(tr):
    p, _ = curve_fit(logistic, tr.t.values, tr.G.values, p0=[tr.G.iloc[:10].mean(), 0.05, 150], maxfev=20000)
    return p[2]


def t50_second_derivative(tr):
    """Inflection by extremum of the 5-point second derivative of the raw signal (naive method)."""
    v, h = tr.G.values, 1.0
    d2 = (-v[4:] + 16 * v[3:-1] - 30 * v[2:-2] + 16 * v[1:-3] - v[:-4]) / (12 * h ** 2)
    return tr.t.values[2:-2][np.argmax(np.abs(d2))]


deg_rows, studies = [], []
for s in range(200):
    ctl = dgp.degradation_trajectories(stabilized=False, seed=10 * s)
    stb = dgp.degradation_trajectories(stabilized=True, seed=10 * s + 1)
    est = {}
    for grp, dd in [("control", ctl), ("stabilized", stb)]:
        est[grp] = dict(fit=[], d2=[], true=[])
        for j, tr in dd.groupby("traj"):
            est[grp]["fit"].append(t50_fit(tr)); est[grp]["d2"].append(t50_second_derivative(tr))
            est[grp]["true"].append(tr.true_t50.iloc[0])
    a, b = np.array(est["control"]["fit"]), np.array(est["stabilized"]["fit"])
    dt = b.mean() - a.mean()
    boot = [rng.choice(b, 5).mean() - rng.choice(a, 5).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    se = np.sqrt(a.var(ddof=1) / 5 + b.var(ddof=1) / 5)
    from scipy import stats
    dof = se ** 4 / ((a.var(ddof=1) / 5) ** 2 / 4 + (b.var(ddof=1) / 5) ** 2 / 4)
    tq = stats.t.ppf(0.975, dof)
    # pseudo-replication: treat every hourly point as independent
    pw = stats.ttest_ind(ctl.G.values, stb.G.values, equal_var=False)
    rp = stats.ttest_ind(a, b, equal_var=False)
    d2a, d2b = np.array(est["control"]["d2"]), np.array(est["stabilized"]["d2"])
    studies.append(dict(study=s, dt_fit=dt, boot_lo=lo, boot_hi=hi, welch_lo=dt - tq * se, welch_hi=dt + tq * se,
                        dt_d2=d2b.mean() - d2a.mean(),
                        err_fit_ctl=np.mean(a - np.array(est["control"]["true"])),
                        err_d2_ctl=np.mean(d2a - np.array(est["control"]["true"])),
                        sd_fit=np.std(a - np.array(est["control"]["true"]), ddof=1),
                        sd_d2=np.std(d2a - np.array(est["control"]["true"]), ddof=1),
                        p_pointwise=pw.pvalue, p_replicate=rp.pvalue))
    if s == 0:
        pd.concat([ctl, stb]).to_csv(OUT / "degradation_example.csv", index=False)
studies = pd.DataFrame(studies); studies.to_csv(OUT / "degradation_studies.csv", index=False)
true_dt = dgp.DEG_TRUE["delta_t"]
RES["degradation"] = dict(
    true_delta=true_dt, mean_dt_fit=studies.dt_fit.mean(), sd_dt_fit=studies.dt_fit.std(),
    mean_dt_d2=studies.dt_d2.mean(), sd_dt_d2=studies.dt_d2.std(),
    bias_fit=studies.err_fit_ctl.mean(), bias_d2=studies.err_d2_ctl.mean(),
    rmse_t50_fit=float(np.sqrt(np.mean(studies.err_fit_ctl ** 2 + studies.sd_fit ** 2))),
    rmse_t50_d2=float(np.sqrt(np.mean(studies.err_d2_ctl ** 2 + studies.sd_d2 ** 2))),
    cover_boot=float(np.mean((studies.boot_lo <= true_dt) & (studies.boot_hi >= true_dt))),
    cover_welch=float(np.mean((studies.welch_lo <= true_dt) & (studies.welch_hi >= true_dt))),
    median_p_pointwise=float(studies.p_pointwise.median()), median_p_replicate=float(studies.p_replicate.median()))
print("degradation", json.dumps(RES["degradation"], indent=1))

json.dump(RES, open(OUT / "results.json", "w"), indent=1, default=float)
print("total", round(time.time() - t0), "s")
