"""Null scenario (true delta_t = 0): false-positive rate of point-wise vs replicate-level tests."""
import json, numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import dgp
p0 = dict(dgp.DEG_TRUE, delta_t=0.0)
logistic = lambda t, G0, k, t50: G0 / (1 + np.exp(k * (t - t50)))
rows = []
for jit in [3.0, 15.0]:
    p = dict(p0, t50_jitter=jit)
    for s in range(500):
        a = dgp.degradation_trajectories(stabilized=False, seed=7 * s, p=p)
        b = dgp.degradation_trajectories(stabilized=True, seed=7 * s + 3, p=p)
        pw = stats.ttest_ind(a.G, b.G, equal_var=False).pvalue
        fa = [curve_fit(logistic, t.t, t.G, p0=[40, .05, 150], maxfev=20000)[0][2] for _, t in a.groupby("traj")]
        fb = [curve_fit(logistic, t.t, t.G, p0=[40, .05, 150], maxfev=20000)[0][2] for _, t in b.groupby("traj")]
        rp = stats.ttest_ind(fa, fb, equal_var=False).pvalue
        rows.append(dict(jitter=jit, p_pointwise=pw, p_replicate=rp))
d = pd.DataFrame(rows); d.to_csv("results/null_test.csv", index=False)
out = d.groupby("jitter").agg(fpr_pointwise=("p_pointwise", lambda x: float(np.mean(x < 0.05))),
                              fpr_replicate=("p_replicate", lambda x: float(np.mean(x < 0.05)))).reset_index()
print(out); json.dump(out.to_dict("records"), open("results/null_test.json", "w"), indent=1)
