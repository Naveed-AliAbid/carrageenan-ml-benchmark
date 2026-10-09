"""Draw the five manuscript figures (600 dpi PNG, LZW TIFF, vector PDF) from results/.
Run after analysis.py, null_test.py and extra_analysis.py:  python3 figures.py"""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import dgp

R = Path("results"); F = Path("figures"); F.mkdir(exist_ok=True)
for old in F.glob("Fig*"):
    old.unlink()
res = json.load(open(R / "results.json")); ex = json.load(open(R / "extra_results.json"))
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "savefig.dpi": 600, "figure.dpi": 150})
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MC = {"Grey-box": CAT[0], "GPR": CAT[1], "Ridge": CAT[2], "SVR": CAT[3], "RF": CAT[4], "MLP": CAT[5]}
IC = dict(zip(dgp.IONS.ion, CAT))
GREY, INK, LIGHT = "#8a8a8a", "#1f1f1f", "#c9c9c9"
ORDER = ["Grey-box", "GPR", "Ridge", "SVR", "RF", "MLP"]


def tag(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")


def save(fig, name):
    fig.savefig(F / f"{name}.png", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(F / f"{name}.tiff", bbox_inches="tight", pad_inches=0.05, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


df = pd.read_csv(R / "synthetic_dataset.csv")
bench = pd.DataFrame(res["benchmark"])

# =========================== Figure 1: ground truth and model performance ============
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.9), gridspec_kw={"hspace": 0.55, "wspace": 0.38})
ax = axs[0, 0]
Rg = np.linspace(0.2, 1.0, 100)
for _, ion in dgp.IONS.iterrows():
    ax.plot(Rg, dgp.ground_truth(2.0, Rg, ion.dG_hyd)["G"] / 1000, color=IC[ion.ion], lw=1.6)
ax.set_xlabel("Ion : sulfate ratio, R"); ax.set_ylabel("True G′ at c = 2 wt% (kPa)")
ax.set_xlim(0.2, 1.0); tag(ax, "(a)")
ax = axs[0, 1]
jit = np.random.default_rng(0)
for ion, d in df.groupby("ion", sort=False):
    ax.scatter(np.abs(d.dG_hyd) + jit.normal(0, 5, len(d)), d.TTF50, s=12, color=IC[ion], alpha=0.8, lw=0)
ax.set_xlabel("|ΔG$_{hyd}$| (kJ mol⁻¹, jittered)"); ax.set_ylabel("Observed TTF$_{50}$ (d)"); tag(ax, "(b)")
ax.legend(handles=[Line2D([], [], color=IC[i], lw=2, label=f"{i}²⁺") for i in dgp.IONS.ion], ncol=1,
          loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=1.4, title="Ion class", title_fontsize=7)

ax = axs[1, 0]
yy = np.arange(len(ORDER))[::-1]
h = 0.36
for k, (tgt, alpha, lab) in enumerate([("TTF50", 1.0, "TTF$_{50}$"), ("G", 0.45, "G′")]):
    bb = bench[bench.target == tgt].set_index("model")
    for i, m in zip(yy, ORDER):
        if m not in bb.index:
            continue
        v = bb.loc[m, "Q2_log"]; yp = i + (0.5 - k) * h
        ax.barh(yp, v, h * 0.86, color=MC[m], alpha=alpha, lw=0)
        if v > 0.3:
            ax.text(v - 0.02, yp, f"{v:.2f}", ha="right", va="center", fontsize=6.5,
                    color="white" if k == 0 else INK, fontweight="bold")
        else:
            ax.text(v + 0.02, yp, f"{v:.2f}", ha="left", va="center", fontsize=6.5, color=INK)
ax.axvline(res["ceiling_TTF50"], color=INK, ls=(0, (3, 2)), lw=0.8)
ax.text(res["ceiling_TTF50"], len(ORDER) - 0.35, "noise\nceiling", ha="center", va="bottom", fontsize=6.5, color=INK)
ax.set_yticks(yy); ax.set_yticklabels(ORDER); ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, len(ORDER) - 0.4)
ax.set_xlabel("Grouped-CV Q² (log target)")
ax.legend(handles=[Patch(color=GREY, label="TTF$_{50}$"), Patch(color=GREY, alpha=0.45, label="G′")],
          loc="lower right", bbox_to_anchor=(1.0, 0.02))
tag(ax, "(c)", x=-0.24)

ax = axs[1, 1]
o = res["oof"]["TTF50"]; yv, yh = np.array(o["y"]), np.array(o["yhat"]); lo, hi = np.array(o["lo"]), np.array(o["hi"])
inside = (yv >= lo) & (yv <= hi)
for msk, col, lab, a in [(inside, MC["GPR"], "interval covers value", 0.55), (~inside, INK, "interval misses value", 0.9)]:
    ax.errorbar(yv[msk], yh[msk], yerr=[yh[msk] - lo[msk], hi[msk] - yh[msk]], fmt="o", ms=2.6, color=col,
                ecolor=col, elinewidth=0.45, alpha=a, label=lab)
ax.plot([55, 215], [55, 215], color=GREY, ls=(0, (3, 2)), lw=0.8)
ax.set_xlim(55, 215); ax.set_ylim(55, 215); ax.set_aspect("equal")
ax.set_xlabel("Observed TTF$_{50}$ (d)"); ax.set_ylabel("Predicted TTF$_{50}$ (d)")
g = bench[(bench.target == "TTF50") & (bench.model == "GPR")].iloc[0]
ax.text(0.97, 0.05, f"GPR, grouped CV\nQ² = {g.Q2:.2f}\n95% coverage = {g.CI95_coverage*100:.0f}%",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.0); tag(ax, "(d)", y=1.1)
save(fig, "Figure1_benchmark")

# =========================== Figure 2: validation design =============================
lk = pd.DataFrame(res["leakage"]); lo_ = pd.DataFrame(ex["loio"]).set_index("model")
bud = pd.read_csv(R / "budget.csv"); cal = ex["calibration"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.55, "width_ratios": [1.15, 1, 1]})
ax = axs[0]
mods = ["RF", "GPR", "Ridge", "MLP"]
for i, m in enumerate(mods):
    r = lk[(lk.model == m) & (lk.cv == "random")].Q2.iloc[0]
    gq = lk[(lk.model == m) & (lk.cv == "group")].Q2.iloc[0]
    li = lo_.loc[m, "Q2_loio"]
    ax.plot([min(gq, r, li), max(gq, r, li)], [i, i], color=LIGHT, lw=1.0, zorder=1)
    ax.scatter(r, i + 0.22, s=28, facecolor="white", edgecolor=MC[m], lw=1.3, zorder=3)
    ax.scatter(gq, i, s=28, color=MC[m], zorder=3)
    ax.scatter(li, i - 0.22, s=28, marker="D", facecolor=MC[m], edgecolor="white", lw=0.6, zorder=4)
ax.set_yticks(range(len(mods))); ax.set_yticklabels(mods); ax.set_xlim(-0.1, 1.05); ax.set_ylim(-0.6, 3.6)
ax.set_xlabel("Out-of-fold Q², TTF$_{50}$")
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="random split (leaky)"),
                   Line2D([], [], marker="o", ls="", color=INK, label="grouped by formulation"),
                   Line2D([], [], marker="D", ls="", color=INK, label="leave one ion out")],
          loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, handletextpad=0.2)
tag(ax, "(a)", x=-0.3, y=1.34)
ax = axs[1]
for m in ["Grey-box", "GPR", "RF"]:
    s = bud[bud.model == m].groupby("n_form").Q2_truth; mu, sd = s.mean(), s.std()
    ax.errorbar(mu.index, mu.values, yerr=sd.values, color=MC[m], marker="o", ms=3.2, lw=1.4, capsize=2,
                elinewidth=0.7, label=m)
ax.set_xscale("log"); ax.set_xticks([6, 12, 24, 60, 120]); ax.set_xticklabels(["6", "12", "24", "60", "120"])
ax.minorticks_off()
ax2 = ax.secondary_xaxis("top"); ax2.set_xticks([6, 12, 24, 60, 120]); ax2.set_xticklabels(["20", "10", "5", "2", "1"]); ax2.minorticks_off()
ax2.set_xlabel("replicates each", fontsize=7); ax2.tick_params(labelsize=6.5)
ax.set_xlabel("Formulations (120 samples total)"); ax.set_ylabel("Q² against noise-free truth")
ax.set_ylim(-0.75, 1.05); ax.legend(loc="lower right"); tag(ax, "(b)", x=-0.3, y=1.34)
ax = axs[2]
lv = np.array(cal["levels"]) * 100
ax.plot([30, 100], [30, 100], color=GREY, ls=(0, (3, 2)), lw=0.8)
ax.plot(lv, np.array(cal["raw"]) * 100, marker="o", ms=3.2, color=MC["GPR"], lw=1.4, label="GPR, raw")
ax.plot(lv, np.array(cal["recal"]) * 100, marker="s", ms=3.2, color=MC["Grey-box"], lw=1.4,
        label=f"recalibrated (×{cal['scale_mean']:.2f})")
ax.set_xlim(30, 100); ax.set_ylim(30, 100)
ax.set_xlabel("Nominal coverage (%)"); ax.set_ylabel("Observed coverage (%)")
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=6.8); tag(ax, "(c)", x=-0.3, y=1.34)
save(fig, "Figure2_validation")

# =========================== Figure 3: data requirements and physics recovery ========
sw = pd.read_csv(R / "sweep.csv"); perf = sw[sw.model != "params"]; pr = sw[sw.model == "params"]
lc = pd.read_csv(R / "learning_curve.csv")
fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.62)
top = [fig.add_subplot(gs[0, i]) for i in range(4)]
for j, (ax, cvn) in enumerate(zip(top, [0.01, 0.04, 0.10, 0.20])):
    d = perf[perf.cv == cvn]; ceil = d.groupby("N").ceiling.mean()
    ax.plot(ceil.index, ceil.values, color=INK, ls=(0, (3, 2)), lw=0.8)
    for m in ["Grey-box", "GPR", "Ridge", "RF"]:
        s = d[d.model == m].groupby("N").Q2_log; mu, sd = s.mean(), s.std(); ok = mu.values > -0.05
        ax.errorbar(mu.index[ok], mu.values[ok], yerr=sd.values[ok], color=MC[m], lw=1.3, ms=3, marker="o",
                    capsize=1.5, elinewidth=0.6)
        for x, v in zip(mu.index[~ok], mu.values[~ok]):
            ax.plot(x, -0.13, marker="v", color=MC[m], ms=4, clip_on=False)
            ax.text(x + 12, -0.13, f"{v:.1f}", fontsize=6, color=MC[m], va="center")
    ax.set_title(f"Replicate CV = {int(cvn*100)}%"); ax.set_xticks([60, 120, 240]); ax.set_ylim(-0.17, 1.03)
    ax.set_xlim(40, 260); ax.set_xlabel("N samples")
    if j == 0:
        ax.set_ylabel("Grouped-CV Q²")
    else:
        ax.set_yticklabels([])
top[0].legend(handles=[Line2D([], [], color=MC[m], marker="o", ms=3, lw=1.3, label=m) for m in
                       ["Grey-box", "GPR", "Ridge", "RF", "MLP"]] +
              [Line2D([], [], color=INK, ls=(0, (3, 2)), lw=0.8, label="noise ceiling")],
              loc="lower left", bbox_to_anchor=(0, 1.15), ncol=6, columnspacing=0.9, handlelength=1.6)
tag(top[0], "(a)", x=-0.42, y=1.12)
bot = [fig.add_subplot(gs[1, i]) for i in range(4)]
ax = bot[0]
ceil = lc.groupby("N").ceiling.mean(); ax.plot(ceil.index, ceil.values, color=INK, ls=(0, (3, 2)), lw=0.8)
for m in ["Grey-box", "GPR", "Ridge", "RF", "MLP"]:
    s = lc[lc.model == m].groupby("N").Q2; mu, sd = s.mean(), s.std(); ok = mu.values > -0.05
    ax.errorbar(mu.index[ok], mu.values[ok], yerr=sd.values[ok], color=MC[m], marker="o", ms=3, lw=1.3,
                capsize=1.5, elinewidth=0.6)
    for x, v in zip(mu.index[~ok], mu.values[~ok]):
        ax.plot(x, -0.13, marker="v", color=MC[m], ms=4, clip_on=False)
ax.set_xscale("log"); ax.set_xticks([60, 120, 240, 480]); ax.set_xticklabels(["60", "120", "240", "480"])
ax.minorticks_off(); ax.set_ylim(-0.17, 1.03); ax.set_xlabel("N samples"); ax.set_ylabel("Grouped-CV Q²")
ax.set_title("CV = 4%")
tag(ax, "(b)", x=-0.5)
for ax, p, lab, t in [(bot[1], "alpha", "α", "(c)"), (bot[2], "beta", "β", "(d)")]:
    grp = [pr[(pr.cv == c) & (pr.N == 120)][p].values for c in [0.01, 0.04, 0.10, 0.20]]
    bp = ax.boxplot(grp, widths=0.55, patch_artist=True, medianprops=dict(color=INK, lw=0.9),
                    whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7), flierprops=dict(ms=2, mec=GREY))
    for patch in bp["boxes"]:
        patch.set(facecolor=MC["Grey-box"], alpha=0.35, lw=0.7)
    ax.axhline(dgp.TRUE[p], color=INK, ls=(0, (3, 2)), lw=0.8)
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.18 * (hi_ - lo_))
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["1", "4", "10", "20"]); ax.set_xlabel("Replicate CV (%)")
    ax.set_ylabel(f"Recovered {lab}"); ax.set_title("N = 120")
    ax.text(0.04, 0.96, f"dashed: true {lab} = {dgp.TRUE[p]}", transform=ax.transAxes, fontsize=6.3, va="top")
    tag(ax, t, x=-0.5)
ax = bot[3]
ms = ex["misspec"]; names = ["correct", "no saturation", "no binding term"]
q = [m["Q2"] for m in ms]; al = [m["params"][1] if m["form"] != "no binding term" else np.nan for m in ms]
xs = np.arange(3)
ax.bar(xs, q, 0.72, color=[MC["Grey-box"], GREY, LIGHT], lw=0)
for x, v, a in zip(xs, q, al):
    ax.text(x, v - 0.04, f"{v:.2f}", ha="center", va="top", fontsize=6, color="white" if x < 2 else INK,
            fontweight="bold")
    ax.text(x, 0.06, "α = n/a" if np.isnan(a) else f"α = {a:.3f}", ha="center", va="bottom", fontsize=6,
            color="white" if x < 2 else INK, rotation=90)
ax.set_xticks(xs); ax.set_xticklabels(["correct", "no\nsatur-\nation", "no ion\nterm"], fontsize=6.8)
ax.set_ylim(0, 1.0); ax.set_ylabel("Grouped-CV Q²"); ax.set_title("Grey-box form"); tag(ax, "(e)", x=-0.5)
save(fig, "Figure3_data_requirements")

# =========================== Figure 4: interpretation ================================
at = pd.DataFrame(res["attribution"]); L = res["lda"]; cm = np.array(L["cm"]); labs = L["labels"]
sc = pd.read_csv(R / "lda_scores.csv")
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), gridspec_kw={"hspace": 0.5, "wspace": 0.45})
names = ["c", "R", "r$_{ion}$", "ΔG$_{hyd}$", "nuisance 1", "nuisance 2", "nuisance 3", "nuisance 4"]
yy = np.arange(len(at))[::-1]; cols = [MC["Grey-box"] if t else GREY for t in at.true_driver]
ax = axs[0, 0]
ax.barh(yy, at.lasso_coef, 0.62, color=cols, lw=0); ax.axvline(0, color=INK, lw=0.7)
for y_, v in zip(yy, at.lasso_coef):
    if abs(v) > 0.03:
        ax.text(v + (0.008 if v > 0 else -0.008), y_, f"{v:+.2f}", va="center", ha="left" if v > 0 else "right",
                fontsize=6.5)
ax.set_yticks(yy); ax.set_yticklabels(names); ax.set_xlim(-0.3, 0.23)
ax.set_xlabel("LASSO coefficient (standardized)"); tag(ax, "(a)", x=-0.36)
ax = axs[0, 1]
ax.barh(yy, at.perm_mean, 0.62, color=cols, lw=0, xerr=at.perm_sd, error_kw=dict(lw=0.6, capsize=1.8, ecolor=INK))
ax.axvline(0, color=INK, lw=0.7); ax.set_yticks(yy); ax.set_yticklabels([]); ax.set_xlim(-0.004, 0.12)
ax.set_xlabel("GPR permutation importance (ΔMSE)")
ax.legend(handles=[Patch(color=MC["Grey-box"], label="true causal driver"), Patch(color=GREY, label="no causal effect")],
          loc="lower right"); tag(ax, "(b)", x=-0.12)
ax = axs[1, 0]
ax.imshow(cm, cmap="Blues", vmin=0, vmax=20)
for i in range(6):
    for j in range(6):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > 11 else INK)
ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xticklabels([f"{l}" for l in labs]); ax.set_yticklabels(labs)
ax.set_xlabel("Predicted ion class"); ax.set_ylabel("True ion class")
ax.set_title(f"Grouped-CV accuracy {L['accuracy']*100:.0f}% (chance 17%)")
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.tick_params(length=0); tag(ax, "(c)", x=-0.36, y=1.1)
ax = axs[1, 1]
for ion, d in sc.groupby("ion", sort=False):
    ax.scatter(d.LD1, d.LD2, s=11, color=IC[ion], lw=0, alpha=0.85, label=f"{ion}²⁺")
ax.set_xlabel(f"LD1 ({L['explained'][0]*100:.0f}% of separation)"); ax.set_ylabel(f"LD2 ({L['explained'][1]*100:.0f}%)")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.1); tag(ax, "(d)", x=-0.12)
save(fig, "Figure4_interpretation")

# =========================== Figure 5: degradation ===================================
exd = pd.read_csv(R / "degradation_example.csv"); st = pd.read_csv(R / "degradation_studies.csv")
D = res["degradation"]; nt = pd.DataFrame(json.load(open(R / "null_test.json")))
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), gridspec_kw={"hspace": 0.6, "wspace": 0.35})
ax = axs[0, 0]
for (grp, d), col in zip(exd.groupby("group"), [GREY, MC["Grey-box"]]):
    for _, tr in d.groupby("traj"):
        ax.plot(tr.t, tr.G, color=col, lw=0.5, alpha=0.85)
ax.text(132, 26, "control", color=GREY, fontsize=7.5); ax.text(205, 43.5, "stabilized", color=MC["Grey-box"], fontsize=7.5)
ax.set_xlim(0, 300); ax.set_ylim(-2, 48)
ax.set_xlabel("Time (h)"); ax.set_ylabel("Synthetic G′ (kPa)"); ax.set_title("One synthetic study, 5 + 5 trajectories", fontsize=7.5); tag(ax, "(a)", y=1.08)
ax = axs[0, 1]
bins = np.arange(-10, 222, 4)
ax.hist(st.dt_d2, bins=bins, color=GREY, alpha=0.75, label="second-derivative extremum", lw=0)
ax.hist(st.dt_fit, bins=bins, color=MC["Grey-box"], alpha=0.85, label="logistic fit", lw=0)
ax.axvline(D["true_delta"], color=INK, ls=(0, (3, 2)), lw=0.8)
ax.text(D["true_delta"] + 4, 140, "true Δt = 180 h", ha="left", va="top", fontsize=7)
ax.set_ylim(0, 145); ax.set_xlim(-10, 222)
ax.set_xlabel("Estimated Δt (h), 200 studies"); ax.set_ylabel("Studies")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.8)); ax.set_title("Recovery of the imposed shift", fontsize=7.5); tag(ax, "(b)", y=1.08)
ax = axs[1, 0]
cov = [D["cover_boot"] * 100, D["cover_welch"] * 100]
ax.bar([0, 1], cov, 0.5, color=[GREY, MC["Grey-box"]], lw=0)
for i, v in enumerate(cov):
    ax.text(i, v - 4, f"{v:.1f}%", ha="center", va="top", fontsize=7.5, color="white", fontweight="bold")
ax.axhline(95, color=INK, ls=(0, (3, 2)), lw=0.8); ax.text(-0.46, 96.5, "nominal 95%", ha="left", va="bottom", fontsize=6.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["percentile bootstrap", "Welch t-interval"]); ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 108); ax.set_ylabel("Coverage of true Δt (%)"); ax.set_title("95% intervals, n = 5 trajectories per group", fontsize=7.5); tag(ax, "(c)", y=1.08)
ax = axs[1, 1]
xx = np.arange(len(nt)); w = 0.34
ax.bar(xx - w / 2 - 0.02, nt.fpr_pointwise * 100, w, color=GREY, lw=0, label="point-wise (all hourly readings)")
ax.bar(xx + w / 2 + 0.02, nt.fpr_replicate * 100, w, color=MC["Grey-box"], lw=0, label="replicate-level (per trajectory)")
for i, r in nt.iterrows():
    ax.text(i - w / 2 - 0.02, r.fpr_pointwise * 100 + 1.0, f"{r.fpr_pointwise*100:.1f}", ha="center", va="bottom", fontsize=7)
    ax.text(i + w / 2 + 0.02, r.fpr_replicate * 100 + 2.6, f"{r.fpr_replicate*100:.1f}", ha="center", va="bottom", fontsize=7)
ax.axhline(5, color=INK, ls=(0, (3, 2)), lw=0.8); ax.text(-0.58, 5.6, "nominal 5%", ha="left", va="bottom", fontsize=6.5)
ax.set_xticks(xx); ax.set_xticklabels([f"t$_{{50}}$ jitter {int(j)} h" for j in nt.jitter]); ax.set_xlim(-0.6, 1.6)
ax.set_ylim(0, 42); ax.set_ylabel("False-positive rate (%)"); ax.set_title("Null scenario, true Δt = 0", fontsize=7.5)
ax.legend(loc="upper left"); tag(ax, "(d)", y=1.08)
save(fig, "Figure5_degradation")
print("5 figures written")
