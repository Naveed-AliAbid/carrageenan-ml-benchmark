"""Synthetic data-generating process (DGP) for metal-crosslinked carrageenan hydrogels.

ALL DATA PRODUCED HERE ARE SYNTHETIC. Every value follows from the equations and
assumed parameters below; nothing is measured. Ion constants are literature values
(Shannon 1976 ionic radii, CN = 6; Marcus 1991 hydration Gibbs energies) used only
to give the virtual ion classes realistic, distinct descriptors.
"""
import numpy as np
import pandas as pd
from scipy.stats import qmc

KB = 1.380649e-23      # J/K
T = 298.15             # K
RT = 8.314e-3 * T      # kJ/mol

# --- ion descriptors (literature constants) ---------------------------------
IONS = pd.DataFrame({
    "ion":     ["Cu", "Zn", "Co", "Fe", "Cd", "Hg"],
    "r_ion":   [0.73, 0.74, 0.745, 0.78, 0.95, 1.02],          # Angstrom, Shannon 1976, CN6 (HS for Co, Fe)
    "dG_hyd":  [-2010., -1955., -1915., -1840., -1755., -1760.],  # kJ/mol, Marcus 1991 (approx.)
})

# --- assumed ground-truth parameters (the "answers" the workflow must recover) -
TRUE = dict(
    kappa=0.08,      # |E_bind| = kappa * |dG_hyd|           (assumed)
    K0=2.0,          # site-binding constant at |E_bind| = 150 kJ/mol (assumed)
    b_K=0.10,        # sensitivity of K to binding energy (assumed)
    rho_max=2.0,     # max crosslink density, 1e24 m^-3 at 3 wt% polymer (assumed)
    Q_ref=20.0,      # equilibrium swelling at rho = 1e24 m^-3 (assumed)
    s0=1.6,          # conductivity scale, mS/cm per wt% (assumed)
    A=7.0,           # lifetime prefactor, days (assumed)
    alpha=0.05,      # lifetime sensitivity to |E_bind|/RT (assumed)
    beta=0.50,       # lifetime exponent on crosslink density (assumed)
)
C_RANGE = (1.0, 3.0)   # polymer fraction, wt%
R_RANGE = (0.2, 1.0)   # ion : sulfate molar ratio


def ground_truth(c, R, dG, p=TRUE):
    """Noise-free properties from formulation variables and ion hydration energy."""
    E = p["kappa"] * np.abs(dG)                                  # kJ/mol
    K = p["K0"] * np.exp(p["b_K"] * (E - 150.0) / RT)
    theta = K * R / (1.0 + K * R)                                # Langmuir site occupancy
    rho = p["rho_max"] * (c / 3.0) * theta                       # 1e24 m^-3
    G = rho * 1e24 * KB * T                                      # Pa, affine rubber elasticity
    Q = p["Q_ref"] * rho ** (-3.0 / 5.0)                         # Flory-type swelling scaling
    sigma = p["s0"] * c * (1.0 - 0.5 * theta)                    # mS/cm, bound ions do not conduct
    ttf = p["A"] * np.exp(p["alpha"] * E / RT) * rho ** p["beta"]  # days
    return dict(E_bind=E, theta=theta, rho=rho, G=G, Q=Q, sigma=sigma, TTF50=ttf)


def make_dataset(n_form_per_ion=4, n_rep=5, cv=0.04, cv_batch=0.02, n_nuisance=4, seed=42):
    """Formulations by Latin hypercube per ion class; replicates with multiplicative noise.

    cv        : replicate-level coefficient of variation (lognormal)
    cv_batch  : formulation-level shared offset (mimics batch-to-batch variation)
    """
    rng = np.random.default_rng(seed)
    rows = []
    fid = 0
    for _, ion in IONS.iterrows():
        lhs = qmc.LatinHypercube(d=2, seed=rng).random(n_form_per_ion)
        cs = C_RANGE[0] + lhs[:, 0] * (C_RANGE[1] - C_RANGE[0])
        Rs = R_RANGE[0] + lhs[:, 1] * (R_RANGE[1] - R_RANGE[0])
        for c, R in zip(cs, Rs):
            gt = ground_truth(c, R, ion.dG_hyd)
            nuis = rng.normal(size=n_nuisance)        # descriptors with no causal effect
            batch = {k: rng.normal(0, cv_batch) for k in ("G", "Q", "sigma", "TTF50")}
            for rep in range(n_rep):
                obs = {k: gt[k] * np.exp(batch[k] + rng.normal(0, cv)) for k in ("G", "Q", "sigma", "TTF50")}
                row = dict(formulation=fid, ion=ion.ion, replicate=rep, c=c, R=R,
                           r_ion=ion.r_ion, dG_hyd=ion.dG_hyd,
                           **{f"nuis{j+1}": nuis[j] for j in range(n_nuisance)},
                           **{f"true_{k}": v for k, v in gt.items()},
                           **obs)
                rows.append(row)
            fid += 1
    return pd.DataFrame(rows)


FEATURES = ["c", "R", "r_ion", "dG_hyd", "nuis1", "nuis2", "nuis3", "nuis4"]
TRUE_DRIVERS = {"c", "R", "dG_hyd"}   # r_ion is correlated with dG_hyd but has no causal role


# --- degradation scenario ------------------------------------------------------
DEG_TRUE = dict(t50_control=105.0, delta_t=180.0, k=0.06, G0=40.0, CN0=5.3, CN_min=2.4,
                sig0=2.0, dsig=1.8, t50_jitter=3.0, obs_cv=0.03)


def degradation_trajectories(n_traj=5, stabilized=False, seed=0, p=DEG_TRUE, hours=np.arange(0, 301)):
    """Logistic loss of intact crosslinks p(t); observables derived from p(t).
    The stabilization shift delta_t is an INPUT of the scenario, not a result."""
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n_traj):
        t50 = p["t50_control"] + (p["delta_t"] if stabilized else 0.0) + rng.normal(0, p["t50_jitter"])
        frac = 1.0 / (1.0 + np.exp(p["k"] * (hours - t50)))
        G = p["G0"] * frac
        sigma = p["sig0"] + p["dsig"] * (1 - frac)
        CN = p["CN_min"] + (p["CN0"] - p["CN_min"]) * frac
        noise = lambda v: v * np.exp(rng.normal(0, p["obs_cv"], size=v.shape))
        out.append(pd.DataFrame(dict(traj=j, t=hours, true_t50=t50, G=noise(G) + 0.0,
                                     sigma=noise(sigma), CN=noise(CN),
                                     group="stabilized" if stabilized else "control")))
    return pd.concat(out, ignore_index=True)
