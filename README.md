# Synthetic benchmark: ML workflow for metal-crosslinked carrageenan hydrogels

This repository contains the code, synthetic benchmark data, and figure generation scripts for the manuscript:
*When does machine learning recover the physics? A synthetic-benchmark tutorial for structure-property modelling of metal-crosslinked carrageenan hydrogels*.

## Data declaration

All data in this repository are synthetic. No laboratory experiments were conducted. Values are generated deterministically from the physical equations and specified parameter sets defined in `dgp.py`. Literature values are used solely for physical ion descriptors (Shannon 1976 ionic radii; Marcus 1991 hydration Gibbs energies).

## Repository structure

### Code

| File | Description |
| --- | --- |
| `dgp.py` | Data-generating process defining ground-truth physics, ion descriptors, noise models, and degradation kinetics. |
| `models.py` | Model architectures, hyperparameter tuning pipelines, grey-box non-linear least squares, and grouped cross-validation routines. |
| `analysis.py` | Primary benchmark workflow computing model metrics, data leakage, noise-by-sample grid sweeps, feature attribution, and degradation fitting. |
| `extra_analysis.py` | Secondary benchmark workflow computing leave-one-ion-out extrapolation, grey-box misspecification, replicate allocation, and GPR calibration. |
| `null_test.py` | Monte Carlo null hypothesis testing comparing false-positive rates of pointwise tests versus replicate-level curve fits. |
| `figures.py` | Publication figure generation script producing Figures 1–5 in vector PDF, 600 DPI PNG, and LZW-compressed TIFF formats. |

### Generated data (`results/`)

| File | Description |
| --- | --- |
| `synthetic_dataset.csv` | Synthetic hydrogel training dataset consisting of 24 formulations with 5 replicates each (120 samples total). |
| `benchmark.csv` | Cross-validation performance metrics (Q², log-scale Q², RMSE, MAPE) across Ridge, RF, SVR, MLP, GPR, and Grey-box models. |
| `leakage.csv` | Evaluation of model Q² under random split versus grouped cross-validation demonstrating replicate data leakage. |
| `sweep.csv` | Model performance evaluated across varying noise levels (CV = 1% to 20%) and formulation counts (N = 12, 24, 48). |
| `attribution.csv` | Feature importance rankings for physical and nuisance descriptors from Random Forest and GPR models. |
| `lda_scores.csv` | Cross-validated classification accuracy and confusion matrix for ion-class identification using Linear Discriminant Analysis. |
| `degradation_example.csv` | Synthetic time-course trajectories showing shear modulus degradation under accelerated testing. |
| `degradation_studies.csv` | Inflection-point parameter recovery comparisons between logistic curve fitting and second-derivative detection across 200 runs. |
| `null_test.csv` | Monte Carlo p-values evaluating type I error rates under a true null effect across 500 simulations. |
| `null_test.json` | Empirical false-positive rates comparing pointwise testing with replicate-level curve fitting under null degradation. |
| `budget.csv` | Model performance trade-offs between formulation count and replicate allocation at a fixed total budget of 120 samples. |
| `learning_curve.csv` | Sample size scaling curves for Grey-box, GPR, Ridge, RF, and MLP models from 24 to 480 formulations. |
| `extra_results.json` | Structured summary data for leave-one-ion-out, model misspecification, budget allocation, and GPR calibration analyses. |
| `results.json` | Complete numerical benchmark output, out-of-fold predictions, noise ceilings, and degradation recovery metrics. |

### Manuscript figures (`figures/`)

| File | Description |
| --- | --- |
| `Figure1_benchmark.*` | Ground-truth response surfaces, synthetic dataset distribution, and grouped cross-validation model comparisons. |
| `Figure2_validation.*` | Demonstration of replicate data leakage and model sensitivity across varying noise levels and formulation sample sizes. |
| `Figure3_data_requirements.*` | Trade-offs in experimental budget allocation (formulations vs. replicates) and model scaling up to N = 480. |
| `Figure4_interpretation.*` | Feature attribution rankings of causal versus non-causal descriptors and linear discriminant ion classification. |
| `Figure5_degradation.*` | Kinetic degradation curve fitting, inflection-point parameter recovery, and false-positive rate evaluation under a null scenario. |

### Manuscript

| Path | Description |
| --- | --- |
| `manuscript/Synthetic_Benchmark_Manuscript.docx` | Full manuscript draft including introduction, methodology, benchmark results, and reproducibility discussion. |

## Installation

Requirements: Python 3.10 or later.

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

Alternatively, create an isolated Conda environment:
```bash
conda env create -f environment.yml
conda activate synthbench
```

## Reproduction workflow

Execute scripts in the following order:

```bash
python3 analysis.py
python3 null_test.py
python3 extra_analysis.py
python3 figures.py
```

### Runtime estimates
- `analysis.py`: ~20 minutes on 2 CPU cores (skips parameter sweep if `results/sweep.csv` exists; delete to recompute).
- `null_test.py`: ~3 minutes on 2 CPU cores.
- `extra_analysis.py`: ~16 minutes on 2 CPU cores.
- `figures.py`: ~1 minute on 2 CPU cores.

All pseudo-random number generator seeds are fixed for exact reproducibility.

## Citation

If you use this benchmark or code, cite the accompanying paper:

```bibtex
@article{synthbench2026,
  title   = {When does machine learning recover the physics? A synthetic-benchmark tutorial for structure--property modelling of metal-crosslinked carrageenan hydrogels},
  author  = {[Author names]},
  journal = {Digital Discovery},
  year    = {2026},
  note    = {In review}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
