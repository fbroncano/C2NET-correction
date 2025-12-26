# C2-Net Correction for Chlorophyll-a in Small Reservoirs

This repository provides a reproducible pipeline to correct chlorophyll-a (Chl-a) estimates in small reservoirs using Sentinel-2 atmospheric processors (C2RCC, C2X, C2XC) and machine learning. The main workflow loads data, builds an optimized 10-feature set (including rescaled processor outputs), evaluates models with cross-validation, and generates the figures used in the manuscript.

---

## Installation

Requirements: Python 3.8+ on macOS/Linux/Windows.

Create a virtual environment and install dependencies:

```bash
cd c2net-correction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Data

Input file expected: `data/resultados.csv` with at least these columns:
- `Embalse` (reservoir name)
- `Medicion` (in-situ chlorophyll-a, mg/m³)
- `C2RCC` (processor output, mg/m³)
- `C2X` (processor output, mg/m³)
- `C2XC` (processor output, mg/m³)

The scripts handle basic validation and will skip records with missing values in the required columns.

---

## Quick Usage

Run the main analysis (cross-validation, LORO, figures):

```bash
python code/final_optimized_analysis.py
```

Analyze feature importance and compare feature sets:

```bash
python code/analyze_feature_importance.py
```

Verify performance by trophic state (mesotrophic, eutrophic, hypertrophic):

```bash
python code/verify_trophic_performance.py
```

Generate all manuscript figures:

```bash
python code/figure_generation_script.py
```

Generate the specific trophic performance figure:

```bash
python code/generate_fig_trophic_performance.py
```

Outputs are written to the `figures/` folder and summary metrics to text files alongside the scripts.

---

## Script Roles

- `code/final_optimized_analysis.py`: Main pipeline. Loads data, computes rescaling, builds the 10-feature set, trains and evaluates models (5/10-fold CV and Leave-One-Reservoir-Out), and generates the core figures and metrics.
- `code/analyze_feature_importance.py`: Compares feature engineering strategies and reports feature importance (ElasticNet, Random Forest, Ridge).
- `code/verify_trophic_performance.py`: Computes RMSE/MAE/R² per trophic state on a stratified test split.
- `code/figure_generation_script.py`: Generates all figures used in the manuscript.
- `code/generate_fig_trophic_performance.py`: Generates the dedicated trophic performance figure.

---

## Methodology (Brief)

1. **Processors and rescaling**: C2X and C2XC are rescaled to match C2RCC statistics (mean and standard deviation) computed from training data only. This mitigates extreme over/underestimation while preserving monotonic relationships.
2. **Optimized 10-feature set**: Base processors (C2RCC, C2X_r, C2XC_r), aggregated statistics (mean/median), interactions (C2X_r×C2RCC, C2XC_r×C2RCC), transformations (C2RCC², √C2RCC), and ratio (C2XC_r/C2RCC).
3. **Validation**: Stratified K-Fold cross-validation (5- and 10-fold) by trophic state and Leave-One-Reservoir-Out (LORO) to assess spatial transferability.

This setup provides a stable and interpretable correction of Chl-a for small, optically complex reservoirs and produces publication-ready visualizations.

---

## Notes

- All scripts use fixed random seeds for reproducibility where applicable.
- Figures are saved under `figures/` with publication-friendly settings.
- If needed, adjust file paths or environment activation commands for your OS.
