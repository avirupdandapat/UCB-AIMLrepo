# Indian Two-Wheeler Market Assessment — Capstone 20.1

Initial report and exploratory data analysis (EDA) for the capstone research question:

> Which factors most significantly predict two-wheeler sales volume and resale value across Indian states, and can a machine learning model reliably forecast regional demand by brand and model segment?

## Data source

This version of the codebase runs against the **user-provided dataset** `bike_sales_india.csv` — 10,000 two-wheeler sale records across 10 states, 8 brands, and 40 models, with columns: State, Avg Daily Distance, Brand, Model, Price, Year of Manufacture, Engine Capacity, Fuel Type, Mileage, Owner Type, Registration Year, Insurance Status, Seller Type, Resale Price, and City Tier. The file arrived with no missing values and no duplicate rows; the notebook's cleaning steps (deduplication, imputation, outlier capping, consistency checks) are still applied in full as a defensive, reusable pipeline that will handle messier data if the file is refreshed later.

There is no explicit body-style/segment column in the raw data, so `segment` (Commuter / Sports / Cruiser / Superbike / Electric) is **engineered** from engine capacity and fuel type — see `build_notebook_real.py` / Section 3 of the notebook for the exact rule.

> An earlier iteration of this repo (see `generate_data.py`, `build_notebook.py`, `build_report.js`) used a **synthetic** dataset because the original Kaggle source wasn't reachable from the build environment. Those files are kept for reference/reproducibility but are no longer the active pipeline — `build_notebook_real.py` and `build_report_real.js` (below) are what produced the current `Indian_Two_Wheeler_EDA.ipynb` and `Indian_Two_Wheeler_Capstone_Report.docx`.

## Repository contents

| File | Description |
|---|---|
| `bike_sales_india.csv` | The real, user-provided dataset — 10,000 records, 15 columns. Source of truth for the current analysis. |
| `build_notebook_real.py` | Generates `Indian_Two_Wheeler_EDA.ipynb` programmatically (via `nbformat`) against `bike_sales_india.csv` — cleaning, feature engineering, EDA charts, regression, classification, and clustering cells. Run with `python build_notebook_real.py`, then execute the notebook (see below). |
| `Indian_Two_Wheeler_EDA.ipynb` | The deliverable notebook, pre-executed with all outputs/plots baked in, using the real dataset. |
| `build_report_real.js` | Generates `Indian_Two_Wheeler_Capstone_Report.docx` (via `docx`/docx-js) — the written summary report with embedded charts and result tables, reflecting the real-data results. Run with `node build_report_real.js`. |
| `Indian_Two_Wheeler_Capstone_Report.docx` | The deliverable Word report: findings summary, methodology, visuals, and model comparison tables. |
| `generate_data.py`, `build_notebook.py`, `build_report.js`, `indian_bike_sales_raw.csv` | Legacy synthetic-data pipeline, kept for reference — not used to produce the current deliverables. |

## Analysis pipeline (notebook)

1. **Setup & Data Loading**
2. **Data Cleaning** — text standardization, duplicate removal, missing-value imputation (median/mode), IQR-based outlier capping
3. **Feature Engineering** — `vehicle_age_years`, `resale_ratio`, `usage_band`, `price_band`, `is_electric`, `high_demand` label
4. **EDA** — regional sales distribution, brand market share, price clustering by segment, resale-value-vs-age, fuel/insurance breakdowns, correlation heatmap, average resale value by state
5. **Regression — resale value**: Linear Regression baseline vs. Bagging, Random Forest, and Gradient Boosting regressors (RMSE / MAE / R²)
6. **Classification — high vs. low demand**: Logistic Regression baseline vs. Decision Tree, Random Forest, AdaBoost, and Gradient Boosting classifiers (Accuracy / F1 / ROC-AUC)
7. **Clustering** — K-Means grouping of states by buyer-behavior profile
8. **Feature importance & key takeaways**

Note on the classification target: `high_demand` is derived from a groupby on `(state, brand, segment)`, so those three columns are deliberately **excluded** from the classifier's features to avoid label leakage (including them lets a model memorize the label instead of learning a real relationship).

## Reproducing from scratch

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nbformat nbclient ipykernel
npm install docx   # only needed if regenerating the .docx report

# bike_sales_india.csv must already be present in the project root
python build_notebook_real.py                # -> Indian_Two_Wheeler_EDA.ipynb (unexecuted)

python -m ipykernel install --user --name capstone-kernel
python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('Indian_Two_Wheeler_EDA.ipynb', as_version=4)
NotebookClient(nb, timeout=600, kernel_name='capstone-kernel').execute()
nbformat.write(nb, 'Indian_Two_Wheeler_EDA.ipynb')
"

node build_report_real.js                     # -> Indian_Two_Wheeler_Capstone_Report.docx
```

## Headline results (real data)

- **Resale price regression:** all four models land close together — Linear Regression baseline (R² = 0.809, RMSE ≈ ₹28,929) is essentially tied with Gradient Boosting (R² = 0.809, RMSE ≈ ₹28,974), with Random Forest and Bagging slightly behind. This near-tie indicates resale price in this dataset is close to a linear function of price and vehicle age, with limited extra non-linear signal for tree ensembles to exploit.
- **Demand classification:** Gradient Boosting best (F1 = 0.810, ROC-AUC = 0.861) vs. Logistic Regression baseline (F1 = 0.788, ROC-AUC = 0.849) — a modest but consistent ensemble edge, suggesting more non-linear structure in demand-tier prediction than in resale pricing.
- **Clustering:** three distinguishable state buyer-behavior profiles emerge based on price, EV share, and usage intensity across the 10 states present.
- **Data quality:** the uploaded file had 0 missing values and 0 duplicate rows, and passed all logical consistency checks (resale price ≤ price; registration year ≥ manufacture year).
