# Bank Marketing Campaign — Classifier Comparison

## Project Summary

A Portuguese bank conducted telephone marketing campaigns to promote term deposit subscriptions. This project builds and compares four machine learning classifiers to predict which customers are most likely to subscribe, enabling the bank to prioritise its calling efforts and reduce wasted outreach.

**Dataset:** `bank.csv` — 4,521 customer contact records, 16 features, binary target (`y`: subscribed yes/no)

---
## Key Findings

### Model Performance — 5-Fold Cross-Validation ROC-AUC

| Rank | Model                | CV ROC-AUC (mean ± std) | Variance  |
|------|----------------------|--------------------------|-----------|
| 🥇 1 | **SVM (RBF)**        | **0.9018 ± 0.0037**      | Lowest    |
| 🥈 2 | **Decision Tree**    | 0.8876 ± 0.0078          | Highest   |
| 🥉 3 | **KNN**              | 0.8777 ± 0.0041          | Low       |
| 4    | **Logistic Regression** | 0.8755 ± 0.0044       | Low       |
| —    | Baseline (always No) | 0.5000 — —               | —         |

> **Recommended model: SVM (RBF)** — highest mean ROC-AUC (0.9018) with the tightest cross-validation spread (±0.0037), indicating both strong performance and consistency across folds. Decision Tree ranks second but shows higher variance (±0.0078), suggesting greater sensitivity to data splits.

### Top Business Insights

1. **Call duration** is the strongest predictor of subscription — longer calls correlate strongly with positive outcomes.
2. **Prior success** (`poutcome = success`) gives a 3–4× lift in subscription probability.
3. **Students and retirees** convert at above-average rates.
4. **Cellular contact** outperforms landline contact.
5. **Customers with housing loans** are less likely to subscribe.

### Actionable Recommendation

Score every prospect with the Logistic Regression model before each campaign and work the calling list from highest to lowest probability. This maximises conversions per agent-hour.

---

## Repository Structure

```
.
├── README.md                          ← This file
├── bank.csv                           ← Raw dataset (semicolon-delimited)
└── bank_marketing_analysis.ipynb      ← Full analysis notebook
```

## Notebook Contents

1. Business Understanding
2. Data Loading & Inspection
3. Exploratory Data Analysis (EDA)
4. Data Preparation (encoding, scaling, train/test split)
5. Baseline Model
6. Model Training with GridSearchCV (KNN, Logistic Regression, Decision Tree, SVM)
7. Model Comparison (ROC curves, confusion matrices, metric bar charts, cross-validation)
8. Findings & Actionable Insights
9. Next Steps & Recommendations

## Requirements

```
python >= 3.9
pandas, numpy, matplotlib, seaborn, scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
