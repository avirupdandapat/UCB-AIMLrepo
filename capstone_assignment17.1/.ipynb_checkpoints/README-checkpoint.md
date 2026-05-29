# What Drives the Price of a Car?
### A CRISP-DM Analysis for Used Car Dealerships

**Author:** Avi | **Date:** April 2026

---

## Project Overview

This project analyses a dataset of ~426,000 used car listings to identify which vehicle attributes most strongly predict price. The goal is to equip used car dealers with data-driven guidance for inventory selection and pricing.

The analysis follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

1. Business Understanding → 2. Data Understanding → 3. Data Preparation → 4. Modeling → 5. Evaluation → 6. Findings

📓 **[View the full notebook → notebooks/used_car_analysis.ipynb](notebooks/used_car_analysis.ipynb)**

---

## Repository Structure

```
used_car_analysis/
├── README.md                          ← This file
├── notebooks/
│   └── used_car_analysis.ipynb        ← Full analysis notebook (50 cells, pre-executed)
└── data/
    ├── vehicles.csv                   ← Raw dataset (426,880 rows × 18 columns)
    └── *.png                          ← All generated plots
```

---

## Summary of Findings

### Model Performance

Five scikit-learn regression models were trained and evaluated. All modelling used `log(price)` as the target to reduce skew, with performance measured by RMSE and R².

| Model | CV RMSE | Test RMSE | Test R² |
|-------|---------|-----------|--------|
| **Random Forest** *(best)* | 0.4534 | **0.4464** | **0.7527** |
| Gradient Boosting | 0.4868 | 0.4822 | 0.7115 |
| Linear Regression | 0.5261 | 0.5383 | 0.6406 |
| Ridge (α = 0.1) | 0.5261 | 0.5383 | 0.6406 |
| Lasso (α = 0.001) | 0.5294 | 0.5449 | 0.6317 |

The **Random Forest** model explains **75.3% of price variance** on held-out data — outperforming all linear models by ~11 percentage points in R². This confirms that non-linear interactions between features (e.g. age × brand) matter.

### Top Price Drivers (Random Forest Feature Importances)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **Vehicle Age** | 47.8% |
| 2 | **Odometer (mileage)** | 11.1% |
| 3 | Fuel type: diesel | 4.5% |
| 4 | Cylinders: 4 cylinders | 3.7% |
| 5 | Cylinders: 8 cylinders | 2.5% |
| 6 | Drive: FWD | 2.0% |
| 7 | Drive: 4WD | 1.9% |
| 8 | Condition: fair | 1.5% |

### Key Insights for Dealers

1. **Newer, lower-mileage vehicles command the highest prices.** Age and odometer together account for nearly 60% of the model's predictive power. Focus acquisition on cars under 10 years old with under 80,000 miles.

2. **Condition directly affects value.** Moving a car from 'fair' to 'good' or 'excellent' generates measurable price uplift. Reconditioning investments often pay off.

3. **Trucks, SUVs, and 4WD vehicles carry price premiums.** These vehicle types consistently show the highest median listing prices.

4. **Diesel vehicles command a modest premium** and are under-represented in most inventories — a potential margin opportunity.

5. **Salvage and rebuilt titles carry large discounts** — typically 30–50% below clean-title equivalents. Price these carefully and be transparent with buyers.

6. **Brand matters.** RAM, GMC, BMW, and Mercedes-Benz command the highest median prices among high-volume brands. Toyota and Honda hold value reliably with age.

---

## Technical Details

- **Language:** Python 3.10
- **Key Libraries:** pandas, numpy, scikit-learn, seaborn, matplotlib
- **All models built with scikit-learn Pipelines** — including preprocessing (imputation, scaling, one-hot encoding), cross-validation (`cross_val_score`), and hyperparameter tuning (`GridSearchCV`)
- **Dataset:** 426,880 listings; cleaned to 372,094 rows; 15% working sample (55,814 rows) used for modelling

---

## Next Steps

- Integrate live auction/retail pricing data for market calibration
- Explore regional demand differences (state × vehicle type interactions)
- Deploy the Random Forest model as an internal pricing tool
- Retrain quarterly as market conditions shift
