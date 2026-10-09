# Forecasting Regional Two-Wheeler Demand in India with Neural Networks

**Technical notebook:** [bike_demand_forecasting.ipynb](bike_demand_forecasting.ipynb)

## Problem statement

Two-wheeler manufacturers and dealers have to decide months in advance how many bikes of each kind to send to each state. If they send too many, money sits idle in dealer stock and bikes get discounted. If they send too few, buyers go to a competitor.

This project builds a machine-learning model that forecasts **next year's demand for each region (state), brand and model segment**, for example *"How many Royal Enfield cruisers will be registered in Punjab next year?"*. The forecasting models are neural networks built in **Keras (TensorFlow)** and **PyTorch**.

## Data

`bike_sales_india.csv` has **10,000 two-wheeler records** covering:

* 10 states, 8 brands and 40 models
* registration years 2015–2024
* price, resale price, engine capacity, fuel type, mileage, owner type, city tier and seller type

**How demand is measured.** The data has no sale dates, so the number of bikes **registered** each year stands in for demand. Each of the 40 models is placed in one of five market segments: **Commuter, Scooter, Cruiser, Performance and Adventure/Touring**. That gives **220 State × Brand × Segment combinations**, and each one has ten years of demand history.

**Data cleaning.** The data had no missing values, no duplicate rows and no impossible records, such as a bike registered before it was built. However, engine capacity varies from 100 cc to 1,000 cc *within a single model*, so it cannot be used to define segments. Segments come from how each manufacturer positions the model in India instead.

## Approach

1. **Exploratory analysis.** Charts of demand over time, by state, by brand and by segment.
2. **Statistical tests.** A trend test on yearly growth, plus chi-square tests of whether brand and segment preferences differ between states, city tiers and years.
3. **Feature engineering.** For each combination and year, the model sees only information that was available *before* that year: past demand, the combination's long-run market share, and national, state, brand and segment trends.
4. **Models compared.**
   * Rules of thumb: "same as last year", 3-year average, and "market share × national trend"
   * Classical machine learning: Ridge regression, Poisson regression and gradient boosting
   * **Keras neural network** and **PyTorch neural network**. Both learn embeddings for state, brand and segment and use a loss function designed for count data. A Keras + PyTorch ensemble averages the two.
5. **Validation.** Every model is tuned with a **grid search** using **walk-forward cross-validation**: train on the past, predict the next year, and repeat for 2021, 2022 and 2023. **2024 is held back** as a final test that no model sees during training or tuning.

### Evaluation metric

The main metric is **Mean Absolute Error (MAE): on average, how many bikes the forecast is off by for one state-brand-segment combination.**

* It is measured in bikes, so planners can read it directly.
* Overstocking and understocking cost roughly the same per bike, so every bike of error should count equally.
* Percentage errors like MAPE don't work here because some combinations sell zero bikes in a year.

For a business-friendly percentage, the report also shows **WAPE** (total absolute error ÷ total actual demand). The two metrics always rank models the same way.

Small numbers are naturally random, so even a perfect forecaster would miss by about **2.5 bikes per combination** in 2024. That is the "noise floor" the models are compared against.

## Results

### Accuracy on the 2024 hold-out year

| Model | MAE (bikes per combination) | WAPE |
|---|---|---|
| **PyTorch neural network** | **2.68** | **25.7 %** |
| Poisson regression | 2.70 | 25.9 % |
| **Neural ensemble (Keras + PyTorch)**, the production model | **2.72** | **26.1 %** |
| Keras neural network | 2.96 | 28.3 % |
| Gradient boosting | 3.11 | 29.8 % |
| Market share × national trend (rule of thumb) | 3.43 | 32.8 % |
| Ridge regression | 3.89 | 37.2 % |
| Same as last year (naïve) | 4.23 | 40.5 % |
| 3-year moving average | 4.85 | 46.4 % |
| *Noise floor (best possible)* | *≈ 2.47* | |

### Accuracy at different planning levels (neural ensemble, 2024)

| Planning level | Forecast error (WAPE) | "Same as last year" error |
|---|---|---|
| Single state × brand × segment | 26 % | 41 % |
| State × brand | 19 % | 33 % |
| Brand × segment | 13 % | 29 % |
| State total / brand total / segment total | 10–11 % | 29 % |
| National total | 10 % (under-forecast) | 29 % |

## Key findings

1. **Demand is growing fast.** Registrations rose from 438 in 2015 to 2,296 in 2024, about **19 % a year** (statistically significant). Growth accelerated to **+41 % in 2024**.
2. **Regional preferences are stable.** No state, city tier or year showed a statistically different preference for any brand or segment. The variation in the charts is random noise. Each state-brand-segment combination keeps a roughly constant share of a growing national market.
3. **The neural networks forecast well.** The PyTorch network and the Keras + PyTorch ensemble cut forecast error by **about 35 %** compared with planning from last year's sales. They land within 0.2–0.3 bikes of the best possible accuracy.
4. **A simpler statistical model does almost as well.** Poisson regression matched the neural networks because the pattern in this data is simple (stable shares in a growing market). The neural networks are still useful because they can take in richer inputs, such as prices, launches and monthly sales, without being redesigned.
5. **Forecasts are most reliable for state, brand or segment totals (about 10 % error).** Single combinations are small and noisy, so they carry about 26 % error.
6. **The biggest source of error is the national growth rate.** 2024's +41 % growth was double the long-run trend, so the model under-forecast the national total by about 10 %. That shortfall makes up most of the error at the state, brand and segment levels.
7. **2025 outlook:** the forecast is about **3,000 registrations (around +30 %)**, with growth in every segment. Performance and Commuter bikes stay the largest segments, and demand stays spread evenly across states and brands.

## Recommendations

* **Plan allocations top-down.** Use the neural forecast for state-level and brand-level volumes, where it is most accurate. Then split each state's volume into segments using that state's stable segment mix.
* **Hold a small safety buffer for each combination** of about ±3 bikes. This covers the random variation that no model can predict.
* **Treat the national growth rate as a management lever.** Review the model's growth assumption each quarter against what the business knows about festive demand, EV incentives and new model launches. This assumption drives most of the forecast error.
* **Stop planning from last year's actuals.** That approach was about 29 % off nationally in 2024 because it ignores growth.

## Next steps

* **Collect dated (monthly) sales data** so the model can capture seasonality, such as the festive season and monsoon, and support re-planning during the year.
* **Add external drivers of growth:** fuel prices, EV subsidies, interest rates, rural income and new model launches. These would help predict growth surges like 2024's.
* **Fix product data quality.** Engine capacity, fuel type and price don't match the model names in this extract.
* **Retrain the model regularly and track its error** against the noise floor, with an alert if it drifts.
* **Try sequence models** such as LSTMs once longer, higher-frequency histories are available.

## Repository structure

```
├── README.md                          <- this non-technical summary
├── bike_demand_forecasting.ipynb      <- full analysis: cleaning, EDA, statistics, models, forecast
└── data
    └── bike_sales_india.csv           <- source data
```

## How to run

1. Put `bike_sales_india.csv` in a `data/` folder next to the notebook.
2. Install the requirements: `pip install pandas numpy scipy matplotlib seaborn scikit-learn tensorflow torch`
3. Run the notebook from top to bottom. It takes about 8–10 minutes on a laptop CPU, mostly for the neural-network grid searches.

Built with Python 3.13, pandas 3.0, scikit-learn 1.9, TensorFlow/Keras 2.21/3.15 and PyTorch 2.14.
