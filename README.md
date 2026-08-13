# Appliance Energy Forecasting

A reproducible time-series forecasting project for modelling and forecasting household appliance energy consumption.

## Project Aim

The aim of this project is to forecast short-term household appliance energy use and evaluate whether increasingly complex forecasting models improve on simple benchmark methods.

The project compares:

* Simple benchmark forecasting methods
* SARIMAX
* A feature-based machine-learning model
* A time-series foundation model

The main research questions are:

1. How well do simple benchmark models forecast appliance energy use?
2. Does SARIMAX improve on the benchmark forecasts?
3. Do sensor, weather, and time-based covariates improve forecast accuracy?
4. Does a feature-based machine-learning model such as XGBoost improve performance?
5. Does a time-series foundation model provide additional benefit?
6. Which model would be most suitable for a practical smart-home energy forecasting system?

## Dataset

This project uses the **Appliances Energy Prediction** dataset.

The target variable is:

```text
Appliances
```

The dataset contains household appliance energy consumption, indoor temperature and humidity measurements, outdoor weather variables, and timestamps.

The original dataset is sampled every 10 minutes.

Important variables include:

```text
date
Appliances
lights
T1, RH_1
T2, RH_2
T3, RH_3
T4, RH_4
T5, RH_5
T6, RH_6
T7, RH_7
T8, RH_8
T9, RH_9
T_out
Press_mm_hg
RH_out
Windspeed
Visibility
Tdewpoint
```

The `T` variables represent indoor temperature measurements, while the `RH` variables represent indoor relative humidity measurements.

The outdoor variables describe weather conditions including temperature, pressure, humidity, wind speed, visibility, and dew point.

## Forecasting Task

The main task is:

> **Forecast appliance energy use over the next 24 hours.**

For the original 10-minute data:

```python
horizon = 24 * 6
horizon = 144
```

If the data is resampled to hourly averages:

```python
horizon = 24
```

For this project, hourly data may be used to make SARIMAX modelling and pipeline execution more manageable.

The recommended test period is the final 14 days of the dataset.

For hourly data:

```python
test_steps = 14 * 24
```

For 10-minute data:

```python
test_steps = 14 * 24 * 6
```

## Models

### 1. Benchmark Models

The project includes several simple benchmark forecasts:

* Mean forecast
* Naive forecast
* Daily seasonal naive forecast
* Weekly seasonal naive forecast
* Drift forecast

For hourly data:

```text
Daily seasonal naive: lag 24
Weekly seasonal naive: lag 168
```

These benchmarks provide a baseline against which the more advanced models are evaluated.

### 2. SARIMAX

A SARIMAX model is fitted to the appliance energy time series.

A starting configuration for hourly data is:

```python
order = (1, 0, 1)
seasonal_order = (1, 1, 1, 24)
```

Two approaches may be considered:

* Target-only SARIMA/SARIMAX
* SARIMAX with exogenous variables

Possible exogenous variables include:

```text
T_out
RH_out
Windspeed
Visibility
Tdewpoint
hour_sin
hour_cos
dow_sin
dow_cos
```

### 3. Feature-Based Machine Learning

A feature-based machine-learning model is used to predict appliance energy consumption.

Possible models include:

* XGBoost
* LightGBM
* Random Forest
* HistGradientBoostingRegressor

This project uses **XGBoost** as the primary feature-based model.

The feature table can include:

```text
Lagged appliance energy use
Rolling means
Rolling standard deviations
Hour-of-day features
Day-of-week features
Weekend indicator
Indoor temperature variables
Indoor humidity variables
Outdoor weather variables
```

All lagged and rolling features must use only past observations.

For example:

```python
data["lag_1"] = data["Appliances"].shift(1)
data["lag_24"] = data["Appliances"].shift(24)
data["lag_168"] = data["Appliances"].shift(168)

data["roll_mean_24"] = (
    data["Appliances"]
    .shift(1)
    .rolling(24)
    .mean()
)

data["roll_std_24"] = (
    data["Appliances"]
    .shift(1)
    .rolling(24)
    .std()
)
```

The `.shift(1)` is important because it prevents the current or future target value from being used when creating features.

### 4. Foundation Model

The project also evaluates a time-series foundation model.

Possible models include:

* Chronos
* TimesFM
* TimeGPT

The foundation model may be used as:

* A target-only forecasting model
* A covariate-informed forecasting model, if supported
* A zero-shot model
* A fine-tuned or adapted model, if appropriate

The implementation should clearly explain how the foundation model is being used and whether it has access to covariates.

## Feature Engineering

### Time-Based Features

Features created from the timestamp include:

```text
hour
dayofweek
is_weekend
hour_sin
hour_cos
dow_sin
dow_cos
```

Example:

```python
data["hour"] = data.index.hour
data["dayofweek"] = data.index.dayofweek
data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

data["hour_sin"] = np.sin(
    2 * np.pi * data["hour"] / 24
)

data["hour_cos"] = np.cos(
    2 * np.pi * data["hour"] / 24
)

data["dow_sin"] = np.sin(
    2 * np.pi * data["dayofweek"] / 7
)

data["dow_cos"] = np.cos(
    2 * np.pi * data["dayofweek"] / 7
)
```

### Lag and Rolling Features

For hourly data, useful target-based features include:

```text
lag_1
lag_24
lag_168
roll_mean_24
roll_std_24
```

These represent recent energy use, the same hour on the previous day, the same hour in the previous week, and recent rolling statistics.

## Repository Structure

```text
appliance-energy-forecasting/
│
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_download_and_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_benchmark_models.ipynb
│   ├── 04_sarimax_models.ipynb
│   ├── 05_feature_based_models.ipynb
│   ├── 06_foundation_model.ipynb
│   └── 07_model_comparison.ipynb
│
├── src/
│   └── appliance_energy/
│       ├── __init__.py
│       ├── config.py
│       ├── pipeline.py
│       ├── data.py
│       ├── features.py
│       ├── evaluation.py
│       ├── plotting.py
│       │
│       └── models/
│           ├── __init__.py
│           ├── benchmarks.py
│           ├── sarimax.py
│           ├── feature_models.py
│           └── foundation.py
│
├── scripts/
│   ├── download_data.py
│   ├── make_features.py
│   ├── run_pipeline.py
│   └── evaluate_models.py
│
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   ├── metrics/
│   └── model_objects/
│
├── reports/
│   ├── report.md
│   └── figures/
│
└── tests/
    ├── test_features.py
    ├── test_evaluation.py
    └── test_benchmarks.py
```

## Installation

Create a Python virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The core packages include:

```text
numpy
pandas
matplotlib
scikit-learn
statsmodels
xgboost
```

Additional packages may be required depending on the foundation model selected.

## Running the Pipeline

The main pipeline entry point is:

```bash
python scripts/run_pipeline.py
```

The pipeline should:

1. Load or download the dataset.
2. Clean and prepare the time series.
3. Create time, lag, rolling, sensor, and weather features.
4. Split the data into training and test sets.
5. Fit benchmark models.
6. Fit the SARIMAX model.
7. Fit the feature-based model.
8. Fit or call the foundation model.
9. Evaluate all forecasts.
10. Save forecasts, metrics, and plots.

## Outputs

### Forecasts

Forecasts are saved to:

```text
outputs/forecasts/all_forecasts.csv
```

The file should contain:

```text
actual
mean
naive
seasonal_naive_daily
seasonal_naive_weekly
drift
sarimax
feature_model
foundation_model
```

### Model Metrics

Model comparison results are saved to:

```text
outputs/metrics/model_comparison.csv
```

Required metrics:

```text
model
MAE
RMSE
MASE
Bias
```

### Figures

Figures are saved to:

```text
outputs/figures/
```

Suggested figures include:

```text
forecast_comparison.png
error_diagnostics.png
residual_acf.png
feature_importance.png
```

## Evaluation

All models are evaluated on the same test period.

The required evaluation metrics are:

### MAE

Mean Absolute Error measures the average absolute difference between forecasts and actual observations.

### RMSE

Root Mean Squared Error gives greater weight to larger forecasting errors.

### MASE

Mean Absolute Scaled Error compares forecasting performance against a naive benchmark.

### Bias

Bias measures the systematic tendency of the model to over- or under-predict.

Models should be compared against the strongest benchmark rather than only against each other.

## Data Leakage

Avoiding data leakage is essential.

Potential sources of leakage include:

* Using future values of `Appliances` in lag or rolling features
* Creating rolling features without shifting the target first
* Scaling the full dataset before the train-test split
* Using future sensor or weather values without discussing forecast realism
* Choosing the final model based only on test-set performance

For example, this is preferred:

```python
data["roll_mean_24"] = (
    data["Appliances"]
    .shift(1)
    .rolling(24)
    .mean()
)
```

Future time-of-day and day-of-week information is known in advance.

Future indoor sensor and weather variables may not be known at the forecast origin. If realised future sensor or weather values from the test set are used, the results should be described as a **conditional forecast**.

## Reproducibility

The project should be reproducible from a fresh clone.

Good practice includes:

* Using clear function names
* Keeping reusable code in `src/`
* Keeping notebooks for exploration and explanation
* Keeping scripts small and focused
* Avoiding large raw data files in GitHub
* Setting random seeds where relevant
* Comparing advanced models against simple benchmarks
* Documenting modelling assumptions
* Explaining which covariates are available at the forecast origin

## Tests

The repository includes tests for important functions.

Tests should cover:

* Forecast lengths
* MASE calculation
* Lag feature leakage
* Missing target values

Run the tests using:

```bash
pytest
```

## Final Report

The final report is 8 pages.

Structure:

1. Introduction
2. Data and preprocessing
3. Exploratory analysis
4. Forecasting design
5. Benchmark models
6. SARIMAX model
7. Feature-based model
8. Foundation model
9. Results and error analysis
10. Discussion and limitations
11. Conclusion

The report should answer:

1. Which benchmark model is strongest, and what does this reveal about appliance energy use?
2. Does SARIMAX improve on the strongest benchmark?
3. Does the feature-based model improve when lag, rolling, time, sensor, and weather features are added?
4. Does the foundation model outperform the simpler models?
5. Which covariates would genuinely be known at the forecast origin?
6. Which model would you recommend for practical smart-home energy forecasting, and why?

The repository should run from a fresh clone using the instructions provided in this README.
