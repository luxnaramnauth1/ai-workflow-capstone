# Capstone summary report

Fill in the numbers from your own runs. Nothing below has been run on the real AAVAIL data.

## 1. Business opportunity
Forecast the next 30 days of revenue to support planning (staffing, inventory, targets).

## 2. Data and EDA  (`python eda.py`)
* Date range, rows, countries: ...
* Data quality issues found (missing values, cancelled invoices, odd prices): ...
* Seasonality / trends seen in `reports/eda/monthly_revenue.png`: ...
* Country concentration (`revenue_by_country.png`): ...

## 3. Modeling approaches compared  (`python compare_models.py`)
| Model | CV RMSE | Holdout RMSE | vs baseline |
|---|---|---|---|
| baseline | | | |
| ridge | | | |
| random forest | | | |
| gradient boosting | | | |

Selected model and why: ...
Hyperparameters / transformations iterated on: ...

## 4. Deployment
Flask API (train, predict, logs, monitor), Docker image, unit tests (API, model, logging) run via `run-tests.py`.

## 5. Post-production analysis  (`python post_production_analysis.py`)
* Relationship between model error and revenue (business metric): ...
* Countries that miss the business target: ...
* Drift / novelty findings: ...

## 6. Limitations and next steps
* ...
