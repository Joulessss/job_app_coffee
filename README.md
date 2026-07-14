# Coffee Forecast Studio

Coffee Forecast Studio is an interactive forecasting product built on top of the coffee consumption dataset in `coffee_db.parquet`.

It combines:

- statistical forecasting with `ARIMA`
- machine learning forecasting with `XGBoost`
- sequence forecasting with `LSTM`
- optional `TimeGPT` evaluation when the Nixtla API is available
- a `Dash` web application for visualization and forecasting
- an `OpenAI` chat analyst that answers questions from the trained artifacts and live forecast outputs

## Product Scope

The product supports three main workflows:

1. Data understanding
   - historical coffee consumption by coffee type
   - country-level consumption views
   - interpretation of which coffee types are harder to forecast

2. Forecasting
   - compares trained model performance on train and test splits
   - loads cached trained models for fast startup
   - generates future forecasts from production-ready cached models

3. LLM analyst
   - answers questions about model behavior, coffee types, country concentration, and forecast risk
   - uses the saved metrics, diagnostics, and live forecast outputs as context

## Main Files

- `sol_coffee.py`
  - model training, evaluation, diagnostics, PDF plot generation, artifact export

- `coffee_service.py`
  - reusable backend service for the dashboard
  - loads cached artifacts and production model weights
  - exposes forecast, chart, and analyst helper methods

- `coffee_dash_app.py`
  - Dash application
  - overview, diagnostics, forecast lab, and analyst chat tabs

- `api/index.py`
  - Vercel Python entrypoint

- `vercel.json`
  - routes all requests to the Python app on Vercel

- `requirements.txt`
  - Python dependencies for deployment

- `coffee_db.parquet`
  - source dataset

- `artifacts/`
  - exported metrics, predictions, diagnostics, and cached trained model objects

- `plots/`
  - generated PDF plots

## How The Product Works

### 1. Training and evaluation

The training pipeline uses a shared train/test split across models:

- train window: historical data before the holdout cutoff
- test window: the last seasons in the dataset

The product evaluates:

- overall metrics by model
- metrics by coffee type
- train vs test comparison
- forecast difficulty diagnostics by coffee type

`XGBoost` and `LSTM` use time-series cross-validation to choose hyperparameters before the final test evaluation.

### 2. Cached startup

For the default dataset, the product does not retrain from zero every time the app starts.

It loads cached artifacts and saved model weights from:

- `artifacts/model_cache/`

This makes the app faster to start and keeps the dashboard behavior consistent with the last trained state.

### 3. Production forecasting

The dashboard forecast tab uses cached production models.

Live forecasting is enabled for:

- `ARIMA`
- `LSTM`

`XGBoost` remains available in diagnostics and evaluation outputs, but the current live inference path is disabled in the dashboard because of an environment-specific native runtime instability during cached inference.

### 4. LLM analyst

The analyst chat uses the OpenAI API and receives structured context from:

- overall model metrics
- metrics by coffee type
- diagnostics about forecast difficulty
- country concentration tables
- live forecast outputs from the dashboard

The analyst is intended to answer questions such as:

- Which coffee type is hardest to forecast and why?
- Which countries dominate a given coffee type?
- Which coffee types show more production instability?
- What does the forecast suggest from an economic or production perspective?

## Local Setup

### Python environment

Use Python 3.12 or a compatible version with the installed libraries.

### Environment variables

Create `.env` from `.env.example` and add only the keys you need:

```env
NIXTLA_API_KEY=
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
```

Important:

- `.env` must stay local
- `.env.example` is safe to commit

### Run the dashboard

```bash
python coffee_dash_app.py
```

### Run the training pipeline directly

```bash
python sol_coffee.py
```

## What The Dashboard Shows

### Overview

- historical coffee consumption by type
- world map of country consumption
- top countries by coffee type

This tab is focused on understanding the data, not the models.

### Diagnostics

- train vs test RMSE comparison
- test MAPE by coffee type
- actual vs predicted curves
- simpler forecast risk interpretation

This tab is focused on model performance and interpretability.

### Forecast Lab

- upload new data
- retrain backend state from the uploaded file
- generate future forecasts

### Analyst Chat

- business and technical Q&A grounded in the trained outputs

## Deployment Notes

For public deployment:

- do not upload `.env`
- set `OPENAI_API_KEY`, `OPENAI_MODEL`, and optionally `NIXTLA_API_KEY` in the hosting platform environment settings
- keep `.env.example` in the repository as the template

If you deploy on Vercel, configure the environment variables in the Vercel project settings instead of storing secrets in the repository.

### Vercel configuration included

This repository now includes:

- `api/index.py`
- `vercel.json`
- `requirements.txt`

These files are needed so Vercel knows how to run the Python app instead of returning `404 NOT_FOUND`.

## Recommended Repository Contents

Commit:

- source code
- dataset if allowed
- artifacts needed for cached startup
- plots if you want generated outputs versioned
- `.env.example`
- `README.md`

Do not commit:

- `.env`
- local caches
- Python bytecode

## Current Limitations

- `TimeGPT` requires external API access and cannot be validated in a network-restricted environment
- `XGBoost` live cached inference is currently excluded from the dashboard forecast controls due to a native runtime issue in this environment
- uploaded data retrains backend state at runtime, so startup caching is only guaranteed for the default local dataset
- this stack is heavy for Vercel serverless deployment because it includes `torch`, `xgboost`, and cached ML artifacts; if Vercel build or runtime limits become a blocker, a platform like Render or Railway is a better fit
