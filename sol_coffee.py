from __future__ import annotations

import itertools
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import torch
from torch import nn
from xgboost import XGBRegressor

try:
    from nixtla import NixtlaClient
except ModuleNotFoundError:
    NixtlaClient = None


RANDOM_SEED = 42
TEST_HORIZON = 5
CV_HORIZON = 3
CV_WINDOWS = 3

DATA_PATH = Path(__file__).with_name("coffee_db.parquet")
PLOTS_DIR = Path(__file__).with_name("plots")
ARTIFACTS_DIR = Path(__file__).with_name("artifacts")
ENV_PATH = Path(__file__).with_name(".env")

PLOTS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class ModelArtifacts:
    name: str
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    metrics_by_type: pd.DataFrame
    plot_path: Path
    best_params: dict[str, Any] | None = None
    cv_results: pd.DataFrame | None = None


def load_env_file(env_path: Path = ENV_PATH) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_and_prepare_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_parquet(data_path)
    data = data.rename(
        columns={
            "Country": "country",
            "Coffee type": "coffee_type",
            "Total_domestic_consumption": "total_domestic_consumption",
        }
    )

    year_cols = [c for c in data.columns if re.fullmatch(r"\d{4}/\d{2}", str(c))]

    coffee_long = data.melt(
        id_vars=["country", "coffee_type", "total_domestic_consumption"],
        value_vars=year_cols,
        var_name="season",
        value_name="consumption",
    )
    coffee_long["start_year"] = coffee_long["season"].str[:4].astype(int)
    coffee_long["consumption"] = pd.to_numeric(coffee_long["consumption"], errors="coerce")
    return coffee_long.dropna(subset=["consumption"])


def build_type_series(coffee_long: pd.DataFrame, coffee_types: list[str] | None = None) -> pd.DataFrame:
    if coffee_types is not None:
        coffee_long = coffee_long[coffee_long["coffee_type"].isin(coffee_types)].copy()

    return (
        coffee_long.groupby(["start_year", "season", "coffee_type"], as_index=False)["consumption"]
        .sum()
        .sort_values(["coffee_type", "start_year"])
        .reset_index(drop=True)
    )


def get_cutoff_year(type_year: pd.DataFrame, test_horizon: int = TEST_HORIZON) -> int:
    return int(type_year["start_year"].max()) - test_horizon + 1


def get_cv_cutoffs(cutoff_year: int, cv_horizon: int = CV_HORIZON, n_windows: int = CV_WINDOWS) -> list[int]:
    first_cutoff = cutoff_year - cv_horizon * n_windows
    return [first_cutoff + i * cv_horizon for i in range(n_windows)]


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    eps = 1e-8
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mape": np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100,
        "smape": np.mean(
            200 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
        ),
        "r2": r2_score(y_true, y_pred),
    }


def compile_metrics(model_name: str, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name in ["train", "test"]:
        frame = predictions[predictions["split"] == split_name]
        if frame.empty:
            continue
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                **regression_metrics(frame["actual"], frame["prediction"]),
            }
        )
    return pd.DataFrame(rows)


def compile_metrics_by_type(model_name: str, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name in ["train", "test"]:
        split_df = predictions[predictions["split"] == split_name]
        if split_df.empty:
            continue
        for coffee_type, frame in split_df.groupby("coffee_type"):
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "coffee_type": coffee_type,
                    **regression_metrics(frame["actual"], frame["prediction"]),
                }
            )
    return pd.DataFrame(rows)


def plot_predictions(predictions: pd.DataFrame, model_name: str) -> Path:
    coffee_types = predictions["coffee_type"].unique().tolist()
    plot_path = PLOTS_DIR / f"{model_name.lower()}_forecast.pdf"
    pdf = canvas.Canvas(str(plot_path), pagesize=landscape(letter))
    page_width, page_height = landscape(letter)

    left_margin = 45
    right_margin = 30
    top_margin = 40
    bottom_margin = 35
    panel_gap = 18
    title_band = 28
    usable_width = page_width - left_margin - right_margin
    panel_height = (page_height - top_margin - bottom_margin - title_band - panel_gap * (len(coffee_types) - 1)) / len(
        coffee_types
    )

    pdf.setTitle(f"{model_name} Forecast")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(left_margin, page_height - 25, f"{model_name} Forecast")

    for idx, coffee_type in enumerate(coffee_types):
        frame = predictions[predictions["coffee_type"] == coffee_type].sort_values("start_year").reset_index(drop=True)
        panel_bottom = page_height - top_margin - title_band - (idx + 1) * panel_height - idx * panel_gap
        panel_top = panel_bottom + panel_height
        plot_left = left_margin + 55
        plot_right = left_margin + usable_width - 10
        plot_bottom = panel_bottom + 25
        plot_top = panel_top - 20
        plot_width = plot_right - plot_left
        plot_height = plot_top - plot_bottom

        years = frame["start_year"].to_numpy(dtype=float)
        actual = frame["actual"].to_numpy(dtype=float)
        predicted = frame["prediction"].to_numpy(dtype=float)
        valid_pred = ~np.isnan(predicted)
        y_values = np.concatenate([actual, predicted[valid_pred]])
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        if y_min == y_max:
            y_min *= 0.95
            y_max *= 1.05 if y_max != 0 else 1.0
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding

        def x_pos(year: float) -> float:
            if years.max() == years.min():
                return plot_left + plot_width / 2
            return plot_left + ((year - years.min()) / (years.max() - years.min())) * plot_width

        def y_pos(value: float) -> float:
            return plot_bottom + ((value - y_min) / (y_max - y_min)) * plot_height

        pdf.setStrokeColor(colors.lightgrey)
        pdf.setLineWidth(0.6)
        for frac in np.linspace(0, 1, 5):
            y = plot_bottom + frac * plot_height
            pdf.line(plot_left, y, plot_right, y)
            tick_value = y_min + frac * (y_max - y_min)
            pdf.setFillColor(colors.black)
            pdf.setFont("Helvetica", 7)
            pdf.drawRightString(plot_left - 6, y - 2, f"{tick_value:,.0f}")

        pdf.setStrokeColor(colors.black)
        pdf.setLineWidth(1)
        pdf.rect(plot_left, plot_bottom, plot_width, plot_height, stroke=1, fill=0)

        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(left_margin, panel_top - 10, coffee_type)

        pdf.setFont("Helvetica", 7)
        for year in years.astype(int):
            pdf.drawCentredString(x_pos(year), plot_bottom - 12, str(year))

        pdf.setStrokeColor(colors.HexColor("#1f2937"))
        pdf.setLineWidth(1.5)
        for i in range(len(frame) - 1):
            pdf.line(x_pos(years[i]), y_pos(actual[i]), x_pos(years[i + 1]), y_pos(actual[i + 1]))
        for year, value in zip(years, actual):
            pdf.circle(x_pos(year), y_pos(value), 2.4, stroke=1, fill=1)

        pred_years = years[valid_pred]
        pred_values = predicted[valid_pred]
        pdf.setStrokeColor(colors.HexColor("#dc2626"))
        pdf.setDash(4, 2)
        pdf.setLineWidth(1.2)
        for i in range(len(pred_years) - 1):
            pdf.line(x_pos(pred_years[i]), y_pos(pred_values[i]), x_pos(pred_years[i + 1]), y_pos(pred_values[i + 1]))
        for year, value in zip(pred_years, pred_values):
            pdf.circle(x_pos(year), y_pos(value), 2.2, stroke=1, fill=0)
        pdf.setDash()

        test_years = frame.loc[frame["split"] == "test", "start_year"]
        if not test_years.empty:
            split_x = x_pos(float(test_years.min()))
            pdf.setStrokeColor(colors.HexColor("#2563eb"))
            pdf.setDash(1, 2)
            pdf.line(split_x, plot_bottom, split_x, plot_top)
            pdf.setDash()

        legend_y = panel_top - 10
        legend_x = plot_right - 180
        pdf.setFont("Helvetica", 8)
        pdf.setStrokeColor(colors.HexColor("#1f2937"))
        pdf.line(legend_x, legend_y, legend_x + 18, legend_y)
        pdf.drawString(legend_x + 24, legend_y - 3, "Actual")
        pdf.setStrokeColor(colors.HexColor("#dc2626"))
        pdf.setDash(4, 2)
        pdf.line(legend_x + 70, legend_y, legend_x + 88, legend_y)
        pdf.setDash()
        pdf.drawString(legend_x + 94, legend_y - 3, "Predicted")

    pdf.save()
    return plot_path


def build_supervised_dataset(type_year: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    rows = []
    for coffee_type, group in type_year.groupby("coffee_type"):
        group = group.sort_values("start_year").reset_index(drop=True)
        values = group["consumption"].to_numpy(dtype=float)
        years = group["start_year"].to_numpy(dtype=int)
        seasons = group["season"].to_numpy()

        for idx in range(n_lags, len(group)):
            history = values[idx - n_lags : idx]
            row = {
                "coffee_type": coffee_type,
                "season": seasons[idx],
                "start_year": years[idx],
                "target": values[idx],
                "rolling_mean": history.mean(),
                "rolling_std": history.std(ddof=0),
                "trend": values[idx - 1] - values[idx - n_lags],
            }
            for lag in range(1, n_lags + 1):
                row[f"lag_{lag}"] = values[idx - lag]
            rows.append(row)

    return pd.DataFrame(rows)


def build_lstm_dataset(type_year: pd.DataFrame, sequence_length: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    type_list = sorted(type_year["coffee_type"].unique())
    type_to_idx = {coffee_type: idx for idx, coffee_type in enumerate(type_list)}

    meta_rows = []
    sequences = []
    targets = []

    for coffee_type, group in type_year.groupby("coffee_type"):
        group = group.sort_values("start_year").reset_index(drop=True)
        values = group["consumption"].to_numpy(dtype=float)
        years = group["start_year"].to_numpy(dtype=int)
        seasons = group["season"].to_numpy()
        static_type = np.zeros(len(type_list), dtype=float)
        static_type[type_to_idx[coffee_type]] = 1.0

        for idx in range(sequence_length, len(group)):
            history = values[idx - sequence_length : idx].reshape(-1, 1)
            repeated_type = np.repeat(static_type.reshape(1, -1), sequence_length, axis=0)
            sequences.append(np.concatenate([history, repeated_type], axis=1))
            targets.append(values[idx])
            meta_rows.append(
                {
                    "coffee_type": coffee_type,
                    "season": seasons[idx],
                    "start_year": years[idx],
                }
            )

    return pd.DataFrame(meta_rows), np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def split_by_year(frame: pd.DataFrame, cutoff_year: int, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = frame[frame["start_year"] < cutoff_year].copy()
    test_df = frame[(frame["start_year"] >= cutoff_year) & (frame["start_year"] < cutoff_year + horizon)].copy()
    return train_df, test_df


def export_artifacts(artifacts: list[ModelArtifacts]) -> None:
    metrics = pd.concat([artifact.metrics for artifact in artifacts], ignore_index=True)
    metrics_by_type = pd.concat([artifact.metrics_by_type for artifact in artifacts], ignore_index=True)
    predictions = pd.concat([artifact.predictions for artifact in artifacts], ignore_index=True)
    metrics.to_csv(ARTIFACTS_DIR / "model_metrics.csv", index=False)
    metrics_by_type.to_csv(ARTIFACTS_DIR / "model_metrics_by_type.csv", index=False)
    predictions.to_csv(ARTIFACTS_DIR / "model_predictions.csv", index=False)

    selected_params = []
    for artifact in artifacts:
        if artifact.best_params is not None:
            selected_params.append({"model": artifact.name, **artifact.best_params})
        if artifact.cv_results is not None:
            artifact.cv_results.to_csv(ARTIFACTS_DIR / f"{artifact.name.lower()}_cv_results.csv", index=False)

    if selected_params:
        pd.DataFrame(selected_params).to_csv(ARTIFACTS_DIR / "selected_model_params.csv", index=False)


def print_summary(artifacts: list[ModelArtifacts]) -> None:
    metrics = pd.concat([artifact.metrics for artifact in artifacts], ignore_index=True)
    metrics_by_type = pd.concat([artifact.metrics_by_type for artifact in artifacts], ignore_index=True)
    print("\nModel metrics")
    print(metrics.round(4).to_string(index=False))

    print("\nModel metrics by coffee type")
    print(metrics_by_type.round(4).to_string(index=False))

    print("\nGenerated plots")
    for artifact in artifacts:
        print(f"{artifact.name}: {artifact.plot_path}")

    selected = []
    for artifact in artifacts:
        if artifact.best_params is not None:
            selected.append({"model": artifact.name, **artifact.best_params})
    if selected:
        print("\nSelected hyperparameters")
        print(pd.DataFrame(selected).to_string(index=False))


def export_interpretability_outputs(
    coffee_long: pd.DataFrame,
    type_year: pd.DataFrame,
    artifacts: list[ModelArtifacts],
) -> None:
    difficulty, diagnostics, top_countries = build_interpretability_report(coffee_long, type_year, artifacts)
    difficulty.to_csv(ARTIFACTS_DIR / "forecast_difficulty_by_type.csv", index=False)
    diagnostics.to_csv(ARTIFACTS_DIR / "series_diagnostics_by_type.csv", index=False)
    top_countries.to_csv(ARTIFACTS_DIR / "top_countries_by_coffee_type.csv", index=False)


def print_interpretability_summary(
    coffee_long: pd.DataFrame,
    type_year: pd.DataFrame,
    artifacts: list[ModelArtifacts],
) -> None:
    difficulty, diagnostics, top_countries = build_interpretability_report(coffee_long, type_year, artifacts)
    print("\nForecast difficulty by coffee type")
    print(
        difficulty[
            [
                "model",
                "coffee_type",
                "rmse",
                "mape",
                "coef_variation",
                "mean_abs_yoy_growth_pct",
                "lag1_autocorr",
                "sign_changes_in_trend",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nTop country influence by coffee type")
    print(
        top_countries[
            ["coffee_type", "country", "country_type_consumption", "coffee_type_share_in_country"]
        ]
        .round(4)
        .to_string(index=False)
    )


def run_arima_by_type(type_year: pd.DataFrame, cutoff_year: int, test_horizon: int = TEST_HORIZON) -> ModelArtifacts:
    predictions = []
    for coffee_type, group in type_year.groupby("coffee_type"):
        group = group.sort_values("start_year").reset_index(drop=True)
        train, test = split_by_year(group, cutoff_year, test_horizon)

        fitted = ARIMA(train["consumption"], order=(1, 1, 1)).fit()
        train_pred = np.asarray(fitted.predict(start=0, end=len(train) - 1))
        test_pred = np.asarray(fitted.forecast(steps=len(test)))

        predictions.append(
            train.rename(columns={"consumption": "actual"})[
                ["coffee_type", "season", "start_year", "actual"]
            ].assign(prediction=train_pred, model="ARIMA", split="train")
        )
        predictions.append(
            test.rename(columns={"consumption": "actual"})[
                ["coffee_type", "season", "start_year", "actual"]
            ].assign(prediction=test_pred, model="ARIMA", split="test")
        )

    pred_df = pd.concat(predictions, ignore_index=True)
    return ModelArtifacts(
        "ARIMA",
        pred_df,
        compile_metrics("ARIMA", pred_df),
        compile_metrics_by_type("ARIMA", pred_df),
        plot_predictions(pred_df, "ARIMA"),
    )


def xgb_pipeline(feature_cols: list[str], params: dict[str, Any]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["coffee_type"]),
            ("num", "passthrough", [c for c in feature_cols if c != "coffee_type"]),
        ]
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
        **params,
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def tune_xgboost(type_year: pd.DataFrame, cutoff_year: int) -> tuple[dict[str, Any], pd.DataFrame]:
    param_grid = {
        "n_lags": [3, 5],
        "n_estimators": [150, 300],
        "max_depth": [2, 4],
        "learning_rate": [0.03, 0.08],
        "subsample": [0.9],
        "colsample_bytree": [0.9],
        "min_child_weight": [1, 3],
    }

    records = []
    best_score = float("inf")
    best_params: dict[str, Any] | None = None

    keys = list(param_grid.keys())
    for values in itertools.product(*(param_grid[key] for key in keys)):
        params = dict(zip(keys, values))
        supervised = build_supervised_dataset(type_year, n_lags=params["n_lags"])
        feature_cols = ["coffee_type", "start_year"] + [f"lag_{i}" for i in range(1, params["n_lags"] + 1)] + [
            "rolling_mean",
            "rolling_std",
            "trend",
        ]
        fold_scores = []
        for fold_idx, fold_cutoff in enumerate(get_cv_cutoffs(cutoff_year), start=1):
            train_df, valid_df = split_by_year(supervised, fold_cutoff, CV_HORIZON)
            if train_df.empty or valid_df.empty:
                continue
            pipeline = xgb_pipeline(
                feature_cols,
                {k: v for k, v in params.items() if k != "n_lags"},
            )
            pipeline.fit(train_df[feature_cols], train_df["target"])
            valid_pred = pipeline.predict(valid_df[feature_cols])
            fold_rmse = regression_metrics(valid_df["target"], valid_pred)["rmse"]
            fold_scores.append(fold_rmse)
            records.append({**params, "fold": fold_idx, "fold_cutoff": fold_cutoff, "rmse": fold_rmse})

        mean_rmse = float(np.mean(fold_scores))
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params

    assert best_params is not None
    cv_results = pd.DataFrame(records)
    summary = (
        cv_results.groupby(keys, as_index=False)["rmse"]
        .mean()
        .rename(columns={"rmse": "cv_rmse"})
        .sort_values("cv_rmse")
        .reset_index(drop=True)
    )
    return best_params, summary


def run_xgboost(type_year: pd.DataFrame, cutoff_year: int) -> ModelArtifacts:
    best_params, cv_results = tune_xgboost(type_year, cutoff_year)
    n_lags = int(best_params["n_lags"])
    supervised = build_supervised_dataset(type_year, n_lags=n_lags)
    train_df, test_df = split_by_year(supervised, cutoff_year, TEST_HORIZON)
    feature_cols = ["coffee_type", "start_year"] + [f"lag_{i}" for i in range(1, n_lags + 1)] + [
        "rolling_mean",
        "rolling_std",
        "trend",
    ]
    pipeline = xgb_pipeline(feature_cols, {k: v for k, v in best_params.items() if k != "n_lags"})
    pipeline.fit(train_df[feature_cols], train_df["target"])

    train_pred = pipeline.predict(train_df[feature_cols])
    test_pred = pipeline.predict(test_df[feature_cols])
    predictions = pd.concat(
        [
            train_df.rename(columns={"target": "actual"})[
                ["coffee_type", "season", "start_year", "actual"]
            ].assign(prediction=train_pred, model="XGBoost", split="train"),
            test_df.rename(columns={"target": "actual"})[
                ["coffee_type", "season", "start_year", "actual"]
            ].assign(prediction=test_pred, model="XGBoost", split="test"),
        ],
        ignore_index=True,
    )

    return ModelArtifacts(
        "XGBoost",
        predictions,
        compile_metrics("XGBoost", predictions),
        compile_metrics_by_type("XGBoost", predictions),
        plot_predictions(predictions, "XGBoost"),
        best_params=best_params,
        cv_results=cv_results,
    )


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def fit_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_eval = X_eval.copy()
    X_train[:, :, 0] = scaler.fit_transform(X_train[:, :, 0].reshape(-1, 1)).reshape(X_train.shape[0], X_train.shape[1])
    X_eval[:, :, 0] = scaler.transform(X_eval[:, :, 0].reshape(-1, 1)).reshape(X_eval.shape[0], X_eval.shape[1])

    target_mean = y_train.mean()
    target_std = y_train.std() if y_train.std() > 0 else 1.0
    y_train_scaled = (y_train - target_mean) / target_std

    model = LSTMRegressor(
        input_size=X_train.shape[2],
        hidden_size=int(params["hidden_size"]),
        dropout=float(params["dropout"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))
    loss_fn = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    model.train()
    for _ in range(int(params["epochs"])):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_tensor), y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(torch.tensor(X_train, dtype=torch.float32)).cpu().numpy()
        eval_pred = model(torch.tensor(X_eval, dtype=torch.float32)).cpu().numpy()

    return train_pred * target_std + target_mean, eval_pred * target_std + target_mean


def tune_lstm(type_year: pd.DataFrame, cutoff_year: int) -> tuple[dict[str, Any], pd.DataFrame]:
    param_grid = {
        "sequence_length": [3, 5],
        "hidden_size": [16, 32],
        "learning_rate": [0.005, 0.01],
        "dropout": [0.0, 0.1],
        "epochs": [200],
    }

    records = []
    best_score = float("inf")
    best_params: dict[str, Any] | None = None
    keys = list(param_grid.keys())

    for values in itertools.product(*(param_grid[key] for key in keys)):
        params = dict(zip(keys, values))
        meta, sequences, targets = build_lstm_dataset(type_year, sequence_length=int(params["sequence_length"]))
        fold_scores = []

        for fold_idx, fold_cutoff in enumerate(get_cv_cutoffs(cutoff_year), start=1):
            train_mask = meta["start_year"] < fold_cutoff
            valid_mask = (meta["start_year"] >= fold_cutoff) & (meta["start_year"] < fold_cutoff + CV_HORIZON)
            if not train_mask.any() or not valid_mask.any():
                continue

            _, valid_pred = fit_lstm_model(
                sequences[train_mask.to_numpy()],
                targets[train_mask.to_numpy()],
                sequences[valid_mask.to_numpy()],
                params,
            )
            fold_rmse = regression_metrics(targets[valid_mask.to_numpy()], valid_pred)["rmse"]
            fold_scores.append(fold_rmse)
            records.append({**params, "fold": fold_idx, "fold_cutoff": fold_cutoff, "rmse": fold_rmse})

        mean_rmse = float(np.mean(fold_scores))
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params

    assert best_params is not None
    cv_results = pd.DataFrame(records)
    summary = (
        cv_results.groupby(keys, as_index=False)["rmse"]
        .mean()
        .rename(columns={"rmse": "cv_rmse"})
        .sort_values("cv_rmse")
        .reset_index(drop=True)
    )
    return best_params, summary


def run_lstm(type_year: pd.DataFrame, cutoff_year: int) -> ModelArtifacts:
    best_params, cv_results = tune_lstm(type_year, cutoff_year)
    meta, sequences, targets = build_lstm_dataset(type_year, sequence_length=int(best_params["sequence_length"]))

    train_mask = meta["start_year"] < cutoff_year
    test_mask = (meta["start_year"] >= cutoff_year) & (meta["start_year"] < cutoff_year + TEST_HORIZON)
    train_pred, test_pred = fit_lstm_model(
        sequences[train_mask.to_numpy()],
        targets[train_mask.to_numpy()],
        sequences[test_mask.to_numpy()],
        best_params,
    )

    predictions = pd.concat(
        [
            meta[train_mask].assign(
                actual=targets[train_mask.to_numpy()],
                prediction=train_pred,
                model="LSTM",
                split="train",
            ),
            meta[test_mask].assign(
                actual=targets[test_mask.to_numpy()],
                prediction=test_pred,
                model="LSTM",
                split="test",
            ),
        ],
        ignore_index=True,
    )

    return ModelArtifacts(
        "LSTM",
        predictions,
        compile_metrics("LSTM", predictions),
        compile_metrics_by_type("LSTM", predictions),
        plot_predictions(predictions, "LSTM"),
        best_params=best_params,
        cv_results=cv_results,
    )


def run_timegpt(type_year: pd.DataFrame, cutoff_year: int) -> ModelArtifacts | None:
    if NixtlaClient is None:
        print("TimeGPT skipped: `nixtla` is not installed in this environment.")
        return None

    api_key = os.getenv("NIXTLA_API_KEY")
    if not api_key:
        print("TimeGPT skipped: set the `NIXTLA_API_KEY` environment variable first.")
        return None

    client = NixtlaClient(api_key=api_key)
    nixtla_df = pd.DataFrame(
        {
            "unique_id": type_year["coffee_type"],
            "ds": pd.to_datetime(type_year["start_year"].astype(str) + "-01-01"),
            "y": type_year["consumption"].astype(float),
        }
    )
    train_nixtla = nixtla_df[nixtla_df["ds"] < pd.Timestamp(f"{cutoff_year}-01-01")].copy()
    train_years = type_year[type_year["start_year"] < cutoff_year][["coffee_type", "season", "start_year", "consumption"]].copy()
    test_years = type_year[(type_year["start_year"] >= cutoff_year) & (type_year["start_year"] < cutoff_year + TEST_HORIZON)][
        ["coffee_type", "season", "start_year", "consumption"]
    ].copy()

    try:
        cv_forecasts = client.cross_validation(
            df=train_nixtla,
            h=CV_HORIZON,
            freq="YS",
            n_windows=CV_WINDOWS,
            step_size=CV_HORIZON,
            model="timegpt-1",
            finetune_steps=10,
            finetune_depth=2,
        )
        forecast = client.forecast(
            df=train_nixtla,
            h=TEST_HORIZON,
            freq="YS",
            model="timegpt-1",
            finetune_steps=10,
            finetune_depth=2,
        )
    except Exception as exc:
        print(f"TimeGPT skipped: API request failed in this environment ({type(exc).__name__}: {exc}).")
        return None

    point_col = next(col for col in cv_forecasts.columns if col.startswith("TimeGPT"))
    cv_forecasts = cv_forecasts.rename(columns={"unique_id": "coffee_type", "y": "actual", point_col: "prediction"})
    cv_forecasts["start_year"] = pd.to_datetime(cv_forecasts["ds"]).dt.year
    cv_forecasts = cv_forecasts.merge(
        train_years.rename(columns={"consumption": "actual"}),
        on=["coffee_type", "start_year"],
        how="left",
        suffixes=("", "_from_source"),
    )
    cv_forecasts["actual"] = cv_forecasts["actual_from_source"].fillna(cv_forecasts["actual"])
    cv_forecasts = cv_forecasts[["coffee_type", "season", "start_year", "actual", "prediction"]].assign(
        model="TimeGPT",
        split="train",
    )

    point_col_test = next(col for col in forecast.columns if col.startswith("TimeGPT"))
    forecast = forecast.rename(columns={"unique_id": "coffee_type", point_col_test: "prediction"})
    forecast["start_year"] = pd.to_datetime(forecast["ds"]).dt.year
    forecast = forecast.merge(
        test_years.rename(columns={"consumption": "actual"}),
        on=["coffee_type", "start_year"],
        how="left",
    )
    forecast = forecast[["coffee_type", "season", "start_year", "actual", "prediction"]].assign(
        model="TimeGPT",
        split="test",
    )

    predictions = pd.concat([cv_forecasts, forecast], ignore_index=True)
    return ModelArtifacts(
        "TimeGPT",
        predictions,
        compile_metrics("TimeGPT", predictions),
        compile_metrics_by_type("TimeGPT", predictions),
        plot_predictions(predictions, "TimeGPT"),
    )


def build_series_diagnostics(type_year: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for coffee_type, frame in type_year.groupby("coffee_type"):
        frame = frame.sort_values("start_year").reset_index(drop=True)
        values = frame["consumption"].to_numpy(dtype=float)
        diffs = np.diff(values)
        growth = np.diff(values) / np.maximum(values[:-1], 1e-8)
        lag1_corr = np.corrcoef(values[:-1], values[1:])[0, 1] if len(values) > 2 else np.nan
        rows.append(
            {
                "coffee_type": coffee_type,
                "mean_consumption": float(values.mean()),
                "std_consumption": float(values.std(ddof=0)),
                "coef_variation": float(values.std(ddof=0) / np.maximum(values.mean(), 1e-8)),
                "mean_abs_yoy_change": float(np.mean(np.abs(diffs))) if len(diffs) else 0.0,
                "mean_abs_yoy_growth_pct": float(np.mean(np.abs(growth)) * 100) if len(growth) else 0.0,
                "lag1_autocorr": float(lag1_corr) if not np.isnan(lag1_corr) else np.nan,
                "sign_changes_in_trend": int(np.sum(np.sign(diffs[1:]) != np.sign(diffs[:-1]))) if len(diffs) > 1 else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("coef_variation", ascending=False).reset_index(drop=True)


def build_country_type_influence(coffee_long: pd.DataFrame) -> pd.DataFrame:
    country_type = (
        coffee_long.groupby(["country", "coffee_type"], as_index=False)["consumption"]
        .sum()
        .rename(columns={"consumption": "country_type_consumption"})
    )
    country_total = (
        coffee_long.groupby("country", as_index=False)["consumption"]
        .sum()
        .rename(columns={"consumption": "country_total_consumption"})
    )
    result = country_type.merge(country_total, on="country", how="left")
    result["coffee_type_share_in_country"] = (
        result["country_type_consumption"] / np.maximum(result["country_total_consumption"], 1e-8)
    )
    return result.sort_values(["coffee_type", "country_type_consumption"], ascending=[True, False]).reset_index(drop=True)


def build_interpretability_report(
    coffee_long: pd.DataFrame,
    type_year: pd.DataFrame,
    artifacts: list[ModelArtifacts],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_by_type = pd.concat([artifact.metrics_by_type for artifact in artifacts], ignore_index=True)
    test_metrics = metrics_by_type[metrics_by_type["split"] == "test"].copy()
    diagnostics = build_series_diagnostics(type_year)
    difficulty = (
        test_metrics.merge(diagnostics, on="coffee_type", how="left")
        .sort_values(["rmse", "mape"], ascending=False)
        .reset_index(drop=True)
    )

    country_influence = build_country_type_influence(coffee_long)
    top_countries = (
        country_influence.groupby("coffee_type", group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )
    return difficulty, diagnostics, top_countries


def run_all_models(data_path: Path = DATA_PATH, coffee_types: list[str] | None = None) -> dict[str, ModelArtifacts]:
    load_env_file()
    coffee_long = load_and_prepare_data(data_path)
    type_year = build_type_series(coffee_long, coffee_types=coffee_types)
    cutoff_year = get_cutoff_year(type_year)

    results = {
        "ARIMA": run_arima_by_type(type_year, cutoff_year=cutoff_year),
        "XGBoost": run_xgboost(type_year, cutoff_year=cutoff_year),
        "LSTM": run_lstm(type_year, cutoff_year=cutoff_year),
    }

    timegpt_result = run_timegpt(type_year, cutoff_year=cutoff_year)
    if timegpt_result is not None:
        results["TimeGPT"] = timegpt_result

    artifacts = list(results.values())
    export_artifacts(artifacts)
    export_interpretability_outputs(coffee_long, type_year, artifacts)
    print_summary(artifacts)
    print_interpretability_summary(coffee_long, type_year, artifacts)
    return results


if __name__ == "__main__":
    run_all_models()
