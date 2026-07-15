from __future__ import annotations

import base64
import io
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from openai import OpenAI
from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from sol_coffee import (
    ARTIFACTS_DIR,
    DATA_PATH,
    LSTMRegressor,
    TEST_HORIZON,
    build_country_type_influence,
    build_interpretability_report,
    build_lstm_dataset,
    build_series_diagnostics,
    build_supervised_dataset,
    build_type_series,
    compile_metrics,
    compile_metrics_by_type,
    get_cutoff_year,
    load_and_prepare_data,
    load_env_file,
    regression_metrics,
    run_arima_by_type,
    run_lstm,
    run_timegpt,
    run_xgboost,
    split_by_year,
    tune_lstm,
    tune_xgboost,
    xgb_pipeline,
)


@dataclass
class LSTMBundle:
    model: LSTMRegressor
    scaler: StandardScaler
    target_mean: float
    target_std: float
    sequence_length: int
    type_list: list[str]
    best_params: dict[str, Any]


class ForecastingService:
    def __init__(self, data_path: Path = DATA_PATH, load_only: bool = False) -> None:
        load_env_file()
        self.data_path = Path(data_path)
        self.cache_dir = ARTIFACTS_DIR / "model_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.source_name = self.data_path.name
        self.coffee_long = pd.DataFrame()
        self.type_year = pd.DataFrame()
        self.cutoff_year = 0
        self.artifacts: dict[str, Any] = {}
        self.arima_models: dict[str, Any] = {}
        self.xgb_bundle: dict[str, Any] = {}
        self.lstm_bundle: LSTMBundle | None = None
        self.metrics = pd.DataFrame()
        self.metrics_by_type = pd.DataFrame()
        self.predictions = pd.DataFrame()
        self.difficulty = pd.DataFrame()
        self.diagnostics = pd.DataFrame()
        self.top_countries = pd.DataFrame()
        if load_only:
            self._init_from_cache()
        else:
            self.fit()

    # Preferred entry point for the serving layer (Dash app, API, etc.).
    # Loads pre-trained artefacts from disk without running any training.
    # Raises FileNotFoundError if the cache is missing so the error is obvious.
    @classmethod
    def from_cache(cls, data_path: Path = DATA_PATH) -> "ForecastingService":
        instance = cls.__new__(cls)
        instance.data_path = Path(data_path)
        instance.cache_dir = ARTIFACTS_DIR / "model_cache"
        instance.cache_dir.mkdir(exist_ok=True)
        instance.source_name = Path(data_path).name
        instance.coffee_long = pd.DataFrame()
        instance.type_year = pd.DataFrame()
        instance.cutoff_year = 0
        instance.artifacts = {}
        instance.arima_models = {}
        instance.xgb_bundle = {}
        instance.lstm_bundle = None
        instance.metrics = pd.DataFrame()
        instance.metrics_by_type = pd.DataFrame()
        instance.predictions = pd.DataFrame()
        instance.difficulty = pd.DataFrame()
        instance.diagnostics = pd.DataFrame()
        instance.top_countries = pd.DataFrame()
        load_env_file()
        instance._init_from_cache()
        return instance

    def _init_from_cache(self) -> None:
        if not self._cache_exists():
            raise FileNotFoundError(
                f"Model cache not found at '{self.cache_dir}'. "
                "Run sol_coffee.py (or ForecastingService().fit()) once to train and cache the models."
            )
        self.coffee_long = load_and_prepare_data(self.data_path)
        self.type_year = build_type_series(self.coffee_long)
        self.cutoff_year = get_cutoff_year(self.type_year)
        self._load_cached_models()

    def fit(self, raw_df: pd.DataFrame | None = None, source_name: str | None = None) -> None:
        is_default_dataset = raw_df is None

        if raw_df is None:
            self.coffee_long = load_and_prepare_data(self.data_path)
            self.source_name = self.data_path.name
        else:
            self.coffee_long = self._prepare_uploaded_dataframe(raw_df)
            self.source_name = source_name or "uploaded_dataset"

        self.type_year = build_type_series(self.coffee_long)
        self.cutoff_year = get_cutoff_year(self.type_year)

        if is_default_dataset and self._cache_exists():
            self._load_cached_models()
        else:
            self.artifacts = {
                "ARIMA": run_arima_by_type(self.type_year, cutoff_year=self.cutoff_year),
                "XGBoost": run_xgboost(self.type_year, cutoff_year=self.cutoff_year),
                "LSTM": run_lstm(self.type_year, cutoff_year=self.cutoff_year),
            }

            timegpt = run_timegpt(self.type_year, cutoff_year=self.cutoff_year)
            if timegpt is not None:
                self.artifacts["TimeGPT"] = timegpt

            self._fit_production_models()

        self.metrics = pd.concat([artifact.metrics for artifact in self.artifacts.values()], ignore_index=True)
        self.metrics_by_type = pd.concat(
            [artifact.metrics_by_type for artifact in self.artifacts.values()],
            ignore_index=True,
        )
        self.predictions = pd.concat([artifact.predictions for artifact in self.artifacts.values()], ignore_index=True)
        self.difficulty, self.diagnostics, self.top_countries = build_interpretability_report(
            self.coffee_long,
            self.type_year,
            list(self.artifacts.values()),
        )
        if is_default_dataset and not self._cache_exists():
            self._save_cached_models()

    def _prepare_uploaded_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        temp_path = ARTIFACTS_DIR / "_uploaded_temp.parquet"
        raw_df.to_parquet(temp_path, index=False)
        prepared = load_and_prepare_data(temp_path)
        temp_path.unlink(missing_ok=True)
        return prepared

    def _fit_production_models(self) -> None:
        self._fit_arima_models()
        self._fit_xgboost_model()
        self._fit_lstm_model()

    def _cache_exists(self) -> bool:
        required = [
            self.cache_dir / "service_bundle.pkl",
            self.cache_dir / "arima_models.pkl",
            self.cache_dir / "xgboost_preprocessor.joblib",
            self.cache_dir / "xgboost_model.json",
            self.cache_dir / "xgboost_bundle.pkl",
            self.cache_dir / "lstm_state.pt",
            self.cache_dir / "lstm_bundle.pkl",
        ]
        return all(path.exists() for path in required)

    def _save_cached_models(self) -> None:
        service_bundle = {
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "metrics_by_type": self.metrics_by_type,
            "predictions": self.predictions,
            "difficulty": self.difficulty,
            "diagnostics": self.diagnostics,
            "top_countries": self.top_countries,
            "source_name": self.source_name,
            "cutoff_year": self.cutoff_year,
            "coffee_long": self.coffee_long,
            "type_year": self.type_year,
        }
        with open(self.cache_dir / "service_bundle.pkl", "wb") as file:
            pickle.dump(service_bundle, file)
        with open(self.cache_dir / "arima_models.pkl", "wb") as file:
            pickle.dump(self.arima_models, file)
        pipeline = self.xgb_bundle["pipeline"]
        joblib.dump(pipeline.named_steps["prep"], self.cache_dir / "xgboost_preprocessor.joblib")
        pipeline.named_steps["model"].save_model(self.cache_dir / "xgboost_model.json")
        with open(self.cache_dir / "xgboost_bundle.pkl", "wb") as file:
            pickle.dump(
                {
                    k: v
                    for k, v in self.xgb_bundle.items()
                    if k != "pipeline"
                },
                file,
            )
        torch.save(self.lstm_bundle.model.state_dict(), self.cache_dir / "lstm_state.pt")
        with open(self.cache_dir / "lstm_bundle.pkl", "wb") as file:
            pickle.dump(
                {
                    "target_mean": self.lstm_bundle.target_mean,
                    "target_std": self.lstm_bundle.target_std,
                    "sequence_length": self.lstm_bundle.sequence_length,
                    "type_list": self.lstm_bundle.type_list,
                    "best_params": self.lstm_bundle.best_params,
                    "scaler": self.lstm_bundle.scaler,
                },
                file,
            )

    def _load_cached_models(self) -> None:
        with open(self.cache_dir / "service_bundle.pkl", "rb") as file:
            service_bundle = pickle.load(file)
        self.artifacts = service_bundle["artifacts"]
        self.metrics = service_bundle["metrics"]
        self.metrics_by_type = service_bundle["metrics_by_type"]
        self.predictions = service_bundle["predictions"]
        self.difficulty = service_bundle["difficulty"]
        self.diagnostics = service_bundle["diagnostics"]
        self.top_countries = service_bundle["top_countries"]
        self.source_name = service_bundle["source_name"]
        self.cutoff_year = service_bundle["cutoff_year"]
        self.coffee_long = service_bundle["coffee_long"]
        self.type_year = service_bundle["type_year"]
        self.arima_models = {}
        self.xgb_bundle = {}
        self.lstm_bundle = None

    def _ensure_arima_models(self) -> None:
        if self.arima_models:
            return
        with open(self.cache_dir / "arima_models.pkl", "rb") as file:
            self.arima_models = pickle.load(file)

    def _ensure_xgb_bundle(self) -> None:
        if self.xgb_bundle:
            return
        preprocessor = joblib.load(self.cache_dir / "xgboost_preprocessor.joblib")
        model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=1, tree_method="hist", verbosity=0)
        model.load_model(self.cache_dir / "xgboost_model.json")
        with open(self.cache_dir / "xgboost_bundle.pkl", "rb") as file:
            xgb_bundle = pickle.load(file)
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])
        self.xgb_bundle = {"pipeline": pipeline, **xgb_bundle}

    def _ensure_lstm_bundle(self) -> None:
        if self.lstm_bundle is not None:
            return
        with open(self.cache_dir / "lstm_bundle.pkl", "rb") as file:
            lstm_meta = pickle.load(file)
        model = LSTMRegressor(
            input_size=1 + len(lstm_meta["type_list"]),
            hidden_size=int(lstm_meta["best_params"]["hidden_size"]),
            dropout=float(lstm_meta["best_params"]["dropout"]),
        )
        model.load_state_dict(torch.load(self.cache_dir / "lstm_state.pt", map_location="cpu"))
        self.lstm_bundle = LSTMBundle(
            model=model.eval(),
            scaler=lstm_meta["scaler"],
            target_mean=lstm_meta["target_mean"],
            target_std=lstm_meta["target_std"],
            sequence_length=lstm_meta["sequence_length"],
            type_list=lstm_meta["type_list"],
            best_params=lstm_meta["best_params"],
        )

    def _fit_arima_models(self) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        self.arima_models = {}
        for coffee_type, frame in self.type_year.groupby("coffee_type"):
            frame = frame.sort_values("start_year")
            self.arima_models[coffee_type] = ARIMA(frame["consumption"], order=(1, 1, 1)).fit()

    def _fit_xgboost_model(self) -> None:
        best_params, _ = tune_xgboost(self.type_year, self.cutoff_year)
        n_lags = int(best_params["n_lags"])
        supervised = build_supervised_dataset(self.type_year, n_lags=n_lags)
        feature_cols = ["coffee_type", "start_year"] + [f"lag_{i}" for i in range(1, n_lags + 1)] + [
            "rolling_mean",
            "rolling_std",
            "trend",
        ]
        pipeline = xgb_pipeline(feature_cols, {k: v for k, v in best_params.items() if k != "n_lags"})
        pipeline.fit(supervised[feature_cols], supervised["target"])
        self.xgb_bundle = {
            "pipeline": pipeline,
            "n_lags": n_lags,
            "feature_cols": feature_cols,
            "best_params": best_params,
        }

    def _fit_lstm_model(self) -> None:
        best_params, _ = tune_lstm(self.type_year, self.cutoff_year)
        sequence_length = int(best_params["sequence_length"])
        meta, sequences, targets = build_lstm_dataset(self.type_year, sequence_length=sequence_length)
        train_mask = meta["start_year"] < self.cutoff_year
        X_train = sequences[train_mask.to_numpy()].copy()
        y_train = targets[train_mask.to_numpy()].copy()

        scaler = StandardScaler()
        X_train[:, :, 0] = scaler.fit_transform(X_train[:, :, 0].reshape(-1, 1)).reshape(X_train.shape[0], X_train.shape[1])
        target_mean = float(y_train.mean())
        target_std = float(y_train.std()) if float(y_train.std()) > 0 else 1.0
        y_train_scaled = (y_train - target_mean) / target_std

        model = LSTMRegressor(
            input_size=X_train.shape[2],
            hidden_size=int(best_params["hidden_size"]),
            dropout=float(best_params["dropout"]),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=float(best_params["learning_rate"]))
        loss_fn = torch.nn.MSELoss()
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

        model.train()
        for _ in range(int(best_params["epochs"])):
            optimizer.zero_grad()
            loss = loss_fn(model(X_train_tensor), y_train_tensor)
            loss.backward()
            optimizer.step()

        self.lstm_bundle = LSTMBundle(
            model=model.eval(),
            scaler=scaler,
            target_mean=target_mean,
            target_std=target_std,
            sequence_length=sequence_length,
            type_list=sorted(self.type_year["coffee_type"].unique()),
            best_params=best_params,
        )

    def forecast(self, model_name: str, horizon: int = 5) -> pd.DataFrame:
        model_name = model_name.upper()
        if model_name == "ARIMA":
            return self._forecast_arima(horizon)
        if model_name == "XGBOOST":
            return self._forecast_xgboost(horizon)
        if model_name == "LSTM":
            return self._forecast_lstm(horizon)
        raise ValueError(f"Unsupported model: {model_name}")

    def _forecast_arima(self, horizon: int) -> pd.DataFrame:
        self._ensure_arima_models()
        rows = []
        for coffee_type, fitted in self.arima_models.items():
            last_year = int(self.type_year[self.type_year["coffee_type"] == coffee_type]["start_year"].max())
            preds = fitted.forecast(steps=horizon)
            for step, pred in enumerate(np.asarray(preds), start=1):
                year = last_year + step
                rows.append(
                    {
                        "model": "ARIMA",
                        "coffee_type": coffee_type,
                        "start_year": year,
                        "season": self._season_from_year(year),
                        "prediction": float(pred),
                    }
                )
        return pd.DataFrame(rows)

    def _forecast_xgboost(self, horizon: int) -> pd.DataFrame:
        self._ensure_xgb_bundle()
        bundle = self.xgb_bundle
        pipeline: Pipeline = bundle["pipeline"]
        preprocessor = pipeline.named_steps["prep"]
        model = pipeline.named_steps["model"]
        n_lags = int(bundle["n_lags"])
        rows = []
        for coffee_type, frame in self.type_year.groupby("coffee_type"):
            history = frame.sort_values("start_year")["consumption"].tolist()
            last_year = int(frame["start_year"].max())
            for step in range(1, horizon + 1):
                current_year = last_year + step
                lag_values = history[-n_lags:]
                feature_row = {
                    "coffee_type": coffee_type,
                    "start_year": current_year,
                    "rolling_mean": float(np.mean(lag_values)),
                    "rolling_std": float(np.std(lag_values, ddof=0)),
                    "trend": float(lag_values[-1] - lag_values[0]),
                }
                for lag in range(1, n_lags + 1):
                    feature_row[f"lag_{lag}"] = float(lag_values[-lag])
                transformed = preprocessor.transform(pd.DataFrame([feature_row]))
                if hasattr(transformed, "toarray"):
                    transformed = transformed.toarray()
                pred = float(model.predict(np.asarray(transformed, dtype=np.float32))[0])
                history.append(pred)
                rows.append(
                    {
                        "model": "XGBoost",
                        "coffee_type": coffee_type,
                        "start_year": current_year,
                        "season": self._season_from_year(current_year),
                        "prediction": pred,
                    }
                )
        return pd.DataFrame(rows)

    def _forecast_lstm(self, horizon: int) -> pd.DataFrame:
        self._ensure_lstm_bundle()
        assert self.lstm_bundle is not None
        bundle = self.lstm_bundle
        type_to_idx = {coffee_type: idx for idx, coffee_type in enumerate(bundle.type_list)}
        rows = []

        for coffee_type, frame in self.type_year.groupby("coffee_type"):
            history = frame.sort_values("start_year")["consumption"].tolist()
            last_year = int(frame["start_year"].max())
            static_type = np.zeros(len(bundle.type_list), dtype=float)
            static_type[type_to_idx[coffee_type]] = 1.0

            for step in range(1, horizon + 1):
                sequence = np.asarray(history[-bundle.sequence_length :], dtype=float).reshape(-1, 1)
                scaled_sequence = bundle.scaler.transform(sequence)
                repeated_type = np.repeat(static_type.reshape(1, -1), bundle.sequence_length, axis=0)
                model_input = np.concatenate([scaled_sequence, repeated_type], axis=1).astype(np.float32)
                with torch.no_grad():
                    pred_scaled = bundle.model(torch.tensor(model_input[None, :, :], dtype=torch.float32)).item()
                pred = pred_scaled * bundle.target_std + bundle.target_mean
                current_year = last_year + step
                history.append(float(pred))
                rows.append(
                    {
                        "model": "LSTM",
                        "coffee_type": coffee_type,
                        "start_year": current_year,
                        "season": self._season_from_year(current_year),
                        "prediction": float(pred),
                    }
                )
        return pd.DataFrame(rows)

    def summary_cards(self) -> dict[str, Any]:
        test_metrics = self.metrics[self.metrics["split"] == "test"].sort_values("rmse")
        best_row = test_metrics.iloc[0]
        return {
            "best_model": best_row["model"],
            "best_test_rmse": float(best_row["rmse"]),
            "best_test_mape": float(best_row["mape"]),
            "coffee_types": int(self.type_year["coffee_type"].nunique()),
            "countries": int(self.coffee_long["country"].nunique()),
            "history_start": int(self.type_year["start_year"].min()),
            "history_end": int(self.type_year["start_year"].max()),
            "test_start": int(self.cutoff_year),
            "test_end": int(self.type_year["start_year"].max()),
            "source_name": self.source_name,
        }

    def historical_type_chart(self) -> go.Figure:
        fig = px.line(
            self.type_year,
            x="start_year",
            y="consumption",
            color="coffee_type",
            markers=True,
            title="Historical coffee consumption by type",
            template="plotly_white",
            labels={"start_year": "Year", "consumption": "Consumption", "coffee_type": "Coffee type"},
        )
        fig.update_yaxes(tickformat=".2s")
        fig.update_layout(legend_title_text="Coffee type")
        return fig

    def country_world_map(self) -> go.Figure:
        country_total = (
            self.coffee_long.groupby("country", as_index=False)["consumption"]
            .sum()
            .sort_values("consumption", ascending=False)
        )
        fig = px.choropleth(
            country_total,
            locations="country",
            locationmode="country names",
            color="consumption",
            color_continuous_scale="YlOrBr",
            title="Total coffee consumption by country",
            template="plotly_white",
            labels={"consumption": "Consumption", "country": "Country"},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), height=540)
        return fig

    def top_countries_chart(self) -> go.Figure:
        country_type = build_country_type_influence(self.coffee_long).groupby("coffee_type", group_keys=False).head(8)
        coffee_types = country_type["coffee_type"].unique().tolist()
        fig = make_subplots(rows=2, cols=2, subplot_titles=coffee_types, horizontal_spacing=0.12, vertical_spacing=0.16)
        colors_map = ["#6f4e37", "#b5651d", "#b22222", "#556b2f"]
        for idx, coffee_type in enumerate(coffee_types):
            row = idx // 2 + 1
            col = idx % 2 + 1
            frame = country_type[country_type["coffee_type"] == coffee_type].sort_values("country_type_consumption", ascending=True)
            fig.add_trace(
                go.Bar(
                    x=frame["country_type_consumption"],
                    y=frame["country"],
                    orientation="h",
                    marker_color=colors_map[idx % len(colors_map)],
                    showlegend=False,
                    hovertemplate="%{y}<br>Consumption=%{x:.2s}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(tickformat=".2s", row=row, col=col, title_text="Consumption")
        fig.update_layout(
            title="Top countries by coffee type",
            template="plotly_white",
            height=760,
            margin=dict(t=70, l=10, r=10, b=10),
        )
        return fig

    def metrics_heatmap(self) -> go.Figure:
        test_metrics = self.metrics_by_type[self.metrics_by_type["split"] == "test"].copy()
        pivot = test_metrics.pivot(index="coffee_type", columns="model", values="mape")
        # RdYlGn_r → low MAPE (good) = green, high MAPE (bad) = red
        fig = px.imshow(
            pivot,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            title="Test MAPE (%) — lower is better",
            labels={"coffee_type": "Variety", "model": "Model", "color": "MAPE (%)"},
        )
        fig.update_yaxes(title_text="Variety")
        fig.update_xaxes(title_text="Model")
        fig.update_layout(
            template="plotly_white",
            coloraxis_colorbar=dict(title="MAPE (%)", ticksuffix="%"),
        )
        return fig

    def prediction_vs_actual_chart(self, model_name: str, coffee_type: str | None = None) -> go.Figure:
        frame = self.predictions[self.predictions["model"] == model_name].copy()
        if coffee_type:
            frame = frame[frame["coffee_type"] == coffee_type].copy()

        # Separate train/test for visual distinction
        melted = frame.melt(
            id_vars=["coffee_type", "start_year", "split"],
            value_vars=["actual", "prediction"],
            var_name="series",
            value_name="Consumption",
        )
        # Label prediction lines by train/test split so they're distinguishable
        melted["trace"] = melted.apply(
            lambda r: "Actual" if r["series"] == "actual" else f"Predicted ({r['split']})",
            axis=1,
        )
        color_map = {
            "Actual": "#111827",
            "Predicted (train)": "#b45309",
            "Predicted (test)": "#dc2626",
        }
        fig = px.line(
            melted,
            x="start_year",
            y="Consumption",
            color="trace",
            facet_col=None if coffee_type else "coffee_type",
            facet_col_wrap=2 if not coffee_type else None,
            title=f"{model_name} — Actual vs Predicted",
            template="plotly_white",
            labels={"start_year": "Year", "Consumption": "Consumption (bags)", "trace": ""},
            color_discrete_map=color_map,
            markers=True,
        )
        fig.update_yaxes(tickformat=".3s", title_text="Consumption (bags)")
        # Only label x-axes that actually show tick labels (bottom row in a facet
        # grid) — labelling every facet's x-axis overlaps the row below's titles.
        fig.for_each_xaxis(lambda ax: ax.update(title_text="Year") if ax.showticklabels is not False else None)
        # Strip "coffee_type=Arabica" → "Arabica" from facet panel titles
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(
            height=480 if coffee_type else 780,
            legend_title_text="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def forecast_chart(self, model_name: str, horizon: int, coffee_type: str | None = None) -> go.Figure:
        future = self.forecast(model_name, horizon)
        history = self.type_year.rename(columns={"consumption": "value"})[
            ["coffee_type", "start_year", "season", "value"]
        ].assign(series="Historical")
        future_plot = future.rename(columns={"prediction": "value"})[
            ["coffee_type", "start_year", "season", "value"]
        ].assign(series=f"{model_name} Forecast")
        plot_df = pd.concat([history, future_plot], ignore_index=True)
        if coffee_type:
            plot_df = plot_df[plot_df["coffee_type"] == coffee_type].copy()
        fig = px.line(
            plot_df,
            x="start_year",
            y="value",
            color="series",
            facet_col=None if coffee_type else "coffee_type",
            facet_col_wrap=2 if not coffee_type else None,
            markers=True,
            title=f"{model_name} — {horizon}-year Forecast",
            template="plotly_white",
            labels={"start_year": "Year", "value": "Consumption (bags)", "series": "", "coffee_type": "Variety"},
            color_discrete_map={"Historical": "#374151", f"{model_name} Forecast": "#b45309"},
        )
        fig.update_yaxes(tickformat=".3s", title_text="Consumption (bags)")
        # Only label x-axes that actually show tick labels (bottom row in a facet
        # grid) — labelling every facet's x-axis overlaps the row below's titles.
        fig.for_each_xaxis(lambda ax: ax.update(title_text="Year") if ax.showticklabels is not False else None)
        # Strip "coffee_type=Arabica" → "Arabica" from facet panel titles
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(
            height=480 if coffee_type else 780,
            legend_title_text="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def train_test_rmse_chart(self) -> go.Figure:
        fig = px.bar(
            self.metrics,
            x="model",
            y="rmse",
            color="split",
            barmode="group",
            template="plotly_white",
            title="Train vs Test RMSE by Model",
            labels={"rmse": "RMSE (bags)", "model": "Model", "split": "Split"},
            color_discrete_map={"train": "#c08457", "test": "#5a341e"},
        )
        fig.update_yaxes(tickformat=".3s", title_text="RMSE (bags)")
        fig.update_xaxes(title_text="Model")
        fig.update_layout(legend_title_text="Split")
        return fig

    def test_mape_by_type_chart(self) -> go.Figure:
        frame = self.metrics_by_type[self.metrics_by_type["split"] == "test"].copy()
        model_colors = {
            "ARIMA": "#5a341e",
            "XGBoost": "#8a5a2b",
            "LSTM": "#b28040",
            "TimeGPT": "#2d6a4f",
        }
        fig = px.bar(
            frame,
            x="coffee_type",
            y="mape",
            color="model",
            barmode="group",
            template="plotly_white",
            title="Test MAPE (%) by Variety",
            labels={"coffee_type": "Variety", "mape": "MAPE (%)", "model": "Model"},
            color_discrete_map=model_colors,
        )
        fig.update_yaxes(ticksuffix="%", title_text="MAPE (%)")
        fig.update_xaxes(title_text="Variety")
        fig.update_layout(legend_title_text="Model")
        return fig

    def forecast_risk_chart(self) -> go.Figure:
        frame = self.diagnostics.copy()
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Variation level", "Average year-over-year growth swing"])
        fig.add_trace(
            go.Bar(x=frame["coffee_type"], y=frame["coef_variation"], marker_color="#8a5a2b", name="Coefficient of variation"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=frame["coffee_type"], y=frame["mean_abs_yoy_growth_pct"], marker_color="#b22222", name="Avg YoY growth change %"),
            row=1,
            col=2,
        )
        fig.update_layout(
            template="plotly_white",
            height=420,
            title="Why some coffee types are harder to forecast",
            showlegend=False,
        )
        return fig

    def forecast_risk_notes(self) -> list[str]:
        frame = self.diagnostics.sort_values(["coef_variation", "mean_abs_yoy_growth_pct"], ascending=False).reset_index(drop=True)
        hardest = frame.iloc[0]
        most_volatile = frame.sort_values("mean_abs_yoy_growth_pct", ascending=False).iloc[0]
        return [
            f"{hardest['coffee_type']} shows the highest overall instability based on variation level.",
            f"{most_volatile['coffee_type']} has the strongest year-over-year swings, which makes turning points harder to predict.",
            "Higher variation and larger year-over-year changes usually lead to less stable forecasts across models.",
        ]

    def get_year_prediction(self, model_name: str, year: int) -> pd.DataFrame:
        """Return a per-variety prediction table for a specific year.
        For historical years returns actual + prediction + error.
        For future years runs live inference and returns prediction only."""
        last_year = int(self.type_year["start_year"].max())
        if year <= last_year:
            preds = self.predictions[
                (self.predictions["model"] == model_name) & (self.predictions["start_year"] == year)
            ][["coffee_type", "actual", "prediction", "split"]].copy()
            preds["error_pct"] = (preds["prediction"] - preds["actual"]).abs() / preds["actual"].abs() * 100
            return preds.reset_index(drop=True)
        # Future year
        horizon = year - last_year
        future = self.forecast(model_name, horizon)
        row = future[future["start_year"] == year][["coffee_type", "prediction"]].copy()
        row["actual"] = None
        row["error_pct"] = None
        row["split"] = "future"
        return row.reset_index(drop=True)

    def get_country_summary(self, country: str) -> dict:
        """Return historical consumption and top varieties for a given country."""
        frame = self.coffee_long[self.coffee_long["country"] == country].copy()
        if frame.empty:
            return {}
        by_type = frame.groupby("coffee_type")["consumption"].sum().sort_values(ascending=False)
        by_year = frame.groupby("start_year")["consumption"].sum().sort_values()
        first = int(by_year.index.min())
        last = int(by_year.index.max())
        cagr = float(((by_year.iloc[-1] / by_year.iloc[0]) ** (1 / max(last - first, 1)) - 1) * 100)
        return {
            "country": country,
            "dominant_variety": by_type.index[0],
            "total_consumption": int(frame["consumption"].sum()),
            "year_range": f"{first}–{last}",
            "cagr_pct": round(cagr, 2),
            "by_variety": by_type.round(0).to_dict(),
            "recent_year": last,
            "recent_consumption": int(by_year.iloc[-1]),
        }

    def available_years(self) -> list[int]:
        """All historical years plus 8 future years for the year selector."""
        last = int(self.type_year["start_year"].max())
        hist = sorted(self.type_year["start_year"].unique().tolist())
        future = list(range(last + 1, last + 9))
        return hist + future

    def parse_upload(self, contents: str, filename: str) -> pd.DataFrame:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        lower = filename.lower()
        if lower.endswith(".csv"):
            return pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        if lower.endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(decoded))
        if lower.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(decoded))
        raise ValueError("Unsupported file type. Use CSV, XLSX, or Parquet.")

    @staticmethod
    def _season_from_year(year: int) -> str:
        return f"{year}/{str((year + 1) % 100).zfill(2)}"


class CoffeeAnalystLLM:
    def __init__(self) -> None:
        load_env_file()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def answer(self, question: str, service: ForecastingService, live_forecast: pd.DataFrame | None = None) -> str:
        if self.client is None:
            return "OPENAI_API_KEY is not configured. Add it to `.env` to enable the analyst chat."

        overall = service.metrics.round(3).to_dict(orient="records")
        by_type = service.metrics_by_type.round(3).to_dict(orient="records")
        diagnostics = service.diagnostics.round(3).to_dict(orient="records")
        top_countries = service.top_countries.round(3).to_dict(orient="records")
        forecast_context = [] if live_forecast is None else live_forecast.round(3).to_dict(orient="records")

        system_prompt = (
            "You are a coffee market forecasting analyst. "
            "Answer only from the supplied data and model outputs. "
            "Explain tradeoffs, uncertainty, model limitations, and which coffee types or countries appear riskier. "
            "If the user asks for an economic or production suggestion, ground it in the provided metrics, diagnostics, "
            "country concentration, and forecast context. Do not invent unsupported facts. "
            "This answer renders inside a compact chat bubble, not a report: keep it scannable. "
            "Prefer short paragraphs and tight bullet lists over long prose. Use bold only for the handful of numbers "
            "or terms that matter most. Do not use markdown headers (#, ##) — a bubble doesn't need document "
            "structure. Skip generic preambles and restating the question; open with the answer."
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Overall model metrics: {overall}\n\n"
            f"Metrics by coffee type: {by_type}\n\n"
            f"Forecast difficulty diagnostics: {diagnostics}\n\n"
            f"Top countries by coffee type: {top_countries}\n\n"
            f"Live forecast context: {forecast_context}\n"
        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text
