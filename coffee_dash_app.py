from __future__ import annotations

import os

from dash import Dash, Input, Output, State, dash_table, dcc, html, no_update, ctx
import pandas as pd

from coffee_service import CoffeeAnalystLLM, ForecastingService


# Load pre-trained model artefacts from disk — no training on startup.
service = ForecastingService.from_cache()
analyst = CoffeeAnalystLLM()

ALL_MODELS = list(service.artifacts.keys())
LIVE_FORECAST_MODELS = ["ARIMA", "XGBoost", "LSTM"]
HAS_TIMEGPT = "TimeGPT" in ALL_MODELS
ALL_YEARS = service.available_years()
LAST_HIST_YEAR = int(service.type_year["start_year"].max())
ALL_COUNTRIES = sorted(service.coffee_long["country"].unique().tolist())

app = Dash(__name__, title="Coffee Forecast Studio", suppress_callback_exceptions=True)
server = app.server


# ─── Helpers ───────────────────────────────────────────────────────────────────

def card(title: str, value: str, subtitle: str = "", accent: str = "#6c4022") -> html.Div:
    return html.Div(
        [
            html.Div(title, className="card-title"),
            html.Div(value, className="card-value", style={"color": accent}),
            html.Div(subtitle, className="card-subtitle"),
        ],
        className="metric-card",
    )


def section_header(title: str, description: str = "") -> html.Div:
    children: list = [html.H2(title, className="section-title")]
    if description:
        children.append(html.P(description, className="section-desc"))
    return html.Div(children, className="section-header")


def info_badge(text: str, color: str = "#5a341e") -> html.Span:
    return html.Span(text, className="info-badge", style={"background": color})


def app_graph(figure, graph_id: str | None = None, height: int = 480) -> dcc.Graph:
    kwargs = {"figure": figure, "className": "graph-inner", "config": {"responsive": False}}
    if graph_id is not None:
        kwargs["id"] = graph_id
    return dcc.Graph(**kwargs, style={"height": f"{height}px", "width": "100%"})


def pretty_name(name: str) -> str:
    return name.replace("_", " ").replace("yoy", "YoY").title()


def fmt_consumption(x: float) -> str:
    """Human-readable consumption numbers instead of scientific notation."""
    if pd.isna(x):
        return ""
    if abs(x) >= 1e9:
        return f"{x / 1e9:.2f} B"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f} M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.1f} K"
    return f"{x:.1f}"


def format_frame(df: pd.DataFrame) -> pd.DataFrame:
    CONSUMPTION_COLS = {"rmse", "mae", "prediction", "actual", "consumption"}
    PCT_COLS = {"mape", "smape", "r2", "coef_variation", "lag1_autocorr", "error_pct"}
    formatted = df.copy()
    for col in formatted.columns:
        if not pd.api.types.is_numeric_dtype(formatted[col]):
            continue
        col_l = col.lower()
        if col_l in PCT_COLS:
            formatted[col] = formatted[col].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
        elif col_l in CONSUMPTION_COLS:
            formatted[col] = formatted[col].map(fmt_consumption)
        else:
            formatted[col] = formatted[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return formatted


def df_table(df: pd.DataFrame, table_id: str) -> dash_table.DataTable:
    display_df = format_frame(df)
    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": pretty_name(c), "id": c} for c in display_df.columns],
        data=display_df.to_dict("records"),
        page_size=12,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#f3efe8", "fontWeight": "700", "border": "none", "fontSize": "13px"},
        style_cell={"padding": "10px 14px", "fontFamily": "Georgia, serif", "border": "none", "textAlign": "left", "fontSize": "14px"},
        style_data={"backgroundColor": "#fffdf8"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#fdf6ee"},
            {"if": {"column_id": "split", "filter_query": '{split} = "test"'}, "color": "#dc2626", "fontWeight": "600"},
            {"if": {"column_id": "split", "filter_query": '{split} = "future"'}, "color": "#2d6a4f", "fontWeight": "600"},
        ],
    )


def chat_bubble(role: str, content: str) -> html.Div:
    is_user = role == "user"
    return html.Div(
        html.Div(
            content,
            className="bubble-user" if is_user else "bubble-assistant",
        ),
        className="chat-row-user" if is_user else "chat-row-assistant",
    )


# ─── Tab layouts ───────────────────────────────────────────────────────────────

def build_overview_layout() -> html.Div:
    summary = service.summary_cards()
    cards = html.Div(
        [
            card("Best model (test RMSE)", summary["best_model"],
                 f"RMSE {summary['best_test_rmse']:,.0f}  ·  MAPE {summary['best_test_mape']:.2f}%", accent="#5a341e"),
            card("Coffee varieties tracked", str(summary["coffee_types"]),
                 f"Across {summary['countries']} producing countries"),
            card("Historical window", f"{summary['history_start']} – {summary['history_end']}",
                 f"Test period: {summary['test_start']} – {summary['test_end']}"),
            card("Active dataset", summary["source_name"].replace(".parquet", ""), "Pre-trained cache loaded"),
        ],
        className="cards-grid",
    )
    model_badges = html.Div(
        [
            html.Span("Models evaluated: ", style={"color": "#7e5d42", "fontSize": "13px", "fontWeight": "600"}),
            *[info_badge(m, "#5a341e" if m != "TimeGPT" else "#2d6a4f") for m in ALL_MODELS],
        ],
        style={"marginBottom": "20px", "display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap"},
    )
    return html.Div([
        section_header(
            "Global Coffee Consumption Overview",
            "Historical trends and geographic distribution of global domestic coffee consumption. "
            "Data spans multiple decades across Arabica, Robusta, and blended varieties.",
        ),
        cards,
        model_badges,
        html.Div([
            html.Div([
                html.H3("Consumption trends by variety", className="chart-title"),
                html.P(
                    "Annual domestic consumption aggregated across all producing countries. "
                    "Each line is one coffee variety; use the legend to isolate varieties.",
                    className="chart-desc",
                ),
                app_graph(service.historical_type_chart(), height=460),
            ], className="graph-panel"),
            html.Div([
                html.H3("Geographic distribution of consumption", className="chart-title"),
                html.P(
                    "Total cumulative consumption per country across the full dataset. "
                    "Warmer tones indicate larger markets.",
                    className="chart-desc",
                ),
                app_graph(service.country_world_map(), height=460),
            ], className="graph-panel"),
        ], className="two-col"),
        html.Div([
            html.H3("Top 8 countries by variety", className="chart-title"),
            html.P(
                "Largest consumers for each coffee variety. Country concentration is a key supply-chain risk indicator.",
                className="chart-desc",
            ),
            app_graph(service.top_countries_chart(), height=760),
        ], className="graph-panel"),
    ])


def build_diagnostics_layout() -> html.Div:
    model_str = ", ".join(ALL_MODELS[:-1]) + (" and " + ALL_MODELS[-1] if len(ALL_MODELS) > 1 else ALL_MODELS[0])
    return html.Div([
        section_header(
            "Model Diagnostics & Forecast Quality",
            f"Compare how {model_str} perform across coffee varieties. "
            "Lower MAPE and RMSE indicate better accuracy. "
            "Explore why some varieties are structurally harder to predict.",
        ),

        # Row 1: RMSE + Heatmap
        html.Div([
            html.Div([
                html.H3("Train vs Test RMSE by model", className="chart-title"),
                html.P(
                    "A narrow gap between train and test bars indicates good generalisation. "
                    "A large gap signals overfitting.",
                    className="chart-desc",
                ),
                app_graph(service.train_test_rmse_chart(), height=420),
            ], className="panel"),
            html.Div([
                html.H3("MAPE heatmap — model × variety", className="chart-title"),
                html.P(
                    "Mean Absolute Percentage Error on the held-out test set. "
                    "Green = low error (accurate), Red = high error. "
                    "Each cell reveals which model handles a variety best.",
                    className="chart-desc",
                ),
                app_graph(service.metrics_heatmap(), height=420),
            ], className="panel"),
        ], className="two-col"),

        # Row 2: MAPE bars + forecast risk
        html.Div([
            html.Div([
                html.H3("Test MAPE (%) by variety & model", className="chart-title"),
                html.P(
                    "Per-variety test error grouped by model. "
                    "Varieties with high MAPE across all models are inherently volatile — not a model failure.",
                    className="chart-desc",
                ),
                app_graph(service.test_mape_by_type_chart(), height=420),
            ], className="panel"),
            html.Div([
                html.H3("Why some varieties are harder to forecast", className="chart-title"),
                html.P(
                    "Coefficient of variation (left) measures overall instability. "
                    "Average YoY growth swing (right) shows how sharply trends reverse year-to-year.",
                    className="chart-desc",
                ),
                app_graph(service.forecast_risk_chart(), height=420),
                html.Ul(
                    [html.Li(note, className="insight-item") for note in service.forecast_risk_notes()],
                    className="insights-list",
                ),
            ], className="panel"),
        ], className="two-col"),

        # Row 3: Actual vs predicted
        html.Div([
            html.H3("Actual vs Predicted — interactive view", className="chart-title"),
            html.P(
                "Black = actual consumption. Orange = prediction on training data. Red = prediction on held-out test data. "
                "A good model keeps orange and red lines close to black throughout.",
                className="chart-desc",
            ),
            html.Div([
                html.Div([
                    html.Label("Model", className="dropdown-label"),
                    dcc.Dropdown(
                        id="diagnostic-model-dropdown",
                        options=[{"label": m, "value": m} for m in ALL_MODELS],
                        value="ARIMA", clearable=False, className="styled-dropdown",
                    ),
                ], style={"flex": "1"}),
                html.Div([
                    html.Label("Variety", className="dropdown-label"),
                    dcc.Dropdown(
                        id="diagnostic-type-dropdown",
                        options=[{"label": "All varieties", "value": "ALL"}]
                        + [{"label": c, "value": c} for c in sorted(service.type_year["coffee_type"].unique())],
                        value="ALL", clearable=False, className="styled-dropdown",
                    ),
                ], style={"flex": "1"}),
            ], className="dropdown-row"),
            app_graph(service.prediction_vs_actual_chart("ARIMA", None), graph_id="diagnostic-model-graph", height=620),
        ], className="panel"),
    ])


def build_forecast_layout() -> html.Div:
    year_opts = [
        {"label": f"{y} {'(history)' if y <= LAST_HIST_YEAR else '(forecast)'}", "value": y}
        for y in ALL_YEARS
    ]
    default_year = LAST_HIST_YEAR + 1  # first future year

    return html.Div([
        section_header(
            "Forecast Lab — Point Inference",
            "Select a model and a target year to see what each variety is predicted to consume. "
            "Historical years show the actual vs predicted comparison. Future years run live inference from the trained model. "
            + ("TimeGPT results are shown in the Diagnostics tab." if HAS_TIMEGPT else ""),
        ),

        # Controls
        html.Div([
            html.Div([
                html.Label("Model", className="dropdown-label"),
                dcc.Dropdown(
                    id="forecast-model-dropdown",
                    options=[{"label": m, "value": m} for m in LIVE_FORECAST_MODELS],
                    value="ARIMA", clearable=False, className="styled-dropdown",
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Target year", className="dropdown-label"),
                dcc.Dropdown(
                    id="forecast-year-dropdown",
                    options=year_opts,
                    value=default_year, clearable=False, className="styled-dropdown",
                ),
            ], style={"flex": "1"}),
            html.Div([
                html.Label("Variety (optional filter)", className="dropdown-label"),
                dcc.Dropdown(
                    id="forecast-type-dropdown",
                    options=[{"label": "All varieties", "value": "ALL"}]
                    + [{"label": c, "value": c} for c in sorted(service.type_year["coffee_type"].unique())],
                    value="ALL", clearable=False, className="styled-dropdown",
                ),
            ], style={"flex": "1"}),
            html.Div(
                html.Button("Run inference", id="run-forecast", className="action-button", style={"marginTop": "24px"}),
                style={"flex": "0 0 auto"},
            ),
        ], className="dropdown-row", style={"alignItems": "flex-end"}),

        # Year summary panel
        html.Div([
            html.H3(id="forecast-summary-title", children="Year prediction summary", className="chart-title"),
            html.P(
                id="forecast-summary-desc",
                children="Select a year and model above, then click Run inference.",
                className="chart-desc",
            ),
            html.Div(id="forecast-point-table"),
        ], className="panel"),

        # Trend chart
        html.Div([
            html.H3("Historical trend with forecast projection", className="chart-title"),
            html.P(
                "Dark line = historical consumption. Amber line = model projection. "
                "The projection extends to the selected target year.",
                className="chart-desc",
            ),
            app_graph(
                service.forecast_chart("ARIMA", ALL_YEARS.index(default_year) - ALL_YEARS.index(LAST_HIST_YEAR)),
                graph_id="future-forecast-graph",
                height=620,
            ),
        ], className="graph-panel"),

        dcc.Store(id="last-forecast-store"),
    ])


EXAMPLE_QUESTIONS = [
    "Which variety is hardest to forecast and why?",
    "Which model performs best overall? Is it significantly better?",
    "Which countries have the highest concentration risk for Robusta?",
    "What production strategy would you recommend given the forecast uncertainty?",
    "Compare ARIMA and LSTM across all coffee varieties.",
]


def build_chat_layout() -> html.Div:
    return html.Div([
        section_header(
            "Analyst Chat",
            "Ask the AI analyst about model performance, country risk, forecast uncertainty, or production strategy. "
            "Optionally focus the conversation on a specific country. "
            "All answers are grounded in the actual model outputs and data.",
        ),

        # Country context selector
        html.Div([
            html.Label("Country context (optional)", className="dropdown-label"),
            html.P(
                "Select a country to prime the analyst with that country's historical consumption profile "
                "and its share of each coffee variety.",
                className="chart-desc", style={"marginBottom": "8px"},
            ),
            dcc.Dropdown(
                id="chat-country-dropdown",
                options=[{"label": "No specific country", "value": "NONE"}]
                + [{"label": c, "value": c} for c in ALL_COUNTRIES],
                value="NONE", clearable=False, className="styled-dropdown",
                style={"maxWidth": "340px"},
            ),
        ], className="panel", style={"marginBottom": "16px"}),

        # Chat window + input
        html.Div([
            # Chat history display
            html.Div(
                id="chat-messages",
                className="chat-window",
                children=[
                    html.Div(
                        "Start the conversation by typing below or clicking one of the example prompts.",
                        className="chat-empty",
                    )
                ],
            ),

            # Example question chips
            html.Div([
                html.Span("Try: ", style={"fontSize": "12px", "color": "#8a5a2b", "fontWeight": "600"}),
                *[
                    html.Button(q, id={"type": "example-question", "index": i},
                                className="example-btn", n_clicks=0)
                    for i, q in enumerate(EXAMPLE_QUESTIONS)
                ],
            ], className="example-questions", style={"marginTop": "12px"}),

            # Input row
            html.Div([
                dcc.Textarea(
                    id="chat-question",
                    placeholder="Type your question here…",
                    className="chat-input",
                    style={"flex": "1", "minHeight": "70px", "resize": "vertical"},
                ),
                html.Div([
                    html.Button("Send", id="ask-analyst", className="action-button",
                                style={"marginTop": "0", "marginLeft": "10px"}),
                    html.Button("Clear", id="clear-chat", className="clear-button",
                                style={"marginLeft": "8px"}),
                ], style={"display": "flex", "flexDirection": "column", "gap": "8px", "alignSelf": "flex-start"}),
            ], style={"display": "flex", "alignItems": "flex-start", "gap": "0", "marginTop": "12px"}),
        ], className="panel"),

        dcc.Store(id="chat-history-store", data=[]),
    ])


# ─── App layout ────────────────────────────────────────────────────────────────

def render_tab_content(active_tab: str) -> html.Div:
    if active_tab == "overview":
        return build_overview_layout()
    if active_tab == "diagnostics":
        return build_diagnostics_layout()
    if active_tab == "forecast":
        return build_forecast_layout()
    if active_tab == "chat":
        return build_chat_layout()
    return build_overview_layout()

app.layout = html.Div([
    # Hero banner
    html.Div([
        html.Div([
            html.Div([
                html.Div("☕", style={"fontSize": "38px", "marginRight": "14px", "lineHeight": "1"}),
                html.Div([
                    html.Div("Coffee Forecast Studio", className="hero-title"),
                    html.Div(
                        "Multi-model time-series forecasting · " + " · ".join(ALL_MODELS),
                        className="hero-subtitle",
                    ),
                ]),
            ], style={"display": "flex", "alignItems": "center"}),
        ], className="hero-left"),
        html.Div([
            html.Div([html.Div(str(len(ALL_MODELS)), className="hero-stat-value"), html.Div("Models", className="hero-stat-label")], className="hero-stat"),
            html.Div([html.Div(str(service.type_year["coffee_type"].nunique()), className="hero-stat-value"), html.Div("Varieties", className="hero-stat-label")], className="hero-stat"),
            html.Div([html.Div(str(service.coffee_long["country"].nunique()), className="hero-stat-value"), html.Div("Countries", className="hero-stat-label")], className="hero-stat"),
        ], className="hero-stats"),
    ], className="hero"),

    # Tabs
    dcc.Tabs([
        dcc.Tab(label="Overview", value="overview", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="Diagnostics", value="diagnostics", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="Forecast Lab", value="forecast", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="Analyst Chat", value="chat", className="tab", selected_className="tab--selected"),
    ], id="main-tabs", value="overview", className="tabs-container"),
    html.Div(id="tab-content", children=render_tab_content("overview")),
], className="app-shell")


# ─── Styles ────────────────────────────────────────────────────────────────────

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      :root {
        --brown-900: #2b1a0e; --brown-800: #3d2416; --brown-700: #5a341e;
        --brown-600: #6c4022; --brown-500: #8a5a2b; --brown-400: #b28040;
        --brown-300: #d9b17b; --brown-100: #f3efe8;
        --cream-bg: #f7f1e6; --cream-card: #fffdf8; --cream-alt: #fdf6ee;
        --green-accent: #2d6a4f;
        --shadow: 0 4px 24px rgba(73,43,18,.09); --radius: 18px; --radius-sm: 10px;
      }
      *, *::before, *::after { box-sizing: border-box; }
      body { margin: 0; background: var(--cream-bg); color: var(--brown-900); font-family: Georgia, serif; font-size: 15px; line-height: 1.6; }
      h2, h3 { margin: 0 0 4px; font-weight: 700; }

      .app-shell { padding: 20px 24px 40px; max-width: 1480px; margin: 0 auto; }

      /* Hero */
      .hero {
        display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;
        padding: 28px 36px; border-radius: 24px;
        background: linear-gradient(135deg, var(--brown-800) 0%, var(--brown-500) 60%, var(--brown-300) 100%);
        color: #fff7ec; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(61,36,22,.3);
      }
      .hero-title { font-size: 36px; font-weight: 700; letter-spacing: 0.4px; line-height: 1.15; }
      .hero-subtitle { margin-top: 6px; font-size: 13px; opacity: .82; letter-spacing: 0.3px; }
      .hero-stats { display: flex; gap: 28px; }
      .hero-stat { text-align: center; }
      .hero-stat-value { font-size: 32px; font-weight: 700; line-height: 1; }
      .hero-stat-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; opacity: .75; margin-top: 4px; }

      /* Tabs */
      .tab { background: transparent !important; border: none !important; padding: 10px 16px !important; font-family: Georgia, serif !important; font-size: 14px !important; color: var(--brown-500) !important; }
      .tab--selected { border-bottom: 3px solid var(--brown-600) !important; color: var(--brown-700) !important; font-weight: 700 !important; background: transparent !important; }

      /* Section headers */
      .section-header { margin: 24px 0 16px; }
      .section-title { font-size: 22px; color: var(--brown-800); margin-bottom: 6px; }
      .section-desc { margin: 0; font-size: 14px; color: var(--brown-500); max-width: 820px; line-height: 1.55; }

      /* Cards */
      .cards-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 16px; }
      .metric-card { background: var(--cream-card); border-radius: var(--radius); padding: 20px 18px; box-shadow: var(--shadow); border-top: 3px solid var(--brown-300); }
      .card-title { font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; color: var(--brown-500); }
      .card-value { font-size: 26px; margin-top: 8px; font-weight: 700; }
      .card-subtitle { margin-top: 6px; color: var(--brown-500); font-size: 13px; }

      /* Panels */
      .panel { background: var(--cream-card); border-radius: var(--radius); padding: 20px 22px; box-shadow: var(--shadow); margin-bottom: 18px; overflow: hidden; }
      .graph-panel { background: var(--cream-card); border-radius: var(--radius); padding: 20px 22px; box-shadow: var(--shadow); margin-bottom: 18px; overflow: hidden; }
      .graph-inner { width: 100%; min-height: 320px; max-height: 800px; overflow: hidden; }
      .graph-inner .js-plotly-plot, .graph-inner .plot-container, .graph-inner .svg-container { height: 100% !important; width: 100% !important; }
      .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 0; }

      /* Chart labels */
      .chart-title { font-size: 16px; color: var(--brown-800); margin-bottom: 4px; }
      .chart-desc { font-size: 13px; color: var(--brown-500); margin: 0 0 12px; line-height: 1.5; max-width: 700px; }

      /* Badges */
      .info-badge { display: inline-block; font-size: 11px; font-family: system-ui, sans-serif; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; color: #fff; padding: 3px 10px; border-radius: 999px; }

      /* Dropdowns */
      .dropdown-row { display: flex; gap: 16px; margin-bottom: 18px; flex-wrap: wrap; align-items: flex-start; }
      .dropdown-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--brown-500); display: block; margin-bottom: 5px; }

      /* Buttons */
      .action-button { display: inline-block; margin-top: 16px; background: var(--brown-700); color: white; border: none; padding: 11px 22px; border-radius: 999px; cursor: pointer; font-size: 14px; font-family: Georgia, serif; transition: background .18s; }
      .action-button:hover { background: var(--brown-500); }
      .clear-button { background: transparent; border: 1.5px solid var(--brown-300); color: var(--brown-600); padding: 8px 16px; border-radius: 999px; cursor: pointer; font-size: 13px; font-family: Georgia, serif; transition: border-color .15s; margin-top: 8px; }
      .clear-button:hover { border-color: var(--brown-600); }

      /* Insights */
      .insights-list { padding-left: 18px; margin: 14px 0 0; }
      .insight-item { font-size: 13px; color: var(--brown-700); margin-bottom: 6px; line-height: 1.5; }

      /* Example questions */
      .example-questions { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 4px; align-items: center; }
      .example-btn { background: var(--brown-100); border: 1px solid var(--brown-300); color: var(--brown-700); border-radius: 999px; padding: 5px 14px; font-size: 12px; font-family: Georgia, serif; cursor: pointer; transition: background .15s, border-color .15s; }
      .example-btn:hover { background: var(--brown-300); border-color: var(--brown-500); }

      /* ── Chat UI ── */
      .chat-window {
        background: var(--cream-alt); border-radius: var(--radius-sm);
        padding: 16px; min-height: 300px; max-height: 520px;
        overflow-y: auto; display: flex; flex-direction: column; gap: 12px;
        border: 1px solid var(--brown-100);
      }
      .chat-empty { color: var(--brown-400); font-size: 14px; font-style: italic; text-align: center; margin: auto; }

      /* User message — right aligned */
      .chat-row-user { display: flex; justify-content: flex-end; }
      .bubble-user {
        background: var(--brown-700); color: #fff7ec; padding: 10px 16px;
        border-radius: 18px 18px 4px 18px; max-width: 72%; font-size: 14px;
        line-height: 1.55; white-space: pre-wrap;
      }
      /* Assistant message — left aligned */
      .chat-row-assistant { display: flex; justify-content: flex-start; }
      .bubble-assistant {
        background: var(--cream-card); color: var(--brown-900); padding: 10px 16px;
        border-radius: 18px 18px 18px 4px; max-width: 82%; font-size: 14px;
        line-height: 1.65; white-space: pre-wrap; box-shadow: 0 2px 8px rgba(73,43,18,.08);
        border: 1px solid var(--brown-100);
      }
      .chat-thinking {
        background: var(--brown-100); color: var(--brown-400); padding: 10px 16px;
        border-radius: 18px; font-size: 13px; font-style: italic; max-width: 200px;
      }

      /* Chat input */
      .chat-input {
        border: 1.5px solid var(--brown-300); border-radius: var(--radius-sm);
        padding: 12px; font-family: Georgia, serif; font-size: 14px;
        background: var(--cream-card); color: var(--brown-900);
        resize: vertical; outline: none; transition: border-color .2s; width: 100%;
      }
      .chat-input:focus { border-color: var(--brown-600); }

      /* Responsive */
      @media (max-width: 980px) {
        .cards-grid { grid-template-columns: 1fr 1fr; }
        .two-col { grid-template-columns: 1fr; }
        .hero { flex-direction: column; align-items: flex-start; }
        .hero-title { font-size: 28px; }
        .dropdown-row { flex-direction: column; }
      }
      @media (max-width: 600px) {
        .cards-grid { grid-template-columns: 1fr; }
        .app-shell { padding: 12px; }
        .hero-title { font-size: 22px; }
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>
"""


# ─── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
)
def update_tab_content(active_tab: str):
    return render_tab_content(active_tab)

@app.callback(
    Output("diagnostic-model-graph", "figure"),
    Input("diagnostic-model-dropdown", "value"),
    Input("diagnostic-type-dropdown", "value"),
)
def update_diagnostic_graph(model_name: str, coffee_type: str):
    return service.prediction_vs_actual_chart(model_name, None if coffee_type == "ALL" else coffee_type)


@app.callback(
    Output("future-forecast-graph", "figure"),
    Output("forecast-point-table", "children"),
    Output("forecast-summary-title", "children"),
    Output("forecast-summary-desc", "children"),
    Output("last-forecast-store", "data"),
    Input("run-forecast", "n_clicks"),
    State("forecast-model-dropdown", "value"),
    State("forecast-year-dropdown", "value"),
    State("forecast-type-dropdown", "value"),
    prevent_initial_call=False,
)
def run_forecast(_n: int | None, model_name: str, target_year: int, coffee_type: str):
    # Horizon = how many years beyond last historical year
    horizon = max(1, target_year - LAST_HIST_YEAR) if target_year > LAST_HIST_YEAR else 1

    # Point prediction table for selected year
    point = service.get_year_prediction(model_name, target_year)
    if coffee_type != "ALL":
        point = point[point["coffee_type"] == coffee_type].copy()

    # Trend chart showing history + projection up to target year
    fig = service.forecast_chart(model_name, horizon, None if coffee_type == "ALL" else coffee_type)

    is_future = target_year > LAST_HIST_YEAR
    period_label = f"{target_year}/{str((target_year + 1) % 100).zfill(2)}"
    title = f"{model_name} predictions for {period_label}"
    desc = (
        f"Forecast for season {period_label} using {model_name}. "
        "No actual values are available for future years."
        if is_future else
        f"Model predictions vs actual consumption for season {period_label}. "
        "Error % = |predicted − actual| / actual × 100."
    )

    table_cols = ["coffee_type", "prediction", "actual", "error_pct", "split"] if not is_future else ["coffee_type", "prediction", "split"]
    table_df = point[[c for c in table_cols if c in point.columns]]
    return fig, df_table(table_df, "forecast-point-datatable"), title, desc, point.to_dict("records")


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.callback(
    Output("chat-history-store", "data"),
    Output("chat-question", "value"),
    Input("ask-analyst", "n_clicks"),
    State("chat-question", "value"),
    State("chat-country-dropdown", "value"),
    State("chat-history-store", "data"),
    State("last-forecast-store", "data"),
    prevent_initial_call=True,
)
def send_message(
    _n: int | None,
    question: str | None,
    country: str,
    history: list[dict],
    forecast_data: list[dict] | None,
) -> tuple:
    if not question or not question.strip():
        return no_update, no_update

    history = list(history or [])
    history.append({"role": "user", "content": question.strip()})

    # Enrich with country context if selected
    country_ctx = ""
    if country and country != "NONE":
        summary = service.get_country_summary(country)
        if summary:
            country_ctx = (
                f"\n\nCountry focus — {country}: "
                f"dominant variety: {summary['dominant_variety']}, "
                f"total consumption: {summary['total_consumption']:,} cups, "
                f"period: {summary['year_range']}, "
                f"CAGR: {summary['cagr_pct']:.1f}%/yr, "
                f"by variety: {summary['by_variety']}."
            )

    augmented_question = question.strip() + country_ctx
    forecast_df = pd.DataFrame(forecast_data) if forecast_data else None
    answer = analyst.answer(augmented_question, service, forecast_df)

    history.append({"role": "assistant", "content": answer})
    return history, ""


@app.callback(
    Output("chat-messages", "children"),
    Input("chat-history-store", "data"),
    Input("clear-chat", "n_clicks"),
)
def render_chat(history: list[dict], _clear: int | None) -> list:
    if ctx.triggered_id == "clear-chat":
        return [html.Div("Conversation cleared. Ask a new question below.", className="chat-empty")]
    if not history:
        return [html.Div("Start the conversation by typing below or clicking one of the example prompts.", className="chat-empty")]
    return [chat_bubble(msg["role"], msg["content"]) for msg in history]


@app.callback(
    Output("chat-history-store", "data", allow_duplicate=True),
    Input("clear-chat", "n_clicks"),
    prevent_initial_call=True,
)
def clear_history(_n: int | None) -> list:
    return []


@app.callback(
    Output("chat-question", "value", allow_duplicate=True),
    [Input({"type": "example-question", "index": i}, "n_clicks") for i in range(len(EXAMPLE_QUESTIONS))],
    prevent_initial_call=True,
)
def fill_example(*_clicks):
    if not ctx.triggered_id:
        return no_update
    return EXAMPLE_QUESTIONS[ctx.triggered_id["index"]]


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False,
    )
