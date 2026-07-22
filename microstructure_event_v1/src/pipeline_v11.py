from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline import download_ticks, ensure_dirs, load_config, preprocess

PIP = {"USDJPY": 0.01}


def _rolling_delta(series: pd.Series, window: str) -> pd.Series:
    return series.rolling(window, min_periods=2).apply(lambda x: x[-1] - x[0], raw=True)


def _baseline_z(series: pd.Series, baseline_mask: pd.Series) -> pd.Series:
    baseline = series[baseline_mask].dropna()
    if baseline.empty:
        return pd.Series(np.nan, index=series.index)
    median = baseline.median()
    mad = (baseline - median).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = baseline.std(ddof=0)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (series - median) / scale


def _stress(z: pd.Series) -> pd.Series:
    z_pos = z.fillna(0).clip(lower=0, upper=6)
    return 100.0 * (1.0 - np.exp(-z_pos / 3.0))


def add_microstructure(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    data = frame.copy().sort_values("timestamp_utc")
    pip = PIP.get(cfg["symbol"], 0.0001)
    indexed = data.set_index("timestamp_utc", drop=False)

    data["velocity_pips_s"] = (_rolling_delta(indexed["mid"], "250ms") / pip / 0.25).to_numpy()
    velocity_indexed = pd.Series(data["velocity_pips_s"].to_numpy(), index=indexed.index)
    data["velocity_1s_pips_s"] = (_rolling_delta(indexed["mid"], "1s") / pip).to_numpy()
    data["acceleration_pips_s2"] = (_rolling_delta(velocity_indexed, "250ms") / 0.25).to_numpy()
    data["acceleration_ewm"] = data["acceleration_pips_s2"].ewm(span=12, adjust=False).mean()
    data["signed_tick"] = np.sign(data["mid"].diff()).fillna(0)

    bid_change = data["bid"].diff()
    ask_change = data["ask"].diff()
    bid_flow = np.where(
        bid_change > 0,
        data["bid_volume"],
        np.where(bid_change == 0, data["bid_volume"] - data["bid_volume"].shift(), -data["bid_volume"].shift()),
    )
    ask_flow = np.where(
        ask_change > 0,
        -data["ask_volume"].shift(),
        np.where(ask_change == 0, data["ask_volume"] - data["ask_volume"].shift(), data["ask_volume"]),
    )
    data["ofi_proxy"] = pd.Series(bid_flow - ask_flow, index=data.index).fillna(0)

    tick_count = pd.Series(1.0, index=indexed.index).rolling("1s").sum()
    data["tick_rate_1s"] = tick_count.to_numpy()
    data["rv_5s"] = indexed["log_return"].pow(2).rolling("5s", min_periods=10).sum().pow(0.5).to_numpy()
    data["spread_expansion"] = data["spread_pips"] / data["spread_pips"].expanding(20).median()

    buy_volume = np.where(data["signed_tick"] > 0, data["volume_proxy"], 0)
    sell_volume = np.where(data["signed_tick"] < 0, data["volume_proxy"], 0)
    window = cfg["vpin_bucket_ticks"]
    imbalance = pd.Series(np.abs(buy_volume - sell_volume), index=data.index).rolling(window, min_periods=20).sum()
    total = data["volume_proxy"].rolling(window, min_periods=20).sum()
    data["vpin_proxy"] = imbalance / total.replace(0, np.nan)

    start = pd.Timestamp(cfg["start_time_jst"], tz=cfg["analysis_timezone"])
    baseline_mask = data["timestamp_jst"] < start
    features = {
        "abs_return": data["log_return"].abs(),
        "abs_velocity": data["velocity_pips_s"].abs(),
        "abs_acceleration": data["acceleration_ewm"].abs(),
        "rv_5s": data["rv_5s"],
        "spread": data["spread_pips"],
        "tick_rate": data["tick_rate_1s"],
        "abs_ofi": data["ofi_proxy"].abs(),
        "vpin": data["vpin_proxy"],
        "tick_interval": data["tick_interval_ms"],
        "low_volume": -data["volume_proxy"],
        "down_velocity": -data["velocity_1s_pips_s"],
    }
    for name, series in features.items():
        data[f"{name}_z"] = _baseline_z(series, baseline_mask)
    return data


def score_events(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    data = frame.copy()
    start = pd.Timestamp(cfg["start_time_jst"], tz=cfg["analysis_timezone"])
    baseline_mask = data["timestamp_jst"] < start

    indexed_return = pd.Series(data["log_return"].to_numpy(), index=data["timestamp_utc"])
    rolling_sum = indexed_return.rolling("3s", min_periods=5).sum()
    rolling_abs = indexed_return.abs().rolling("3s", min_periods=5).sum().replace(0, np.nan)
    persistence = (rolling_sum.abs() / rolling_abs).to_numpy()
    reversal = (np.sign(data["log_return"]) != np.sign(data["log_return"].shift())).astype(float)
    reversal_rate = pd.Series(reversal.to_numpy(), index=data["timestamp_utc"]).rolling("3s", min_periods=5).mean().to_numpy()
    data["directional_persistence"] = persistence
    data["reversal_rate"] = reversal_rate

    data["persistence_z"] = _baseline_z(pd.Series(persistence, index=data.index), baseline_mask)
    data["low_reversal_z"] = _baseline_z(pd.Series(1 - reversal_rate, index=data.index), baseline_mask)
    data["sustained_move_z"] = _baseline_z(pd.Series(rolling_sum.abs().to_numpy(), index=data.index), baseline_mask)

    s_return = _stress(data["abs_return_z"])
    s_velocity = _stress(data["abs_velocity_z"])
    s_accel = _stress(data["abs_acceleration_z"])
    s_rv = _stress(data["rv_5s_z"])
    s_spread = _stress(data["spread_z"])
    s_tick = _stress(data["tick_rate_z"])
    s_ofi = _stress(data["abs_ofi_z"])
    s_vpin = _stress(data["vpin_z"])
    s_persistence = _stress(data["persistence_z"])
    s_low_reversal = _stress(data["low_reversal_z"])
    s_sustained = _stress(data["sustained_move_z"])
    s_drought = _stress(data["tick_interval_z"])
    s_low_volume = _stress(data["low_volume_z"])
    s_down_velocity = _stress(data["down_velocity_z"])

    data["flash_crash_score"] = (
        0.25 * s_return + 0.20 * s_velocity + 0.15 * s_accel + 0.15 * s_rv + 0.15 * s_spread + 0.10 * s_tick
    )
    data["stop_cascade_probability"] = (
        0.30 * s_persistence + 0.25 * s_tick + 0.20 * s_velocity + 0.15 * s_ofi + 0.10 * s_low_reversal
    )
    data["liquidity_stress"] = (
        0.30 * s_spread + 0.25 * s_drought + 0.20 * s_low_volume + 0.15 * s_rv + 0.10 * s_vpin
    )
    data["intervention_score"] = (
        0.30 * s_down_velocity + 0.20 * s_sustained + 0.15 * s_tick + 0.15 * s_ofi + 0.10 * s_spread + 0.10 * s_low_reversal
    )

    conditions = [
        data["liquidity_stress"] >= 65,
        (data["stop_cascade_probability"] >= 65) & (data["reversal_rate"] >= 0.45),
        (data["intervention_score"] >= 65) & (data["directional_persistence"] >= 0.70),
        data["flash_crash_score"] >= 65,
    ]
    labels = ["Liquidity Vacuum", "Stop Hunt", "Intervention-like Flow", "News Shock"]
    data["event_class"] = np.select(conditions, labels, default="Normal")
    return data


def build_dashboard(frame: pd.DataFrame, cfg: dict, output_path: str | Path) -> str | Path:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("USDJPY Bid / Ask / Mid", "250ms Velocity / Acceleration", "Event Scores"),
    )
    for column, name in [("bid", "Bid"), ("ask", "Ask"), ("mid", "Mid")]:
        figure.add_trace(go.Scatter(x=frame["timestamp_jst"], y=frame[column], name=name), row=1, col=1)
    figure.add_trace(go.Scatter(x=frame["timestamp_jst"], y=frame["velocity_pips_s"], name="Velocity 250ms"), row=2, col=1)
    figure.add_trace(go.Scatter(x=frame["timestamp_jst"], y=frame["acceleration_ewm"], name="Acceleration EWMA"), row=2, col=1)
    for column, name in [
        ("flash_crash_score", "Flash Crash"),
        ("intervention_score", "Intervention"),
        ("stop_cascade_probability", "Stop Cascade"),
        ("liquidity_stress", "Liquidity Stress"),
    ]:
        figure.add_trace(go.Scatter(x=frame["timestamp_jst"], y=frame[column], name=name), row=3, col=1)
    figure.update_yaxes(range=[0, 100], row=3, col=1)
    figure.update_layout(height=950, title="USDJPY Microstructure Event Monitor v1.1", hovermode="x unified")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def summary(frame: pd.DataFrame) -> dict:
    def peak(column: str, absolute: bool = False) -> dict:
        series = frame[column].abs() if absolute else frame[column]
        index = series.idxmax()
        return {"value": float(frame.loc[index, column]), "timestamp": str(frame.loc[index, "timestamp_jst"])}

    score_columns = ["flash_crash_score", "intervention_score", "stop_cascade_probability", "liquidity_stress"]
    dominant_column = max(score_columns, key=lambda c: float(frame[c].max()))
    dominant_index = frame[dominant_column].idxmax()
    return {
        "max_velocity_250ms": peak("velocity_pips_s", absolute=True),
        "max_acceleration_250ms": peak("acceleration_pips_s2", absolute=True),
        "max_spread_pips": peak("spread_pips"),
        "flash_crash_score": peak("flash_crash_score"),
        "intervention_score": peak("intervention_score"),
        "stop_cascade_probability": peak("stop_cascade_probability"),
        "liquidity_stress": peak("liquidity_stress"),
        "event_class": frame.loc[dominant_index, "event_class"],
        "dominant_score": dominant_column,
    }


def write_report(frame: pd.DataFrame, cfg: dict, html_path: str | Path) -> str | Path:
    result = summary(frame)
    rows = "".join(f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in result.items())
    html = f"""<!doctype html><meta charset='utf-8'><title>Event Report</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:40px auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>USDJPY Microstructure Event Report v1.1</h1>
<p>{cfg['start_time_jst']} ～ {cfg['end_time_jst']} JST</p><table>{rows}</table>
<p>速度は250ms固定時間窓、標準化はイベント前Baselineの中央値/MADを使用。スコアは有界変換後の加重平均です。</p>
<p>注記：OFI、VPIN、介入判定はDukascopyのBest Bid/Askティックに基づくProxyであり、市場全体の確定的判定ではありません。</p>"""
    Path(html_path).write_text(html, encoding="utf-8")
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(Path(html_path).with_suffix(".pdf")))
    except Exception as error:
        print("PDF出力を省略:", error)
    return html_path
