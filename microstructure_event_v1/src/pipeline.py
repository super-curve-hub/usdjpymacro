from __future__ import annotations

import json
import lzma
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots
from scipy.special import expit

PIP = {"USDJPY": 0.01}


def load_config(path="../config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs(cfg):
    for path in cfg["paths"].values():
        Path("../" + path).mkdir(parents=True, exist_ok=True)


def _duka_url(symbol, hour_utc):
    return (
        f"https://datafeed.dukascopy.com/datafeed/{symbol}/"
        f"{hour_utc:%Y}/{hour_utc.month - 1:02d}/{hour_utc:%d}/{hour_utc:%H}h_ticks.bi5"
    )


def _parse_bi5(content, hour_utc):
    raw = lzma.decompress(content)
    rows = []
    for offset in range(0, len(raw), 20):
        chunk = raw[offset:offset + 20]
        if len(chunk) < 20:
            break
        ms, ask_i, bid_i, ask_volume, bid_volume = struct.unpack(">3i2f", chunk)
        rows.append(
            (
                hour_utc + pd.Timedelta(milliseconds=ms),
                bid_i,
                ask_i,
                bid_volume,
                ask_volume,
            )
        )
    return pd.DataFrame(
        rows,
        columns=["timestamp_utc", "bid_i", "ask_i", "bid_volume", "ask_volume"],
    )


def download_ticks(cfg):
    symbol = cfg["symbol"]
    timezone = cfg["analysis_timezone"]
    start = pd.Timestamp(cfg["baseline_start_time_jst"], tz=timezone).tz_convert("UTC")
    end = pd.Timestamp(cfg["end_time_jst"], tz=timezone).tz_convert("UTC")
    hours = pd.date_range(start.floor("h"), end.floor("h"), freq="h", tz="UTC")
    cache_dir = Path("../" + cfg["paths"]["cache"])
    raw_dir = Path("../" + cfg["paths"]["raw"])
    frames = []

    for hour in hours:
        cache_path = cache_dir / f"{symbol}_{hour:%Y%m%d_%H}.parquet"
        if cache_path.exists():
            frames.append(pd.read_parquet(cache_path))
            continue
        response = requests.get(_duka_url(symbol, hour), timeout=30)
        response.raise_for_status()
        frame = _parse_bi5(response.content, hour)
        scale = 1000 if symbol.endswith("JPY") else 100000
        frame["bid"] = frame.pop("bid_i") / scale
        frame["ask"] = frame.pop("ask_i") / scale
        frame.to_parquet(cache_path, index=False)
        frames.append(frame)

    output = pd.concat(frames).sort_values("timestamp_utc")
    output = output[(output["timestamp_utc"] >= start) & (output["timestamp_utc"] <= end)].copy()
    output["timestamp_jst"] = output["timestamp_utc"].dt.tz_convert(timezone)
    raw_path = raw_dir / f"{symbol}_{start:%Y%m%d_%H%M}_{end:%H%M}.parquet"
    output.to_parquet(raw_path, index=False)
    return output, raw_path


def robust_z(series, min_periods=20):
    median = series.expanding(min_periods=min_periods).median()
    mad = (series - median).abs().expanding(min_periods=min_periods).median()
    return (series - median) / (1.4826 * mad.replace(0, np.nan))


def preprocess(frame, cfg):
    data = frame.copy().sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
    pip = PIP.get(cfg["symbol"], 0.0001)
    data["mid"] = (data["bid"] + data["ask"]) / 2
    data["spread"] = data["ask"] - data["bid"]
    data["spread_pips"] = data["spread"] / pip
    data["spread_bps"] = 10000 * data["spread"] / data["mid"]
    data["log_return"] = np.log(data["mid"]).diff()
    data["tick_interval_ms"] = data["timestamp_utc"].diff().dt.total_seconds() * 1000
    data["volume_proxy"] = (data["bid_volume"] + data["ask_volume"]) / 2
    return data


def add_microstructure(frame, cfg):
    data = frame.copy()
    pip = PIP.get(cfg["symbol"], 0.0001)
    delta_t = data["tick_interval_ms"].div(1000).clip(lower=0.001)
    data["velocity_pips_s"] = data["mid"].diff().div(pip).div(delta_t)
    data["acceleration_pips_s2"] = data["velocity_pips_s"].diff().div(delta_t)
    data["acceleration_ewm"] = data["acceleration_pips_s2"].ewm(span=20, adjust=False).mean()
    data["signed_tick"] = np.sign(data["mid"].diff()).fillna(0)

    bid_change = data["bid"].diff()
    ask_change = data["ask"].diff()
    bid_flow = np.where(
        bid_change > 0,
        data["bid_volume"],
        np.where(
            bid_change == 0,
            data["bid_volume"] - data["bid_volume"].shift(),
            -data["bid_volume"].shift(),
        ),
    )
    ask_flow = np.where(
        ask_change > 0,
        -data["ask_volume"].shift(),
        np.where(
            ask_change == 0,
            data["ask_volume"] - data["ask_volume"].shift(),
            data["ask_volume"],
        ),
    )
    data["ofi_proxy"] = pd.Series(bid_flow - ask_flow, index=data.index).fillna(0)
    data["tick_rate_1s"] = 1 / data["tick_interval_ms"].div(1000).rolling(20, min_periods=5).median()
    data["rv_5s"] = data["log_return"].pow(2).rolling(100, min_periods=10).sum().pow(0.5)
    data["spread_expansion"] = data["spread_pips"] / data["spread_pips"].expanding(20).median()

    buy_volume = np.where(data["signed_tick"] > 0, data["volume_proxy"], 0)
    sell_volume = np.where(data["signed_tick"] < 0, data["volume_proxy"], 0)
    window = cfg["vpin_bucket_ticks"]
    imbalance = pd.Series(np.abs(buy_volume - sell_volume), index=data.index).rolling(window, min_periods=20).sum()
    total = data["volume_proxy"].rolling(window, min_periods=20).sum()
    data["vpin_proxy"] = imbalance / total.replace(0, np.nan)

    absolute_columns = {"log_return", "velocity_pips_s", "acceleration_ewm", "ofi_proxy"}
    columns = [
        "log_return",
        "velocity_pips_s",
        "acceleration_ewm",
        "rv_5s",
        "spread_pips",
        "tick_rate_1s",
        "ofi_proxy",
        "vpin_proxy",
    ]
    for column in columns:
        source = data[column].abs() if column in absolute_columns else data[column]
        data[column + "_z"] = robust_z(source)
    return data


def score_events(frame):
    data = frame.copy()

    def z(column):
        return data[column].fillna(0).clip(-8, 8)

    data["flash_crash_score"] = 100 * expit(
        -3
        + 0.25 * z("log_return_z")
        + 0.20 * z("velocity_pips_s_z")
        + 0.15 * z("acceleration_ewm_z")
        + 0.15 * z("rv_5s_z")
        + 0.15 * z("spread_pips_z")
        + 0.10 * z("tick_rate_1s_z")
    )

    persistence = (
        data["log_return"].rolling(50, min_periods=10).sum().abs()
        / data["log_return"].abs().rolling(50, min_periods=10).sum().replace(0, np.nan)
    )
    reversal = (
        np.sign(data["log_return"]) != np.sign(data["log_return"].shift())
    ).rolling(50, min_periods=10).mean()
    persistence_z = robust_z(persistence).fillna(0)
    reversal_z = robust_z(reversal).fillna(0)
    data["directional_persistence"] = persistence

    data["stop_cascade_probability"] = 100 * expit(
        -3
        + persistence_z
        + 0.8 * z("tick_rate_1s_z")
        + 0.8 * z("velocity_pips_s_z")
        + 0.5 * z("ofi_proxy_z")
        - 0.4 * reversal_z
    )
    data["liquidity_stress"] = 100 * expit(
        -3
        + 0.30 * z("spread_pips_z")
        + 0.25 * robust_z(data["tick_interval_ms"]).fillna(0)
        + 0.20 * robust_z(-data["volume_proxy"]).fillna(0)
        + 0.15 * z("rv_5s_z")
        + 0.10 * z("vpin_proxy_z")
    )
    sustained_move = robust_z(
        data["log_return"].rolling(50, min_periods=10).sum().abs()
    ).fillna(0)
    low_reversal = robust_z(1 - reversal).fillna(0)
    data["intervention_score"] = 100 * expit(
        -3
        + 0.25 * z("velocity_pips_s_z")
        + 0.20 * sustained_move
        + 0.15 * z("tick_rate_1s_z")
        + 0.15 * z("ofi_proxy_z")
        + 0.10 * z("spread_pips_z")
        + 0.10 * low_reversal
        + 0.05 * persistence_z
    )

    conditions = [
        data["liquidity_stress"] >= 70,
        (data["stop_cascade_probability"] >= 65) & (reversal >= 0.5),
        (data["intervention_score"] >= 70) & (persistence >= 0.75),
        data["flash_crash_score"] >= 65,
    ]
    labels = ["Liquidity Vacuum", "Stop Hunt", "Intervention-like Flow", "News Shock"]
    data["event_class"] = np.select(conditions, labels, default="Normal")
    return data


def build_dashboard(frame, cfg, output_path="../output/dashboard.html"):
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("USDJPY Bid / Ask / Mid", "Velocity / Acceleration", "Event Scores"),
    )
    for column, name in [("bid", "Bid"), ("ask", "Ask"), ("mid", "Mid")]:
        figure.add_trace(go.Scatter(x=frame["timestamp_jst"], y=frame[column], name=name), row=1, col=1)
    figure.add_trace(go.Scatter(x=frame["timestamp_jst"], y=frame["velocity_pips_s"], name="Velocity"), row=2, col=1)
    figure.add_trace(go.Scatter(x=frame["timestamp_jst"], y=frame["acceleration_ewm"], name="Acceleration EWMA"), row=2, col=1)
    for column, name in [
        ("flash_crash_score", "Flash Crash"),
        ("intervention_score", "Intervention"),
        ("stop_cascade_probability", "Stop Cascade"),
        ("liquidity_stress", "Liquidity Stress"),
    ]:
        figure.add_trace(go.Scatter(x=frame["timestamp_jst"], y=frame[column], name=name), row=3, col=1)
    figure.update_layout(height=950, title="USDJPY Microstructure Event Monitor v1.0", hovermode="x unified")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def summary(frame):
    def peak(column):
        index = frame[column].idxmax()
        return {
            "value": float(frame.loc[index, column]),
            "timestamp": str(frame.loc[index, "timestamp_jst"]),
        }

    return {
        "max_velocity": peak("velocity_pips_s"),
        "max_acceleration": peak("acceleration_pips_s2"),
        "max_spread_pips": peak("spread_pips"),
        "flash_crash_score": peak("flash_crash_score"),
        "intervention_score": peak("intervention_score"),
        "stop_cascade_probability": peak("stop_cascade_probability"),
        "liquidity_stress": peak("liquidity_stress"),
        "event_class": frame.loc[frame["flash_crash_score"].idxmax(), "event_class"],
    }


def write_report(frame, cfg, html_path="../output/event_report.html"):
    result = summary(frame)
    rows = "".join(f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in result.items())
    html = f"""<!doctype html><meta charset='utf-8'><title>Event Report</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:40px auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>USDJPY Microstructure Event Report v1.0</h1>
<p>{cfg['start_time_jst']} ～ {cfg['end_time_jst']} JST</p><table>{rows}</table>
<p>注記：OFI、VPIN、介入判定はDukascopyのBest Bid/Askティックに基づくProxyであり、市場全体の確定的判定ではありません。</p>"""
    Path(html_path).write_text(html, encoding="utf-8")
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(Path(html_path).with_suffix(".pdf")))
    except Exception as error:
        print("PDF出力を省略:", error)
    return html_path
