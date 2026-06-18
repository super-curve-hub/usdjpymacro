import os
import json
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import QuantLib as ql

from datetime import datetime, timezone, timedelta
from pathlib import Path


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

FRED_API_KEY = os.environ.get("FRED_API_KEY")


def fred_series(series_id: str) -> pd.Series:
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY is not set.")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "asc",
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    df = pd.DataFrame(r.json()["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df.dropna().set_index("date")["value"].sort_index()


def get_usdjpy_history() -> pd.Series:
    df = yf.download(
        "JPY=X",
        period="2y",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("USDJPY data download failed.")

    close = df["Close"].squeeze()
    close.name = "USDJPY"
    return close.dropna()


def tenor_to_period(tenor: str) -> ql.Period:
    tenor = str(tenor).strip().upper()
    num = int(tenor[:-1])
    unit = tenor[-1]

    if unit == "D":
        return ql.Period(num, ql.Days)
    if unit == "W":
        return ql.Period(num, ql.Weeks)
    if unit == "M":
        return ql.Period(num, ql.Months)
    if unit == "Y":
        return ql.Period(num, ql.Years)

    raise ValueError(f"Unknown tenor: {tenor}")


def build_ois_curve(
    quote_csv: Path,
    currency: str,
    today: ql.Date,
) -> ql.YieldTermStructureHandle:
    ql.Settings.instance().evaluationDate = today
    quotes = pd.read_csv(quote_csv)

    if currency == "USD":
        day_count = ql.Actual360()
        index = ql.Sofr()
        settlement_days = 2

    elif currency == "JPY":
        calendar = ql.Japan()
        day_count = ql.Actual365Fixed()
        index = ql.OvernightIndex(
            "TONA",
            0,
            ql.JPYCurrency(),
            calendar,
            ql.Actual365Fixed(),
        )
        settlement_days = 2

    else:
        raise ValueError("currency must be USD or JPY")

    helpers = []

    for _, row in quotes.iterrows():
        tenor = tenor_to_period(row["tenor"])
        rate = float(row["rate"])

        if not math.isfinite(rate):
            raise ValueError(f"Invalid rate: {row}")

        helpers.append(
            ql.OISRateHelper(
                settlement_days,
                tenor,
                ql.QuoteHandle(ql.SimpleQuote(rate)),
                index,
            )
        )

    curve = ql.PiecewiseLogCubicDiscount(
        today,
        helpers,
        day_count,
    )

    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


def zero_rate(
    curve: ql.YieldTermStructureHandle,
    today: ql.Date,
    years: int,
) -> float:
    target = today + ql.Period(years, ql.Years)

    return curve.zeroRate(
        target,
        ql.Actual365Fixed(),
        ql.Compounded,
        ql.Annual,
    ).rate()


def forward_rate(
    curve: ql.YieldTermStructureHandle,
    today: ql.Date,
    start_y: int,
    end_y: int,
) -> float:
    start = today + ql.Period(start_y, ql.Years)
    end = today + ql.Period(end_y, ql.Years)

    return curve.forwardRate(
        start,
        end,
        ql.Actual365Fixed(),
        ql.Compounded,
        ql.Annual,
    ).rate()


def zscore(series: pd.Series, window: int = 252) -> float:
    x = series.dropna().tail(window)

    if len(x) < 30:
        return 0.0

    std = x.std(ddof=1)

    if std == 0 or np.isnan(std):
        return 0.0

    return float((x.iloc[-1] - x.mean()) / std)


def history_zscore(
    history: pd.DataFrame,
    column: str,
    window: int = 252,
) -> float:
    if history is None or column not in history.columns:
        return 0.0

    x = pd.to_numeric(history[column], errors="coerce").dropna().tail(window)

    if len(x) < 30:
        return 0.0

    std = x.std(ddof=1)

    if std == 0 or np.isnan(std):
        return 0.0

    return float((x.iloc[-1] - x.mean()) / std)


def classify_regime(score: float) -> str:
    if score > 1.5:
        return "STRONG USDJPY BULL"
    if score > 0.5:
        return "USDJPY BULL"
    if score > -0.5:
        return "NEUTRAL"
    if score > -1.5:
        return "JPY BULL"
    return "STRONG JPY BULL"


def update_ois_history(latest: dict) -> pd.DataFrame:
    history_file = OUT / "ois_history.csv"

    history_row = pd.DataFrame([{
        "date": latest["valuation_date"],
        "USD_2Y_OIS": latest["USD_2Y_OIS"],
        "JPY_2Y_OIS": latest["JPY_2Y_OIS"],
        "OIS_spread": latest["OIS_spread"],
        "USD_2Y2Y": latest["USD_2Y2Y_forward"],
        "JPY_2Y2Y": latest["JPY_2Y2Y_forward"],
        "Forward_spread": latest["Forward_spread"],
    }])

    if history_file.exists():
        old = pd.read_csv(history_file)

        history = pd.concat(
            [old, history_row],
            ignore_index=True,
        )

        history = (
            history
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
        )

    else:
        history = history_row

    history.to_csv(history_file, index=False)
    return history


def make_dashboard(latest: dict) -> None:
    labels = [
        "USDJPY",
        "US10Y Real",
        "US2Y Treasury",
        "SOFR",
        "USD 2Y OIS",
        "JPY 2Y OIS",
        "OIS Spread",
        "USD 2Y2Y",
        "JPY 2Y2Y",
        "Forward Spread",
        "Rate Score",
        "Forward Score",
        "Real Score",
        "Momentum Score",
        "Macro Score",
    ]

    values = [
        latest["USDJPY"],
        latest["US10Y_real_yield"] * 100,
        latest["US2Y_treasury"] * 100,
        latest["SOFR"] * 100,
        latest["USD_2Y_OIS"] * 100,
        latest["JPY_2Y_OIS"] * 100,
        latest["OIS_spread"] * 100,
        latest["USD_2Y2Y_forward"] * 100,
        latest["JPY_2Y2Y_forward"] * 100,
        latest["Forward_spread"] * 100,
        latest["Rate_score"],
        latest["Forward_score"],
        latest["Real_yield_score"],
        latest["Momentum_score"],
        latest["Macro_score"],
    ]

    plt.figure(figsize=(10, 14))
    plt.barh(labels, values)
    plt.axvline(0, linewidth=1)

    plt.title(
        "USDJPY Macro / OIS Regime Dashboard\n"
        f"{latest['Regime']} | Score {latest['Macro_score']:.2f}"
    )

    plt.tight_layout()
    plt.savefig(OUT / "dashboard.png", dpi=200)
    plt.close()


def main() -> None:
    jst = timezone(timedelta(hours=9))
    now_jst = datetime.now(jst)

    today = ql.Date(
        now_jst.day,
        now_jst.month,
        now_jst.year,
    )

    ql.Settings.instance().evaluationDate = today

    real10 = fred_series("DFII10") / 100.0
    us2y = fred_series("DGS2") / 100.0
    sofr = fred_series("SOFR") / 100.0

    usdjpy_series = get_usdjpy_history()
    usdjpy_spot = float(usdjpy_series.iloc[-1])
    usdjpy_ret20 = usdjpy_series.pct_change(20)

    usd_curve = build_ois_curve(
        DATA / "usd_ois_quotes.csv",
        "USD",
        today,
    )

    jpy_curve = build_ois_curve(
        DATA / "jpy_ois_quotes.csv",
        "JPY",
        today,
    )

    usd_2y_ois = zero_rate(usd_curve, today, 2)
    jpy_2y_ois = zero_rate(jpy_curve, today, 2)

    usd_2y2y = forward_rate(usd_curve, today, 2, 4)
    jpy_2y2y = forward_rate(jpy_curve, today, 2, 4)

    ois_spread = usd_2y_ois - jpy_2y_ois
    forward_spread = usd_2y2y - jpy_2y2y

    real_yield_score = zscore(real10)
    momentum_score = zscore(usdjpy_ret20)

    latest_pre_score = {
        "valuation_date": f"{today.year():04d}-{today.month():02d}-{today.dayOfMonth():02d}",
        "USD_2Y_OIS": float(usd_2y_ois),
        "JPY_2Y_OIS": float(jpy_2y_ois),
        "OIS_spread": float(ois_spread),
        "USD_2Y2Y_forward": float(usd_2y2y),
        "JPY_2Y2Y_forward": float(jpy_2y2y),
        "Forward_spread": float(forward_spread),
    }

    history = update_ois_history(latest_pre_score)

    rate_score = history_zscore(history, "OIS_spread")
    forward_score = history_zscore(history, "Forward_spread")

    if rate_score == 0.0:
        rate_score = float(np.clip(ois_spread / 0.025, -3, 3))

    if forward_score == 0.0:
        forward_score = float(np.clip(forward_spread / 0.025, -3, 3))

    macro_score = float(
        0.35 * rate_score
        + 0.35 * forward_score
        + 0.20 * real_yield_score
        + 0.10 * momentum_score
    )

    latest = {
        "timestamp_jst": now_jst.isoformat(),
        "valuation_date": latest_pre_score["valuation_date"],

        "USDJPY": usdjpy_spot,
        "USDJPY_date": str(usdjpy_series.index[-1].date()),

        "US10Y_real_yield": float(real10.iloc[-1]),
        "US10Y_real_yield_date": str(real10.index[-1].date()),

        "US2Y_treasury": float(us2y.iloc[-1]),
        "US2Y_treasury_date": str(us2y.index[-1].date()),

        "SOFR": float(sofr.iloc[-1]),
        "SOFR_date": str(sofr.index[-1].date()),

        "USD_2Y_OIS": latest_pre_score["USD_2Y_OIS"],
        "JPY_2Y_OIS": latest_pre_score["JPY_2Y_OIS"],
        "OIS_spread": latest_pre_score["OIS_spread"],

        "USD_2Y2Y_forward": latest_pre_score["USD_2Y2Y_forward"],
        "JPY_2Y2Y_forward": latest_pre_score["JPY_2Y2Y_forward"],
        "Forward_spread": latest_pre_score["Forward_spread"],

        "Rate_score": float(rate_score),
        "Forward_score": float(forward_score),
        "Real_yield_score": float(real_yield_score),
        "Momentum_score": float(momentum_score),
        "Macro_score": float(macro_score),

        "Regime": classify_regime(macro_score),

        "data_note": (
            "FRED and Yahoo Finance are live data. "
            "OIS curves are currently built from data/usd_ois_quotes.csv "
            "and data/jpy_ois_quotes.csv. "
            "OIS_spread and Forward_spread are stored in output/ois_history.csv. "
            "When history length reaches 30 rows, Rate_score and Forward_score "
            "use historical z-scores. Until then, fallback scaling is used."
        ),
    }

    with open(OUT / "latest.json", "w", encoding="utf-8") as f:
        json.dump(latest, f, indent=2, ensure_ascii=False)

    pd.DataFrame([latest]).to_csv(OUT / "latest.csv", index=False)

    make_dashboard(latest)

    print(json.dumps(latest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
