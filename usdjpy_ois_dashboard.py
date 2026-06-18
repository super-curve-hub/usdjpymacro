import os
import json
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        raise RuntimeError("FRED_API_KEY is not set. Add it to GitHub Secrets or local environment.")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "asc",
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    payload = r.json()
    if "observations" not in payload:
        raise RuntimeError(f"Unexpected FRED response for {series_id}: {payload}")

    df = pd.DataFrame(payload["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"].sort_index()


def tenor_to_period(tenor: str) -> ql.Period:
    tenor = str(tenor).strip().upper()
    unit = tenor[-1]
    num = int(tenor[:-1])

    if unit == "D":
        return ql.Period(num, ql.Days)
    if unit == "W":
        return ql.Period(num, ql.Weeks)
    if unit == "M":
        return ql.Period(num, ql.Months)
    if unit == "Y":
        return ql.Period(num, ql.Years)

    raise ValueError(f"Unknown tenor: {tenor}")


def build_ois_curve(quote_csv: Path, currency: str, today: ql.Date) -> ql.YieldTermStructureHandle:
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
            raise ValueError(f"Invalid rate in {quote_csv}: {row}")

        helpers.append(
            ql.OISRateHelper(
                settlement_days,
                tenor,
                ql.QuoteHandle(ql.SimpleQuote(rate)),
                index,
            )
        )

    curve = ql.PiecewiseLogCubicDiscount(today, helpers, day_count)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


def zero_rate(curve: ql.YieldTermStructureHandle, today: ql.Date, years: int) -> float:
    target = today + ql.Period(years, ql.Years)
    return curve.zeroRate(
        target,
        ql.Actual365Fixed(),
        ql.Compounded,
        ql.Annual,
    ).rate()


def forward_rate(curve: ql.YieldTermStructureHandle, today: ql.Date, start_y: int, end_y: int) -> float:
    start = today + ql.Period(start_y, ql.Years)
    end = today + ql.Period(end_y, ql.Years)
    return curve.forwardRate(
        start,
        end,
        ql.Actual365Fixed(),
        ql.Compounded,
        ql.Annual,
    ).rate()


def zscore_last(series: pd.Series, window: int = 252) -> float:
    x = series.dropna().tail(window)
    if len(x) < 30:
        return np.nan

    std = x.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan

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


def make_dashboard(latest: dict) -> None:
    labels = [
        "USD 2Y OIS",
        "JPY 2Y OIS",
        "OIS Spread",
        "USD 2Y2Y",
        "JPY 2Y2Y",
        "Forward Spread",
        "US10Y Real",
        "US2Y Treasury",
        "SOFR",
    ]

    values = [
        latest["USD_2Y_OIS"] * 100,
        latest["JPY_2Y_OIS"] * 100,
        latest["OIS_spread"] * 100,
        latest["USD_2Y2Y_forward"] * 100,
        latest["JPY_2Y2Y_forward"] * 100,
        latest["Forward_spread"] * 100,
        latest["US10Y_real_yield"] * 100,
        latest["US2Y_treasury"] * 100,
        latest["SOFR"] * 100,
    ]

    plt.figure(figsize=(9, 12))
    plt.barh(labels, values)
    plt.axvline(0, linewidth=1)
    plt.title(
        "USDJPY OIS Regime Dashboard\n"
        f"{latest['Regime']} / Macro Score {latest['Macro_score']:.2f}"
    )
    plt.xlabel("%")
    plt.tight_layout()
    plt.savefig(OUT / "dashboard.png", dpi=200)
    plt.close()


def main() -> None:
    jst = timezone(timedelta(hours=9))
    now_jst = datetime.now(jst)

    today = ql.Date(now_jst.day, now_jst.month, now_jst.year)
    ql.Settings.instance().evaluationDate = today

    usd_curve = build_ois_curve(DATA / "usd_ois_quotes.csv", "USD", today)
    jpy_curve = build_ois_curve(DATA / "jpy_ois_quotes.csv", "JPY", today)

    usd_2y_ois = zero_rate(usd_curve, today, 2)
    jpy_2y_ois = zero_rate(jpy_curve, today, 2)
    usd_2y2y = forward_rate(usd_curve, today, 2, 4)
    jpy_2y2y = forward_rate(jpy_curve, today, 2, 4)

    ois_spread = usd_2y_ois - jpy_2y_ois
    fwd_spread = usd_2y2y - jpy_2y2y

    us10_real = fred_series("DFII10") / 100.0
    us2y = fred_series("DGS2") / 100.0
    sofr = fred_series("SOFR") / 100.0

    rate_score = float(np.clip(ois_spread / 0.025, -3, 3))
    forward_score = float(np.clip(fwd_spread / 0.025, -3, 3))
    real_yield_score = zscore_last(us10_real)

    if np.isnan(real_yield_score):
        real_yield_score = 0.0

    macro_score = float(
        0.40 * rate_score
        + 0.35 * forward_score
        + 0.25 * real_yield_score
    )

    latest = {
        "timestamp_jst": now_jst.isoformat(),
        "valuation_date": f"{today.year():04d}-{today.month():02d}-{today.dayOfMonth():02d}",
        "US10Y_real_yield": float(us10_real.iloc[-1]),
        "US10Y_real_yield_date": str(us10_real.index[-1].date()),
        "US2Y_treasury": float(us2y.iloc[-1]),
        "US2Y_treasury_date": str(us2y.index[-1].date()),
        "SOFR": float(sofr.iloc[-1]),
        "SOFR_date": str(sofr.index[-1].date()),
        "USD_2Y_OIS": float(usd_2y_ois),
        "JPY_2Y_OIS": float(jpy_2y_ois),
        "OIS_spread": float(ois_spread),
        "USD_2Y2Y_forward": float(usd_2y2y),
        "JPY_2Y2Y_forward": float(jpy_2y2y),
        "Forward_spread": float(fwd_spread),
        "Rate_score": rate_score,
        "Forward_score": forward_score,
        "Real_yield_score": float(real_yield_score),
        "Macro_score": macro_score,
        "Regime": classify_regime(macro_score),
        "data_note": "OIS curves are built from data/usd_ois_quotes.csv and data/jpy_ois_quotes.csv. Replace these quotes with live or verified market quotes for production use.",
    }

    with open(OUT / "latest.json", "w", encoding="utf-8") as f:
        json.dump(latest, f, indent=2, ensure_ascii=False)

    pd.DataFrame([latest]).to_csv(OUT / "latest.csv", index=False)
    make_dashboard(latest)

    print(json.dumps(latest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
