from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_DIR = PROJECT_DIR / "notebooks"
SRC_DIR = PROJECT_DIR / "src"

os.chdir(NOTEBOOK_DIR)
sys.path.insert(0, str(SRC_DIR))

from pipeline import (  # noqa: E402
    add_microstructure,
    build_dashboard,
    download_ticks,
    ensure_dirs,
    load_config,
    preprocess,
    score_events,
    summary,
    write_report,
)


def main() -> None:
    cfg = load_config(PROJECT_DIR / "config.json")
    ensure_dirs(cfg)

    raw, raw_path = download_ticks(cfg)
    processed = preprocess(raw, cfg)
    features = add_microstructure(processed, cfg)
    scored = score_events(features)

    start = pd.Timestamp(cfg["start_time_jst"], tz=cfg["analysis_timezone"])
    end = pd.Timestamp(cfg["end_time_jst"], tz=cfg["analysis_timezone"])
    event = scored[
        (scored["timestamp_jst"] >= start) & (scored["timestamp_jst"] <= end)
    ].copy()
    if event.empty:
        raise RuntimeError("指定したイベント区間にティックがありません。")

    processed_dir = PROJECT_DIR / cfg["paths"]["processed"]
    output_dir = PROJECT_DIR / cfg["paths"]["output"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = processed_dir / "USDJPY_microstructure_full.parquet"
    event_path = processed_dir / "USDJPY_event_20260722_1630_1635.parquet"
    scored.to_parquet(full_path, index=False)
    event.to_parquet(event_path, index=False)

    dashboard_path = output_dir / "dashboard.html"
    report_path = output_dir / "event_report.html"
    build_dashboard(event, cfg, dashboard_path)
    write_report(event, cfg, report_path)

    result = summary(event)
    result.update(
        {
            "raw_ticks": int(len(raw)),
            "event_ticks": int(len(event)),
            "raw_path": str(raw_path),
            "processed_path": str(full_path),
            "event_path": str(event_path),
            "dashboard_path": str(dashboard_path),
            "report_path": str(report_path),
        }
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
