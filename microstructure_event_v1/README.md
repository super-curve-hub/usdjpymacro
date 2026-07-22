# USDJPY Microstructure Event Monitor v1.0

DukascopyのBest Bid/Askティックを用いて、USDJPYの短時間イベントを再現・検証するNotebook群です。

## 対象時間

- 分析区間: 2026-07-22 16:30:00–16:35:00 JST
- Baseline: 2026-07-22 15:30:00–16:30:00 JST

## Notebook

1. `01_Data_Download.ipynb`
2. `02_Preprocessing.ipynb`
3. `03_Market_Microstructure.ipynb`
4. `04_Event_Detection.ipynb`
5. `05_Dashboard.ipynb`
6. `06_Report.ipynb`

## 実行

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

上から順番に実行してください。出力は`output/`、中間データは`data/`へ保存されます。

## 指標

- Tick Velocity / Acceleration
- Rolling Volatility
- Spread Expansion
- Signed Tick
- OFI Proxy
- VPIN Proxy
- Flash Crash Score
- Intervention Score
- Stop Cascade Probability
- Liquidity Stress Index

## 注意

Dukascopyは単一データ供給元のBest Bid/Askです。OFI、VPIN、介入判定は市場全体の板・約定フローではなくProxyです。イベント分類は確定判定ではなく、通常時分布からの相対的異常検出として利用してください。
