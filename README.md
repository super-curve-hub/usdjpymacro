# USDJPY Macro OIS Dashboard

毎朝 05:00 JST に GitHub Actions で実行するドル円OISレジーム監視ダッシュボードです。

## 必要な設定

GitHub Repository Secrets に以下を登録してください。

```text
FRED_API_KEY
```

## 実行内容

- FRED APIから `DFII10`, `DGS2`, `SOFR` を取得
- `data/usd_ois_quotes.csv` からSOFR OISカーブをQuantLibで構築
- `data/jpy_ois_quotes.csv` からTONA OISカーブをQuantLibで構築
- `USD 2Y OIS`, `JPY 2Y OIS`, `2Y2Y Forward` を計算
- `output/latest.json`, `output/latest.csv`, `output/dashboard.png` を生成

## 注意

初期状態のOISクォートCSVはサンプルです。本番運用では検証済みの市場クォートに置き換えてください。
