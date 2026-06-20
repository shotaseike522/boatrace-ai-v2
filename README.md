# 競艇AI予想サイト (v2)

友人と楽しむための競艇予想サイト。AI予想・荒れやすさ・近似100レースの配当分布や
着順発生率を、スマホ向けの見やすい画面で表示する。

## 構成

```
boat_model/                 予測モデル本体（特徴量・モデル・artifacts入出力）
run_train_models.py         月次: 過去データからモデルを学習し artifacts/ に保存
run_site_predictions.py     日次: 出走表CSVから予測CSVを生成
daily_prep.py                日次: 出走表取得 + 予測実行（自動化用）
app.py                       Streamlit: スマホ向け予想表示画面
data/                        学習用CSV・出走表CSVの置き場
artifacts/                   学習済みモデルの置き場
outputs/                     予測結果CSVの置き場
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

### 1. モデルを学習する（最初の1回、以降は月1回程度）

`data/` フォルダに学習用CSVを置く。

```
data/dataset_1_past_201307_202602.csv
data/dataset_2_recent_202603_202605.csv
```

学習を実行する。

```bash
python run_train_models.py
```

`artifacts/` フォルダにモデルファイル一式が保存される。

### 2. 当日の出走表から予測を作る（毎日）

`data/races_YYYYMMDD.csv` を用意し、以下を実行する。

```bash
python run_site_predictions.py --input data/races_20260701.csv --output outputs/site_predictions_20260701.csv
```

### 3. 画面で確認する

```bash
streamlit run app.py
```

ブラウザが開き、競艇場・レースを選ぶとAI予想が表示される。

## 自動化（daily_prep.py）

`daily_prep.py` は出走表の取得 → `run_site_predictions.py` の実行までを
1コマンドで行う。GitHub Actions等のスケジューラから定期実行する想定。

```bash
python daily_prep.py
```
