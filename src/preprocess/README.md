# TPEXT Preprocessing Package


## 概要

TPEXTのpreprocessパッケージは、日本語と英語のテキストデータの前処理を効率的に行うためのツール群です。
大規模なテキストデータセットの処理、統計分析、品質チェック機能を提供します。

## 主要機能

### 1. 日本語テキスト処理 (`japanese_text_processor`)
- Janomeを使用した形態素解析
- 名詞の抽出とフィルタリング
- URLとメンション（@ユーザー名）の除去
- チャンク単位での効率的な処理

### 2. 英語テキスト処理 (`english_text_processor`)
- spaCyを使用した言語処理
- レンマ化（語幹の抽出）
- ストップワードの除去
- 品詞フィルタリング（名詞、動詞、形容詞のみ）

### 3. データ結合機能 (`processed_data_combiner`)
- 複数のPickleファイルの結合
- 重複データの検出と除去
- 処理済みデータの統合

### 4. 統計分析機能 (`data_statistics`)
- 単語数の分布分析
- データ品質の評価
- パーセンタイル分析
- 詳細な統計レポートの生成

### 5. データ閲覧機能 (`data_viewer`)
- Pickleファイルの内容確認
- データサンプルの表示
- カラム情報の詳細表示

### 6. CSV検査機能 (`csv_inspector`)
- CSVファイルの構造分析
- データ型の自動判定
- 欠損値とユニーク値の統計

## インストール

### 必要な依存関係

```bash
pip install pandas numpy spacy janome pandarallel
```

### spaCy英語モデルのダウンロード

```bash
python -m spacy download en_core_web_sm
```

## 使用例

### 日本語テキストの前処理

```python
from tpext.preprocess import JapaneseTextProcessor

# 処理器の初期化
processor = JapaneseTextProcessor()

# CSVファイルを読み込んで処理
processor.load_csv_data("input_data.csv", text_column="text")

# テキストの前処理を実行
processor.preprocess_text()

# 結果をPickleファイルに保存
processor.save_processed_data("output/processed_japanese.pkl")
```

### 英語テキストの前処理

```python
from tpext.preprocess import EnglishTextProcessor

# 処理器の初期化
processor = EnglishTextProcessor()

# データの読み込み
processor.load_data("english_data.csv")

# 前処理の実行
processor.preprocess_texts()

# 結果の保存
processor.save_to_pickle("output/processed_english.pkl")
```

### データの結合

```python
from tpext.preprocess import combine_pickle_files

# 複数のPickleファイルを結合
combined_data = combine_pickle_files(
    input_directory="processed_files/",
    output_file="combined_data.pkl"
)
```

### 統計分析

```python
from tpext.preprocess import analyze_processed_data

# データの統計分析を実行
results = analyze_processed_data(
    file_path="processed_data.pkl",
    text_column="processedtext",
    display_results=True
)

# 分析結果の確認
print(f"総データ数: {results['basic_statistics']['total_posts']:,}")
print(f"平均単語数: {results['basic_statistics']['mean_word_count']:.2f}")
```

### データ閲覧

```python
from tpext.preprocess import view_data

# Pickleファイルの内容を確認
data = view_data("processed_data.pkl", n_samples=10)
```

### CSV検査

```python
from tpext.preprocess import inspect_csv

# CSVファイルの詳細分析
inspection_result = inspect_csv("raw_data.csv")
```

## コマンドライン使用例

各モジュールはスタンドアロンスクリプトとしても実行できます：

### 日本語テキスト処理

```bash
python -m tpext.preprocess.japanese_text_processor \
    --input-file data.csv \
    --text-column content \
    --output-dir output/ \
    --chunk-size 1000
```

### 英語テキスト処理

```bash
python -m tpext.preprocess.english_text_processor \
    --input data.csv \
    --text-column text \
    --output processed_english.pkl
```

### 統計分析

```bash
python -m tpext.preprocess.data_statistics \
    --file processed_data.pkl \
    --text-column processedtext \
    --log-level INFO
```

### データ閲覧

```bash
python -m tpext.preprocess.data_viewer \
    --file processed_data.pkl \
    --samples 20 \
    --columns processedtext,original_text
```

### CSV検査

```bash
python -m tpext.preprocess.csv_inspector \
    --directory data/ \
    --output-file csv_analysis.json
```

## 設定とオプション

### パフォーマンス設定

- `pandarallel`を使用した並列処理の有効化
- チャンクサイズの調整によるメモリ使用量の最適化
- ログレベルの設定（DEBUG, INFO, WARNING, ERROR）

### カスタマイズ

各処理クラスは以下のパラメータでカスタマイズ可能です：

- **JapaneseTextProcessor**: 除外する品詞、最小/最大単語長
- **EnglishTextProcessor**: ストップワードリスト、品詞フィルタ
- **DataStatisticsAnalyzer**: 分析対象カラム、パーセンタイル設定

## ファイル形式

### サポートされる入力形式
- CSV (.csv)
- Pickle (.pkl)

### 出力形式
- Pickle (.pkl) - 処理済みデータ
- JSON (.json) - 統計レポート
- CSV (.csv) - 結合データ（オプション）

## エラーハンドリング

- ファイル存在チェック
- データ形式の検証
- メモリ不足時の警告
- 処理中断時の自動保存