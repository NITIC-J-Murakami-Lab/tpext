import pandas as pd
import spacy
from pandarallel import pandarallel
import re
import os
import glob

# --- 1. 初期設定 ---
# pandarallelの初期化
pandarallel.initialize(progress_bar=True)

# spaCyの英語モデルをロードするためのグローバル変数
_nlp = None

def lemmatize_english_text(text):
    """
    英語のテキストをクリーニングし、主要な単語をレンマ化（見出し語化）する関数。
    (この関数の内容は変更ありません)
    """
    global _nlp
    if _nlp is None:
        _nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    lemmatized_text = []
    if isinstance(text, str):
        text = text.lower()
        cleaned_text = re.sub(r'https?://[^\s]+', '', text)
        cleaned_text = re.sub(r'@[a-zA-Z0-9_]+', '', cleaned_text)
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', cleaned_text)
        doc = _nlp(cleaned_text)
        lemmatized_text = [
            token.lemma_ for token in doc 
            if not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']
        ]
    return lemmatized_text

# --- 2. メイン処理 ---

# ★★★ 変更点: 入力と出力のディレクトリを定義 ★★★
input_csv_directory = '/mnt/ExtreamSSD/en_post'  # <<<< 実際のCSVフォルダパスに変更
output_pkl_directory = 'processed_pkls_en'       # <<<< 出力先のフォルダ名

print(f"入力ディレクトリ: {input_csv_directory}")
print(f"出力ディレクトリ: {output_pkl_directory}")

# 出力ディレクトリがなければ作成
os.makedirs(output_pkl_directory, exist_ok=True)

# 入力ディレクトリ内の全CSVファイルのリストを取得
csv_files = glob.glob(os.path.join(input_csv_directory, '*.csv'))

if not csv_files:
    print(f"エラー: ディレクトリ '{input_csv_directory}' 内にCSVファイルが見つかりません。")
    exit()

print(f"{len(csv_files)}個のCSVファイルを処理対象として検出しました。")

# ★★★ 変更点: メインループをファイルごとに行うように変更 ★★★
# DataFrameを結合せず、1ファイルずつループ処理する
for i, csv_path in enumerate(csv_files):
    file_name = os.path.basename(csv_path)
    print(f"\n--- ファイル {i+1}/{len(csv_files)} の処理を開始: {file_name} ---")

    # 出力ファイルパスを生成 (例: 'my_data.csv' -> 'my_data_processed.pkl')
    output_base_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(output_pkl_directory, f"{output_base_name}_processed.pkl")

    # 既に処理済みのファイルが存在するかチェック
    if os.path.exists(output_path):
        print(f"出力ファイルが既に存在するため、このファイルはスキップします。 ({output_path})")
        continue

    try:
        # CSVファイルを1つだけ読み込む
        df = pd.read_csv(csv_path)

        if 'text' not in df.columns:
            print(f"警告: 'text' カラムが見つかりません。このファイルをスキップします。")
            continue

        # 'text' カラムに対して並列処理でレンマ化を適用
        df['processed_text'] = df['text'].parallel_apply(lemmatize_english_text)

        # 処理済みのデータフレームを単一のPKLファイルとして保存
        df.to_pickle(output_path)
        print(f"処理が完了し、{output_path} に保存しました。")

    except Exception as e:
        print(f"ファイル '{file_name}' の処理中にエラーが発生しました: {e}")
        # エラーが発生しても次のファイルの処理に進む
        continue

print("\n全てのファイルの処理が完了しました。")