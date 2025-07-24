#チャンクごとに処理を行うスクリプト
import pandas as pd
from janome.tokenizer import Tokenizer
from pandarallel import pandarallel
import numpy as np
import re
import glob 
import os   

pandarallel.initialize(progress_bar=True)
_tokenizer = None

def extract_nouns(text):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer()
    
    newtext = []
    if isinstance(text, str):
        # 1. URLとメンションを削除
        cleaned_text = re.sub(r'https?://[^\s]+', '', text)
        cleaned_text = re.sub(r'@[a-zA-Z0-9_]+', '', cleaned_text)
        
        # 2. 日本語、英数字、空白以外の文字（記号、絵文字など）を全て削除
        cleaned_text = re.sub(r'[^ぁ-んァ-ヶ一-龥a-zA-Z0-9\sー]', '', cleaned_text)
        
        # 3. 形態素解析
        for token in _tokenizer.tokenize(cleaned_text):
            part_of_speech = token.part_of_speech.split(',')[0]
            if part_of_speech == '名詞':
                newtext.append(token.surface)
    return newtext
    
# --- メインの処理（チャンク保存方法を修正） ---
print("pklファイルの読み込みを開始します...")
try:
    df = pd.read_pickle('/mnt/ExtreamSSD/ja_df.pkl')
except FileNotFoundError:
    print("エラー: pklファイルが見つかりません。")
    exit()

print(f"合計 {len(df)} 行のデータを処理します。")

chunk_size = 500000
# --- 変更点：中間ファイルを保存するディレクトリを指定 ---
output_dir = 'processed_chunks'
final_output_path = 'ja_df_processed.pkl'

# 中間ディレクトリがなければ作成
os.makedirs(output_dir, exist_ok=True) 

for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
    chunk_end = chunk_start + chunk_size
    chunk_df = df.iloc[chunk_start:chunk_end]

    print(f"\n--- チャンク {i+1}/{len(df) // chunk_size + 1} の処理を開始 ---")

    processed_series = chunk_df['text'].parallel_apply(extract_nouns)
    chunk_df['processedtext'] = processed_series
    
    processed_chunk = chunk_df.drop('text', axis=1)
    
    chunk_output_path = os.path.join(output_dir, f'chunk_{i}.pkl')
    processed_chunk.to_pickle(chunk_output_path)
    print(f"チャンク {i+1} を {chunk_output_path} に保存しました。")

print("\n全てのチャンク処理が完了しました。")
