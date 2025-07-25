import pandas as pd
import numpy as np

try:
    # pickleファイルからDataFrameを読み込みます
    # 8000万行の場合、メモリ使用量に応じて時間がかかる可能性があります
    print("DataFrameを読み込んでいます...")
    df = pd.read_pickle('/mnt/ExtreamSSD/ja_proc.pkl')
    print("読み込みが完了しました。")

    # 'processedtext'カラムのリストの長さを計算し、'word_count'カラムを作成します
    print("各投稿の単語数を計算しています...")
    df['word_count'] = df['processedtext'].apply(len)
    print("計算が完了しました。")

    # 平均単語数を計算します
    mean_word_count = df['word_count'].mean()
    # 単語数の標準偏差を計算します
    std_dev = df['word_count'].std()

    # 単語数が0の投稿の数を数えます
    zero_word_posts = (df['word_count'] == 0).sum()

    # 全投稿数を取得します
    total_posts = len(df)

    # 単語数が0の投稿の割合を計算します
    zero_word_post_ratio = (zero_word_posts / total_posts) * 100 if total_posts > 0 else 0

    # --- 結果の表示 ---
    print("\n--- 統計情報 ---")
    print(f"総ポスト数: {total_posts:,}")
    print(f"単語数の標準偏差: {std_dev:.2f}")
    print(f"平均単語数: {mean_word_count:.2f}")
    print(f"単語数が0のポスト数: {zero_word_posts:,}")
    print(f"単語数が0のポストの割合: {zero_word_post_ratio:.2f}%\n")
    
    print("--- 各投稿の単語数（データの先頭5件） ---")
    print(df[['processedtext', 'word_count']].head())


except FileNotFoundError:
    print("\nエラー: ファイルが見つかりません。ファイルをアップロードしてください。")
except Exception as e:
    print(f"\n処理中に予期せぬエラーが発生しました: {e}")