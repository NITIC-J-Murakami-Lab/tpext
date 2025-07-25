#メモリ使用量を考慮し、別scriptでpklファイルを結合する
import pandas as pd
import glob
import os

# pklファイルが格納されているフォルダのパス
folder_path = 'processed_chunks' 

# 結合後のDataFrameを格納するリスト
list_of_dfs = []

# フォルダ内の全pklファイルのパスを取得
# ファイル名が 'data_01.pkl', 'data_02.pkl' のように連番になっている場合を想定
file_paths = sorted(glob.glob(os.path.join(folder_path, '*.pkl')))

# ファイルを一つずつ読み込み、リストに追加
for path in file_paths:
    try:
        df_part = pd.read_pickle(path)
        list_of_dfs.append(df_part)
        print(f"Successfully loaded and appended: {os.path.basename(path)}")
    except Exception as e:
        print(f"Could not read file {path}: {e}")

# リストに格納された全てのDataFrameを結合
if list_of_dfs:
    combined_df = pd.concat(list_of_dfs, ignore_index=True)
    
    # 結合後のDataFrameを新しいファイルに保存
    combined_df.to_pickle('processed_ja.pkl')

    print("\nAll files have been successfully combined and saved.")
    print("Combined DataFrame Info:")
    combined_df.info(memory_usage='deep')
else:
    print("No dataframes to combine.")