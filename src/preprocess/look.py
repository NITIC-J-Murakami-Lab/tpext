import pickle
import pandas as pd

# 確認したいpklファイルのパス
file_path = 'processed_pkls_en/english_posts_1_processed.pkl' 

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# データがpandasのDataFrameの場合
if isinstance(data, pd.DataFrame):
    print("DataFrameの先頭5行:")
    # .head()で先頭部分を表示
    print(data.head())
else:
    print("データはDataFrameではありません。")