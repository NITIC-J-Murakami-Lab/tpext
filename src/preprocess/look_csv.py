import pandas as pd
import os
import glob

def look_csv(file_path, num_rows=10):
    """
    CSVファイルの最初の数行を表示する関数
    
    Args:
        file_path (str): CSVファイルのパス
        num_rows (int): 表示する行数（デフォルト: 10）
    """
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(file_path)
        
        print(f"ファイル: {file_path}")
        print(f"総行数: {len(df)}")
        print(f"総列数: {len(df.columns)}")
        print(f"列名: {list(df.columns)}")
        print("\n--- 最初の{num_rows}行 ---")
        print(df.head(num_rows))
        print("\n--- データ型情報 ---")
        print(df.dtypes)
        print("\n--- 基本統計情報 ---")
        print(df.describe())
        
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
    except Exception as e:
        print(f"エラー: {e}")

def find_csv_files(directory="."):
    """
    指定されたディレクトリ内のCSVファイルを検索
    
    Args:
        directory (str): 検索するディレクトリ（デフォルト: 現在のディレクトリ）
    """
    csv_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)
    
    if csv_files:
        print("見つかったCSVファイル:")
        for i, file in enumerate(csv_files, 1):
            print(f"{i}. {file}")
        return csv_files
    else:
        print("CSVファイルが見つかりませんでした。")
        return []

if __name__ == "__main__":
    # 特定のCSVファイルパスを指定してください
    csv_file_path = "/mnt/ExtreamSSD/en_post/english_posts_1.csv"  # 例: 実際のCSVファイルパスに変更

    if csv_file_path.strip():
        print(f"\n指定されたCSVファイルを表示します: {csv_file_path}")
        look_csv(csv_file_path.strip(), num_rows=10)
    
    # 特定のファイルを指定したい場合（例）
    # look_csv("path/to/your/file.csv", num_rows=5)
