"""処理済みデータファイルの閲覧・確認モジュール

このモジュールはpklファイルやCSVファイルの内容を簡単に確認するための
機能を提供する。データの概要把握やデバッグに有用。

主な機能:
- pklファイルの内容確認
- データ構造の表示
- 基本統計情報の表示
- サンプルデータの表示

Classes:
    DataViewer: データファイルの閲覧機能を提供するクラス
"""
from typing import Union, Any, Optional, Dict
import pandas as pd
import pickle
from pathlib import Path
import argparse
import logging
import sys


# 公開API（import時に見えるもの）
__all__ = [
    'DataViewer',
    'view_data'
]

# ログ設定（デフォルト）
logger = logging.getLogger(__name__)


class DataViewer:
    """データファイルの閲覧・確認を行うクラス
    
    Attributes:
        file_path (Optional[Path]): 対象ファイルのパス
        data (Any): 読み込まれたデータ
    """
    
    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        """DataViewerを初期化する
        
        Args:
            file_path (Optional[Union[str, Path]], optional): データファイルのパス. Defaults to None.
        """
        self.file_path = Path(file_path) if file_path else None
        self.data: Any = None
    
    def load_data(self, file_path: Union[str, Path]) -> Any:
        """データファイルを読み込む
        
        Args:
            file_path (Union[str, Path]): 読み込み対象のファイルパス
            
        Returns:
            Any: 読み込まれたデータ
            
        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: ファイル形式が対応していない場合
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        
        self.file_path = file_path
        
        try:
            if file_path.suffix.lower() == '.pkl':
                logger.info(f"pklファイルを読み込み中: {file_path}")
                with open(file_path, 'rb') as f:
                    self.data = pickle.load(f)
                    
            elif file_path.suffix.lower() == '.csv':
                logger.info(f"CSVファイルを読み込み中: {file_path}")
                self.data = pd.read_csv(file_path)
                
            else:
                raise ValueError(f"対応していないファイル形式: {file_path.suffix}")
            
            logger.info("読み込み完了")
            return self.data
            
        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {e}")
            raise
    
    def get_data_type_info(self) -> Dict[str, Any]:
        """データの型情報を取得する
        
        Returns:
            Dict[str, Any]: データの型情報
        """
        if self.data is None:
            return {"error": "データが読み込まれていません"}
        
        info = {
            "data_type": type(self.data).__name__,
            "module": type(self.data).__module__
        }
        
        if isinstance(self.data, pd.DataFrame):
            info.update({
                "shape": self.data.shape,
                "columns": list(self.data.columns),
                "dtypes": dict(self.data.dtypes),
                "memory_usage": f"{self.data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            })
        elif isinstance(self.data, (list, tuple)):
            info.update({
                "length": len(self.data),
                "sample_types": [type(item).__name__ for item in self.data[:5]]
            })
        elif isinstance(self.data, dict):
            info.update({
                "keys": list(self.data.keys()),
                "key_count": len(self.data)
            })
        
        return info
    
    def display_basic_info(self) -> None:
        """基本情報を表示する"""
        if self.data is None:
            print("エラー: データが読み込まれていません")
            return
        
        print(f"\\n=== ファイル情報 ===")
        print(f"ファイルパス: {self.file_path}")
        print(f"ファイルサイズ: {self.file_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        print(f"\\n=== データ型情報 ===")
        type_info = self.get_data_type_info()
        for key, value in type_info.items():
            print(f"{key}: {value}")
    
    def display_dataframe_info(self, head_rows: int = 5) -> None:
        """DataFrameの詳細情報を表示する
        
        Args:
            head_rows (int, optional): 表示する先頭行数. Defaults to 5.
        """
        if not isinstance(self.data, pd.DataFrame):
            print("データはDataFrameではありません")
            return
        
        df = self.data
        
        print(f"\\n=== DataFrame概要 ===")
        print(f"形状: {df.shape}")
        print(f"カラム数: {len(df.columns)}")
        print(f"行数: {len(df)}")
        
        print(f"\\n=== カラム情報 ===")
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            null_count = df[col].isna().sum()
            dtype = df[col].dtype
            print(f"  {col}: {dtype} (非null: {non_null_count:,}, null: {null_count:,})")
        
        print(f"\\n=== 先頭{head_rows}行のデータ ===")
        print(df.head(head_rows).to_string())
        
        # 数値カラムがある場合は基本統計情報を表示
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(f"\\n=== 数値カラムの基本統計 ===")
            print(df[numeric_cols].describe().to_string())
    
    def display_sample_data(self, sample_size: int = 10) -> None:
        """サンプルデータを表示する
        
        Args:
            sample_size (int, optional): サンプルサイズ. Defaults to 10.
        """
        if self.data is None:
            print("エラー: データが読み込まれていません")
            return
        
        print(f"\\n=== サンプルデータ (最大{sample_size}件) ===")
        
        if isinstance(self.data, pd.DataFrame):
            sample_df = self.data.head(sample_size)
            print(sample_df.to_string())
            
        elif isinstance(self.data, (list, tuple)):
            for i, item in enumerate(self.data[:sample_size]):
                print(f"[{i}]: {item}")
                
        elif isinstance(self.data, dict):
            items = list(self.data.items())[:sample_size]
            for key, value in items:
                print(f"{key}: {value}")
                
        else:
            print(f"データ: {self.data}")
    
    def search_columns(self, pattern: str) -> None:
        """カラム名を検索する（DataFrameの場合）
        
        Args:
            pattern (str): 検索パターン
        """
        if not isinstance(self.data, pd.DataFrame):
            print("データはDataFrameではありません")
            return
        
        matching_cols = [col for col in self.data.columns if pattern.lower() in col.lower()]
        
        if matching_cols:
            print(f"\\n=== パターン '{pattern}' に一致するカラム ===")
            for col in matching_cols:
                print(f"  {col}")
        else:
            print(f"パターン '{pattern}' に一致するカラムが見つかりません")
    
    def view_full_data(
        self, 
        file_path: Union[str, Path],
        head_rows: int = 5,
        sample_size: int = 10
    ) -> None:
        """データファイルの完全な閲覧を実行する
        
        Args:
            file_path (Union[str, Path]): データファイルのパス
            head_rows (int, optional): DataFrameの表示行数. Defaults to 5.
            sample_size (int, optional): サンプルデータサイズ. Defaults to 10.
        """
        self.load_data(file_path)
        self.display_basic_info()
        
        if isinstance(self.data, pd.DataFrame):
            self.display_dataframe_info(head_rows)
        else:
            self.display_sample_data(sample_size)


def view_data(
    file_path: Union[str, Path],
    head_rows: int = 5,
    sample_size: int = 10
) -> Any:
    """データファイルを閲覧する便利関数
    
    Args:
        file_path (Union[str, Path]): データファイルのパス
        head_rows (int, optional): DataFrameの表示行数. Defaults to 5.
        sample_size (int, optional): サンプルデータサイズ. Defaults to 10.
        
    Returns:
        Any: 読み込まれたデータ
    """
    viewer = DataViewer()
    viewer.view_full_data(file_path, head_rows, sample_size)
    return viewer.data


def setup_logging(level: str) -> None:
    """ログレベルを設定する
    
    Args:
        level (str): ログレベル (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'無効なログレベル: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True  # 既存の設定を上書き
    )


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する
    
    Returns:
        argparse.Namespace: 解析された引数
    """
    parser = argparse.ArgumentParser(
        description='データファイル閲覧ツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # pklファイルを閲覧
  python data_viewer.py -f data.pkl

  # CSVファイルを閲覧（表示行数指定）
  python data_viewer.py -f data.csv --head-rows 10

  # カラム検索機能付きで閲覧
  python data_viewer.py -f data.pkl --search-pattern text

  # 詳細ログで実行
  python data_viewer.py -f data.pkl --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        required=True,
        help='閲覧対象のデータファイルパス'
    )
    
    parser.add_argument(
        '--head-rows',
        type=int,
        default=5,
        help='DataFrameの表示行数 (デフォルト: 5)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=10,
        help='サンプルデータサイズ (デフォルト: 10)'
    )
    
    parser.add_argument(
        '--search-pattern',
        type=str,
        help='カラム名検索パターン（DataFrameの場合）'
    )
    
    parser.add_argument(
        '-l', '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='ログレベル (デフォルト: INFO)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser.parse_args()


def main() -> None:
    """メイン処理関数"""
    # コマンドライン引数を解析
    args = parse_args()
    
    # ログレベルを設定
    setup_logging(args.log_level)
    
    try:
        # データビューアを初期化
        viewer = DataViewer()
        
        # データを読み込んで表示
        viewer.view_full_data(
            file_path=args.file,
            head_rows=args.head_rows,
            sample_size=args.sample_size
        )
        
        # カラム検索が指定されている場合
        if args.search_pattern:
            viewer.search_columns(args.search_pattern)
        
        logger.info("データ閲覧が正常に完了しました")
        
    except FileNotFoundError as e:
        logger.error(f"ファイルエラー: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"設定エラー: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("処理が中断されました")
        sys.exit(1)
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()