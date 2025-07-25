"""CSVファイルの検査・閲覧モジュール

このモジュールはCSVファイルの内容を詳細に調査し、
データ品質の確認や前処理前の分析を支援する機能を提供する。

主な機能:
- CSVファイルの詳細分析
- ディレクトリ内のCSV検索
- データ品質チェック
- カラム情報の詳細表示

Classes:
    CSVInspector: CSVファイルの検査・分析を行うクラス
"""
from typing import List, Dict, Any, Union, Optional
import pandas as pd
import os
import glob
from pathlib import Path
import argparse
import logging
import sys


# 公開API（import時に見えるもの）
__all__ = [
    'CSVInspector',
    'inspect_csv',
    'find_csv_files'
]

# ログ設定（デフォルト）
logger = logging.getLogger(__name__)


class CSVInspector:
    """CSVファイルの検査・分析を行うクラス
    
    Attributes:
        file_path (Optional[Path]): 対象CSVファイルのパス
        df (Optional[pd.DataFrame]): 読み込まれたDataFrame
    """
    
    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        """CSVInspectorを初期化する
        
        Args:
            file_path (Optional[Union[str, Path]], optional): CSVファイルのパス. Defaults to None.
        """
        self.file_path = Path(file_path) if file_path else None
        self.df: Optional[pd.DataFrame] = None
    
    def load_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """CSVファイルを読み込む
        
        Args:
            file_path (Union[str, Path]): CSVファイルのパス
            **kwargs: pd.read_csvに渡す追加引数
            
        Returns:
            pd.DataFrame: 読み込まれたDataFrame
            
        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: CSVファイルの読み込みに失敗した場合
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        
        if not file_path.suffix.lower() == '.csv':
            raise ValueError(f"CSVファイルではありません: {file_path}")
        
        try:
            logger.info(f"CSVファイルを読み込み中: {file_path}")
            self.file_path = file_path
            self.df = pd.read_csv(file_path, **kwargs)
            logger.info(f"読み込み完了: {len(self.df)} 行")
            return self.df
            
        except Exception as e:
            logger.error(f"CSVファイル読み込みエラー: {e}")
            raise ValueError(f"CSVファイルの読み込みに失敗しました: {e}")
    
    def get_basic_info(self) -> Dict[str, Any]:
        """基本情報を取得する
        
        Returns:
            Dict[str, Any]: 基本情報の辞書
        """
        if self.df is None:
            return {"error": "データが読み込まれていません"}
        
        file_size = self.file_path.stat().st_size / (1024 * 1024) if self.file_path else 0
        
        return {
            "file_path": str(self.file_path),
            "file_size_mb": round(file_size, 2),
            "shape": self.df.shape,
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
    
    def get_column_info(self) -> pd.DataFrame:
        """カラム情報の詳細を取得する
        
        Returns:
            pd.DataFrame: カラム情報のDataFrame
        """
        if self.df is None:
            return pd.DataFrame({"error": ["データが読み込まれていません"]})
        
        column_info = []
        for col in self.df.columns:
            info = {
                "カラム名": col,
                "データ型": str(self.df[col].dtype),
                "非null数": self.df[col].notna().sum(),
                "null数": self.df[col].isna().sum(),
                "null率(%)": round(self.df[col].isna().mean() * 100, 2),
                "ユニーク数": self.df[col].nunique(),
                "メモリ使用量(KB)": round(self.df[col].memory_usage(deep=True) / 1024, 2)
            }
            
            # 数値型の場合は統計情報を追加
            if pd.api.types.is_numeric_dtype(self.df[col]):
                info.update({
                    "最小値": self.df[col].min(),
                    "最大値": self.df[col].max(),
                    "平均値": round(self.df[col].mean(), 2),
                    "標準偏差": round(self.df[col].std(), 2)
                })
            
            # 文字列型の場合は長さ情報を追加
            elif pd.api.types.is_string_dtype(self.df[col]) or self.df[col].dtype == 'object':
                text_lengths = self.df[col].astype(str).str.len()
                info.update({
                    "最短文字数": text_lengths.min(),
                    "最長文字数": text_lengths.max(),
                    "平均文字数": round(text_lengths.mean(), 2)
                })
            
            column_info.append(info)
        
        return pd.DataFrame(column_info)
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """データ品質レポートを生成する
        
        Returns:
            Dict[str, Any]: データ品質レポート
        """
        if self.df is None:
            return {"error": "データが読み込まれていません"}
        
        # 重複行の確認
        duplicate_rows = self.df.duplicated().sum()
        
        # 完全に空の行の確認
        empty_rows = self.df.isnull().all(axis=1).sum()
        
        # カラムごとのnull率
        null_rates = self.df.isnull().mean() * 100
        high_null_columns = null_rates[null_rates > 50].index.tolist()
        
        # 数値カラムの異常値確認（IQR方式）
        numeric_columns = self.df.select_dtypes(include=['number']).columns
        outlier_info = {}
        
        for col in numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outlier_info[col] = {
                "外れ値数": outliers,
                "外れ値率(%)": round(outliers / len(self.df) * 100, 2),
                "下限": round(lower_bound, 2),
                "上限": round(upper_bound, 2)
            }
        
        return {
            "総行数": len(self.df),
            "重複行数": duplicate_rows,
            "重複率(%)": round(duplicate_rows / len(self.df) * 100, 2),
            "完全空行数": empty_rows,
            "高null率カラム(50%以上)": high_null_columns,
            "数値カラム外れ値情報": outlier_info
        }
    
    def display_sample_data(self, head_rows: int = 10, tail_rows: int = 5) -> None:
        """サンプルデータを表示する
        
        Args:
            head_rows (int, optional): 先頭行数. Defaults to 10.
            tail_rows (int, optional): 末尾行数. Defaults to 5.
        """
        if self.df is None:
            print("エラー: データが読み込まれていません")
            return
        
        print(f"\\n=== 先頭{head_rows}行のデータ ===")
        print(self.df.head(head_rows).to_string())
        
        if len(self.df) > head_rows:
            print(f"\\n=== 末尾{tail_rows}行のデータ ===")
            print(self.df.tail(tail_rows).to_string())
    
    def display_statistics(self) -> None:
        """統計情報を表示する"""
        if self.df is None:
            print("エラー: データが読み込まれていません")
            return
        
        print("\\n=== 数値カラムの基本統計 ===")
        numeric_df = self.df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 0:
            print(numeric_df.describe().to_string())
        else:
            print("数値カラムがありません")
    
    def search_in_data(self, pattern: str, column: Optional[str] = None, case_sensitive: bool = False) -> pd.DataFrame:
        """データ内を検索する
        
        Args:
            pattern (str): 検索パターン
            column (Optional[str], optional): 特定カラムでの検索. Defaults to None.
            case_sensitive (bool, optional): 大文字小文字を区別するか. Defaults to False.
            
        Returns:
            pd.DataFrame: 検索結果
        """
        if self.df is None:
            return pd.DataFrame({"error": ["データが読み込まれていません"]})
        
        if column and column not in self.df.columns:
            return pd.DataFrame({"error": [f"カラム '{column}' が見つかりません"]})
        
        try:
            if column:
                # 特定カラムでの検索
                mask = self.df[column].astype(str).str.contains(
                    pattern, case=case_sensitive, na=False
                )
            else:
                # 全カラムでの検索
                mask = self.df.astype(str).apply(
                    lambda x: x.str.contains(pattern, case=case_sensitive, na=False)
                ).any(axis=1)
            
            return self.df[mask]
            
        except Exception as e:
            logger.error(f"検索エラー: {e}")
            return pd.DataFrame({"error": [f"検索エラー: {e}"]})
    
    def inspect_full_csv(
        self,
        file_path: Union[str, Path],
        head_rows: int = 10,
        tail_rows: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """CSVファイルの完全検査を実行する
        
        Args:
            file_path (Union[str, Path]): CSVファイルのパス
            head_rows (int, optional): 表示する先頭行数. Defaults to 10.
            tail_rows (int, optional): 表示する末尾行数. Defaults to 5.
            **kwargs: pd.read_csvに渡す追加引数
            
        Returns:
            Dict[str, Any]: 検査結果の辞書
        """
        # CSVを読み込み
        self.load_csv(file_path, **kwargs)
        
        # 基本情報を取得
        basic_info = self.get_basic_info()
        column_info = self.get_column_info()
        quality_report = self.get_data_quality_report()
        
        # 情報を表示
        print("\\n=== CSVファイル検査結果 ===")
        print(f"ファイル: {basic_info['file_path']}")
        print(f"ファイルサイズ: {basic_info['file_size_mb']} MB")
        print(f"形状: {basic_info['shape']}")
        print(f"メモリ使用量: {basic_info['memory_usage_mb']} MB")
        
        print("\\n=== カラム情報 ===")
        print(column_info.to_string(index=False))
        
        print("\\n=== データ品質レポート ===")
        for key, value in quality_report.items():
            if key != "数値カラム外れ値情報":
                print(f"{key}: {value}")
        
        if quality_report.get("数値カラム外れ値情報"):
            print("\\n=== 外れ値情報 ===")
            for col, info in quality_report["数値カラム外れ値情報"].items():
                print(f"{col}: {info}")
        
        # サンプルデータを表示
        self.display_sample_data(head_rows, tail_rows)
        
        # 統計情報を表示
        self.display_statistics()
        
        return {
            "basic_info": basic_info,
            "column_info": column_info,
            "quality_report": quality_report
        }


def find_csv_files(directory: Union[str, Path] = ".") -> List[Path]:
    """指定されたディレクトリ内のCSVファイルを検索する
    
    Args:
        directory (Union[str, Path], optional): 検索ディレクトリ. Defaults to ".".
        
    Returns:
        List[Path]: 見つかったCSVファイルのリスト
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"ディレクトリが見つかりません: {directory}")
        return []
    
    # 再帰的にCSVファイルを検索
    csv_files = list(directory.rglob("*.csv"))
    
    if csv_files:
        print(f"\\n=== 見つかったCSVファイル ({len(csv_files)}件) ===")
        for i, file_path in enumerate(csv_files, 1):
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"{i:3d}. {file_path} ({file_size:.2f} MB)")
    else:
        print(f"ディレクトリ '{directory}' 内にCSVファイルが見つかりませんでした")
    
    return csv_files


def inspect_csv(
    file_path: Union[str, Path],
    head_rows: int = 10,
    tail_rows: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """CSVファイルを検査する便利関数
    
    Args:
        file_path (Union[str, Path]): CSVファイルのパス
        head_rows (int, optional): 表示する先頭行数. Defaults to 10.
        tail_rows (int, optional): 表示する末尾行数. Defaults to 5.
        **kwargs: pd.read_csvに渡す追加引数
        
    Returns:
        Dict[str, Any]: 検査結果
    """
    inspector = CSVInspector()
    return inspector.inspect_full_csv(file_path, head_rows, tail_rows, **kwargs)


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
        description='CSVファイル検査ツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # CSVファイルを検査
  python csv_inspector.py -f data.csv

  # ディレクトリ内のCSVファイルを検索
  python csv_inspector.py --find-csv-dir /path/to/directory

  # 表示行数を指定して検査
  python csv_inspector.py -f data.csv --head-rows 20 --tail-rows 10

  # データ内検索機能付きで検査
  python csv_inspector.py -f data.csv --search-pattern "error" --search-column "message"

  # 詳細ログで実行
  python csv_inspector.py -f data.csv --log-level DEBUG
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-f', '--file',
        type=str,
        help='検査対象のCSVファイルパス'
    )
    
    group.add_argument(
        '--find-csv-dir',
        type=str,
        help='CSVファイルを検索するディレクトリパス'
    )
    
    parser.add_argument(
        '--head-rows',
        type=int,
        default=10,
        help='表示する先頭行数 (デフォルト: 10)'
    )
    
    parser.add_argument(
        '--tail-rows',
        type=int,
        default=5,
        help='表示する末尾行数 (デフォルト: 5)'
    )
    
    parser.add_argument(
        '--search-pattern',
        type=str,
        help='データ内検索パターン'
    )
    
    parser.add_argument(
        '--search-column',
        type=str,
        help='検索対象カラム（指定しない場合は全カラム）'
    )
    
    parser.add_argument(
        '--case-sensitive',
        action='store_true',
        help='大文字小文字を区別して検索'
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
        if args.find_csv_dir:
            # CSVファイル検索モード
            find_csv_files(args.find_csv_dir)
            
        else:
            # CSVファイル検査モード
            inspector = CSVInspector()
            
            # CSVを検査
            result = inspector.inspect_full_csv(
                file_path=args.file,
                head_rows=args.head_rows,
                tail_rows=args.tail_rows
            )
            
            # データ内検索が指定されている場合
            if args.search_pattern:
                print(f"\\n=== 検索結果 (パターン: '{args.search_pattern}') ===")
                search_result = inspector.search_in_data(
                    pattern=args.search_pattern,
                    column=args.search_column,
                    case_sensitive=args.case_sensitive
                )
                
                if len(search_result) > 0:
                    print(f"見つかった行数: {len(search_result)}")
                    print(search_result.head(10).to_string())
                else:
                    print("一致する行が見つかりませんでした")
        
        logger.info("CSV検査が正常に完了しました")
        
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