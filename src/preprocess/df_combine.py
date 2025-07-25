"""日本語テキスト処理済みチャンクファイルの結合モジュール

このモジュールは、分割処理された複数のpklファイルを効率的に結合する機能を提供する。
メモリ使用量を考慮した安全な結合処理を実装している。

主な機能:
- 複数のpklファイルの自動検出と結合
- メモリ効率を考慮した段階的な結合処理
- エラーハンドリングとログ出力
- 結合後のデータ統計情報の表示

Classes:
    DataFrameCombiner: DataFrameの結合処理を行うクラス
"""
from typing import List, Optional, Union
import pandas as pd
import glob
import os
from pathlib import Path
import logging
import argparse
import sys


# 公開API（import時に見えるもの）
__all__ = [
    'DataFrameCombiner',
    'combine_pickle_files'
]

# ログ設定（デフォルト）
logger = logging.getLogger(__name__)


class DataFrameCombiner:
    """複数のpklファイルを結合するクラス
    
    Attributes:
        input_dir (Path): 入力ディレクトリのパス
        output_file (Path): 出力ファイルのパス
        file_pattern (str): 対象ファイルのパターン
    """
    
    def __init__(
        self, 
        input_dir: Union[str, Path] = 'processed_chunks',
        output_file: Union[str, Path] = 'processed_ja.pkl',
        file_pattern: str = '*.pkl',
        skip_validation: bool = False
    ):
        """DataFrameCombinerを初期化する
        
        Args:
            input_dir (Union[str, Path], optional): 入力ディレクトリ. Defaults to 'processed_chunks'.
            output_file (Union[str, Path], optional): 出力ファイル名. Defaults to 'processed_ja.pkl'.
            file_pattern (str, optional): ファイルパターン. Defaults to '*.pkl'.
            skip_validation (bool, optional): 整合性チェックをスキップ. Defaults to False.
        """
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.file_pattern = file_pattern
        self.skip_validation = skip_validation
        
        # 入力ディレクトリの存在確認
        if not self.input_dir.exists():
            raise FileNotFoundError(f"入力ディレクトリが見つかりません: {self.input_dir}")
    
    def _get_target_files(self) -> List[Path]:
        """対象ファイルのリストを取得する
        
        Returns:
            List[Path]: ソート済みのファイルパスリスト
            
        Raises:
            FileNotFoundError: 対象ファイルが見つからない場合
        """
        file_paths = sorted(self.input_dir.glob(self.file_pattern))
        
        if not file_paths:
            raise FileNotFoundError(
                f"対象ファイルが見つかりません: {self.input_dir}/{self.file_pattern}"
            )
        
        logger.info(f"{len(file_paths)} 個のファイルが見つかりました")
        return file_paths
    
    def _load_single_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """単一のpklファイルを読み込む
        
        Args:
            file_path (Path): 読み込み対象のファイルパス
            
        Returns:
            Optional[pd.DataFrame]: 読み込まれたDataFrame、失敗時はNone
        """
        try:
            df = pd.read_pickle(file_path)
            logger.info(f"読み込み成功: {file_path.name} ({len(df):,} 行)")
            return df
        except Exception as e:
            logger.error(f"読み込み失敗: {file_path.name} - {e}")
            return None
    
    def _validate_dataframes(self, dataframes: List[pd.DataFrame]) -> bool:
        """DataFrameリストの整合性を確認する
        
        Args:
            dataframes (List[pd.DataFrame]): 確認対象のDataFrameリスト
            
        Returns:
            bool: 整合性に問題がなければTrue
        """
        if not dataframes:
            logger.error("結合対象のDataFrameが存在しません")
            return False
        
        # カラム名の一致確認
        first_columns = set(dataframes[0].columns)
        for i, df in enumerate(dataframes[1:], 1):
            if set(df.columns) != first_columns:
                logger.warning(
                    f"DataFrame {i} のカラムが異なります: "
                    f"期待値={first_columns}, 実際={set(df.columns)}"
                )
        
        # データ型の確認
        first_dtypes = dataframes[0].dtypes
        for i, df in enumerate(dataframes[1:], 1):
            if not df.dtypes.equals(first_dtypes):
                logger.warning(f"DataFrame {i} のデータ型が異なります")
        
        return True
    
    def _combine_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """DataFrameリストを結合する
        
        Args:
            dataframes (List[pd.DataFrame]): 結合対象のDataFrameリスト
            
        Returns:
            pd.DataFrame: 結合されたDataFrame
        """
        logger.info("DataFrameの結合を開始します...")
        
        try:
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"結合完了: {len(combined_df):,} 行")
            return combined_df
        except Exception as e:
            logger.error(f"DataFrameの結合に失敗しました: {e}")
            raise
    
    def _save_combined_dataframe(self, df: pd.DataFrame) -> None:
        """結合されたDataFrameを保存する
        
        Args:
            df (pd.DataFrame): 保存対象のDataFrame
        """
        try:
            # 出力ディレクトリが存在しない場合は作成
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_pickle(self.output_file)
            logger.info(f"保存完了: {self.output_file}")
            
            # ファイルサイズを表示
            file_size = self.output_file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"ファイルサイズ: {file_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"ファイルの保存に失敗しました: {e}")
            raise
    
    def _display_statistics(self, df: pd.DataFrame) -> None:
        """結合後のDataFrameの統計情報を表示する
        
        Args:
            df (pd.DataFrame): 統計情報を表示するDataFrame
        """
        logger.info("=== 結合後のDataFrame統計情報 ===")
        logger.info(f"行数: {len(df):,}")
        logger.info(f"列数: {len(df.columns)}")
        logger.info(f"カラム名: {list(df.columns)}")
        
        # メモリ使用量を表示
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        logger.info(f"メモリ使用量: {memory_usage:.2f} MB")
        
        # 各カラムの情報
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            logger.info(f"  {col}: {non_null_count:,} 非null値 ({df[col].dtype})")
    
    def combine_files(self) -> pd.DataFrame:
        """複数のpklファイルを結合する
        
        Returns:
            pd.DataFrame: 結合されたDataFrame
            
        Raises:
            FileNotFoundError: 対象ファイルが見つからない場合
            ValueError: DataFrameの結合に失敗した場合
        """
        logger.info("ファイル結合処理を開始します...")
        
        # 対象ファイルを取得
        file_paths = self._get_target_files()
        
        # ファイルを順次読み込み
        dataframes: List[pd.DataFrame] = []
        failed_files: List[Path] = []
        
        for file_path in file_paths:
            df = self._load_single_file(file_path)
            if df is not None:
                dataframes.append(df)
            else:
                failed_files.append(file_path)
        
        # 失敗したファイルがある場合は警告
        if failed_files:
            logger.warning(f"{len(failed_files)} 個のファイルの読み込みに失敗しました")
            for failed_file in failed_files:
                logger.warning(f"  失敗ファイル: {failed_file}")
        
        # DataFrameの整合性確認（スキップオプションがある場合は実行しない）
        if not self.skip_validation:
            if not self._validate_dataframes(dataframes):
                raise ValueError("DataFrameの整合性チェックに失敗しました")
        else:
            logger.info("整合性チェックをスキップしました")
        
        # DataFrameを結合
        combined_df = self._combine_dataframes(dataframes)
        
        # 結果を保存
        self._save_combined_dataframe(combined_df)
        
        # 統計情報を表示
        self._display_statistics(combined_df)
        
        logger.info("ファイル結合処理が正常に完了しました")
        return combined_df
    
    def get_file_info(self) -> pd.DataFrame:
        """対象ファイルの情報を取得する
        
        Returns:
            pd.DataFrame: ファイル情報のDataFrame
        """
        file_paths = self._get_target_files()
        
        file_info = []
        for file_path in file_paths:
            try:
                # ファイルサイズを取得
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                
                # DataFrameの行数を取得（ヘッダーのみ読み込み）
                df_sample = pd.read_pickle(file_path)
                row_count = len(df_sample)
                
                file_info.append({
                    'ファイル名': file_path.name,
                    'ファイルサイズ(MB)': round(file_size, 2),
                    '行数': row_count,
                    'パス': str(file_path)
                })
                
            except Exception as e:
                file_info.append({
                    'ファイル名': file_path.name,
                    'ファイルサイズ(MB)': 'エラー',
                    '行数': 'エラー',
                    'パス': str(file_path),
                    'エラー': str(e)
                })
        
        return pd.DataFrame(file_info)


def combine_pickle_files(
    input_dir: Union[str, Path] = 'processed_chunks',
    output_file: Union[str, Path] = 'processed_ja.pkl',
    file_pattern: str = '*.pkl',
    skip_validation: bool = False
) -> pd.DataFrame:
    """複数のpklファイルを結合する便利関数
    
    Args:
        input_dir (Union[str, Path], optional): 入力ディレクトリ. Defaults to 'processed_chunks'.
        output_file (Union[str, Path], optional): 出力ファイル名. Defaults to 'processed_ja.pkl'.
        file_pattern (str, optional): ファイルパターン. Defaults to '*.pkl'.
        skip_validation (bool, optional): 整合性チェックをスキップ. Defaults to False.
        
    Returns:
        pd.DataFrame: 結合されたDataFrame
    """
    combiner = DataFrameCombiner(input_dir, output_file, file_pattern, skip_validation)
    return combiner.combine_files()


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
        description='複数のpklファイルを結合するツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で実行
  python processed_data_combiner.py

  # カスタム設定で実行
  python processed_data_combiner.py -i data_chunks -o combined.pkl -p "*.pickle"

  # ファイル情報のみ表示
  python processed_data_combiner.py --info-only

  # 整合性チェックをスキップして実行
  python processed_data_combiner.py --no-validate

  # デバッグモードで実行
  python processed_data_combiner.py --log-level DEBUG

  # モジュールとしての実行例
  python -m tpext.preprocess.processed_data_combiner -i data/ -o result.pkl
        """
    )
    
    # 必須ではない位置引数
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        default='processed_chunks',
        help='入力ディレクトリのパス (デフォルト: processed_chunks)'
    )
    
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        default='processed_ja.pkl',
        help='出力ファイル名 (デフォルト: processed_ja.pkl)'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default='*.pkl',
        help='ファイルパターン (デフォルト: *.pkl)'
    )
    
    parser.add_argument(
        '-l', '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='ログレベル (デフォルト: INFO)'
    )
    
    parser.add_argument(
        '--info-only',
        action='store_true',
        help='ファイル情報のみ表示して終了'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='データ整合性チェックをスキップ'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='統計情報の表示を最小限に抑制'
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
        # DataFrameCombinerを初期化
        combiner = DataFrameCombiner(
            input_dir=args.input_dir,
            output_file=args.output_file,
            file_pattern=args.pattern,
            skip_validation=args.no_validate
        )
        
        # ファイル情報のみの場合
        if args.info_only:
            logger.info("=== 対象ファイル一覧 ===")
            file_info_df = combiner.get_file_info()
            print(file_info_df.to_string(index=False))
            logger.info("ファイル情報の表示が完了しました")
            return
        
        # 通常の処理
        if not args.quiet:
            logger.info("=== 対象ファイル一覧 ===")
            file_info_df = combiner.get_file_info()
            print(file_info_df.to_string(index=False))
        
        # ファイル結合を実行
        combined_df = combiner.combine_files()
        
        logger.info("全ての処理が正常に完了しました")
        
        # 処理結果のサマリを表示
        if not args.quiet:
            logger.info(f"出力ファイル: {args.output_file}")
            logger.info(f"結合された行数: {len(combined_df):,}")
        
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