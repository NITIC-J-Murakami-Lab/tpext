"""処理済みテキストデータの統計情報を分析するモジュール

このモジュールは前処理が完了したテキストデータの品質を評価し、
統計情報を提供する機能を実装している。

主な機能:
- 単語数分析
- データ品質チェック
- 統計情報の表示
- 処理済みデータの概要生成

Classes:
    DataStatisticsAnalyzer: データ統計分析を行うクラス
"""
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import sys


# 公開API（import時に見えるもの）
__all__ = [
    'DataStatisticsAnalyzer',
    'analyze_processed_data'
]

# ログ設定（デフォルト）
logger = logging.getLogger(__name__)


class DataStatisticsAnalyzer:
    """処理済みテキストデータの統計分析を行うクラス
    
    Attributes:
        text_column (str): 分析対象のテキストカラム名
        data_file (Optional[Path]): データファイルのパス
    """
    
    def __init__(
        self,
        text_column: str = 'processedtext',
        data_file: Optional[Union[str, Path]] = None
    ):
        """DataStatisticsAnalyzerを初期化する
        
        Args:
            text_column (str, optional): 分析対象のテキストカラム名. Defaults to 'processedtext'.
            data_file (Optional[Union[str, Path]], optional): データファイルのパス. Defaults to None.
        """
        self.text_column = text_column
        self.data_file = Path(data_file) if data_file else None
        self._df: Optional[pd.DataFrame] = None
    
    def load_data(self, file_path: Union[str, Path]) -> None:
        """データファイルを読み込む
        
        Args:
            file_path (Union[str, Path]): 読み込み対象のファイルパス
            
        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: ファイル形式が対応していない場合
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        
        logger.info(f"データファイルを読み込み中: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.pkl':
                self._df = pd.read_pickle(file_path)
            elif file_path.suffix.lower() == '.csv':
                self._df = pd.read_csv(file_path)
            else:
                raise ValueError(f"対応していないファイル形式: {file_path.suffix}")
            
            logger.info(f"読み込み完了: {len(self._df):,} 行")
            
        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {e}")
            raise
    
    def _validate_data(self) -> None:
        """データの妥当性を確認する
        
        Raises:
            ValueError: データが読み込まれていない、または必要なカラムが存在しない場合
        """
        if self._df is None:
            raise ValueError("データが読み込まれていません。load_data()を先に実行してください。")
        
        if self.text_column not in self._df.columns:
            raise ValueError(f"テキストカラム '{self.text_column}' が見つかりません。")
    
    def calculate_word_counts(self) -> pd.Series:
        """各行の単語数を計算する
        
        Returns:
            pd.Series: 各行の単語数
        """
        self._validate_data()
        
        logger.info("単語数を計算中...")
        
        # テキストカラムがリスト形式の場合（前処理済み）
        if self._df[self.text_column].apply(lambda x: isinstance(x, list)).all():
            word_counts = self._df[self.text_column].apply(len)
        else:
            # 文字列形式の場合は空白で分割
            word_counts = self._df[self.text_column].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
        
        return word_counts
    
    def get_basic_statistics(self) -> Dict[str, Any]:
        """基本統計情報を取得する
        
        Returns:
            Dict[str, Any]: 基本統計情報の辞書
        """
        self._validate_data()
        
        word_counts = self.calculate_word_counts()
        
        stats = {
            'total_posts': len(self._df),
            'mean_word_count': word_counts.mean(),
            'std_word_count': word_counts.std(),
            'median_word_count': word_counts.median(),
            'min_word_count': word_counts.min(),
            'max_word_count': word_counts.max(),
            'zero_word_posts': (word_counts == 0).sum(),
            'zero_word_ratio': (word_counts == 0).mean() * 100
        }
        
        return stats
    
    def get_distribution_info(self) -> Dict[str, Any]:
        """単語数分布の詳細情報を取得する
        
        Returns:
            Dict[str, Any]: 分布情報の辞書
        """
        word_counts = self.calculate_word_counts()
        
        distribution = {
            'percentiles': {
                '25%': word_counts.quantile(0.25),
                '50%': word_counts.quantile(0.50),
                '75%': word_counts.quantile(0.75),
                '90%': word_counts.quantile(0.90),
                '95%': word_counts.quantile(0.95),
                '99%': word_counts.quantile(0.99)
            },
            'word_count_ranges': {
                '0語': (word_counts == 0).sum(),
                '1-5語': ((word_counts >= 1) & (word_counts <= 5)).sum(),
                '6-10語': ((word_counts >= 6) & (word_counts <= 10)).sum(),
                '11-20語': ((word_counts >= 11) & (word_counts <= 20)).sum(),
                '21-50語': ((word_counts >= 21) & (word_counts <= 50)).sum(),
                '51語以上': (word_counts > 50).sum()
            }
        }
        
        return distribution
    
    def display_statistics(self) -> None:
        """統計情報を整形して表示する"""
        basic_stats = self.get_basic_statistics()
        distribution_info = self.get_distribution_info()
        
        print("\n=== データ統計情報 ===)
        print(f"総ポスト数: {basic_stats['total_posts']:,}")
        print(f"平均単語数: {basic_stats['mean_word_count']:.2f}")
        print(f"単語数の標準偏差: {basic_stats['std_word_count']:.2f}")
        print(f"中央値: {basic_stats['median_word_count']:.2f}")
        print(f"最小単語数: {basic_stats['min_word_count']}")
        print(f"最大単語数: {basic_stats['max_word_count']}")
        print(f"単語数が0のポスト数: {basic_stats['zero_word_posts']:,}")
        print(f"単語数が0のポストの割合: {basic_stats['zero_word_ratio']:.2f}%")
        
        print("\n=== 分布情報 ===)
        print("パーセンタイル:")
        for percentile, value in distribution_info['percentiles'].items():
            print(f"  {percentile}: {value:.2f}語")
        
        print("\n単語数範囲別の分布:")
        for range_name, count in distribution_info['word_count_ranges'].items():
            percentage = (count / basic_stats['total_posts']) * 100
            print(f"  {range_name}: {count:,} 件 ({percentage:.2f}%)")
    
    def get_sample_data(self, n: int = 5) -> pd.DataFrame:
        """サンプルデータを取得する
        
        Args:
            n (int, optional): 取得する行数. Defaults to 5.
            
        Returns:
            pd.DataFrame: サンプルデータ
        """
        self._validate_data()
        
        word_counts = self.calculate_word_counts()
        sample_df = self._df[[self.text_column]].copy()
        sample_df['word_count'] = word_counts
        
        return sample_df.head(n)
    
    def analyze_full_data(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """データファイルの完全分析を実行する
        
        Args:
            file_path (Union[str, Path]): 分析対象のファイルパス
            
        Returns:
            Dict[str, Any]: 分析結果の辞書
        """
        self.load_data(file_path)
        
        basic_stats = self.get_basic_statistics()
        distribution_info = self.get_distribution_info()
        
        return {
            'basic_statistics': basic_stats,
            'distribution_info': distribution_info,
            'sample_data': self.get_sample_data()
        }


def analyze_processed_data(
    file_path: Union[str, Path],
    text_column: str = 'processedtext',
    display_results: bool = True
) -> Dict[str, Any]:
    """処理済みデータの統計分析を実行する便利関数
    
    Args:
        file_path (Union[str, Path]): 分析対象のファイルパス
        text_column (str, optional): テキストカラム名. Defaults to 'processedtext'.
        display_results (bool, optional): 結果を表示するか. Defaults to True.
        
    Returns:
        Dict[str, Any]: 分析結果
    """
    analyzer = DataStatisticsAnalyzer(text_column=text_column)
    result = analyzer.analyze_full_data(file_path)
    
    if display_results:
        analyzer.display_statistics()
        print("\\n=== サンプルデータ（先頭5件） ===\")
        print(result['sample_data'].to_string(index=False))
    
    return result


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
        description='処理済みテキストデータの統計分析ツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で実行
  python data_statistics.py -f processed_data.pkl

  # カスタムテキストカラムを指定
  python data_statistics.py -f data.pkl -c processed_text

  # 詳細ログで実行
  python data_statistics.py -f data.pkl --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        type=str,
        required=True,
        help='分析対象のデータファイルパス'
    )
    
    parser.add_argument(
        '-c', '--text-column',
        type=str,
        default='processedtext',
        help='テキストカラム名 (デフォルト: processedtext)'
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
        # データ統計分析を実行
        result = analyze_processed_data(
            file_path=args.file,
            text_column=args.text_column,
            display_results=True
        )
        
        logger.info("統計分析が正常に完了しました")
        
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