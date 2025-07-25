"""チャンクごとに日本語テキストの前処理を行うモジュール

このモジュールは大量の日本語テキストデータを効率的に処理するために、
データをチャンクに分割して並列処理を行い、名詞のみを抽出する。

主な機能:
- 日本語テキストからの名詞抽出
- 大容量データのチャンク分割処理
- 並列処理による高速化
- 中間結果の保存と再開機能

Classes:
    JapaneseTextProcessor: 日本語テキストの前処理を行うクラス
    ChunkProcessor: チャンク分割処理を行うクラス
"""
from typing import List, Optional, Union, Any
import pandas as pd
from janome.tokenizer import Tokenizer
from pandarallel import pandarallel
import numpy as np
import re
import glob 
import os   
from pathlib import Path
import argparse
import logging


# 公開API
__all__ = [
    'JapaneseTextProcessor',
    'ChunkProcessor', 
    'load_data'
]

# ログ設定（デフォルト）
logger = logging.getLogger(__name__)


class JapaneseTextProcessor:
    """日本語テキストの前処理を行うクラス
    
    Attributes:
        _tokenizer (Optional[Tokenizer]): Janome形態素解析器
        _url_pattern (re.Pattern): URL削除用の正規表現パターン
        _mention_pattern (re.Pattern): メンション削除用の正規表現パターン
        _clean_pattern (re.Pattern): 不要文字削除用の正規表現パターン
    """
    
    def __init__(self):
        """JapaneseTextProcessorを初期化する"""
        self._tokenizer: Optional[Tokenizer] = None
        
        # 正規表現パターンをコンパイル（高速化のため）
        self._url_pattern = re.compile(r'https?://[^\s]+')
        self._mention_pattern = re.compile(r'@[a-zA-Z0-9_]+')
        self._clean_pattern = re.compile(r'[^ぁ-んァ-ヶ一-龥a-zA-Z0-9\sー]')
    
    def _get_tokenizer(self) -> Tokenizer:
        """Tokenizerインスタンスを取得する（遅延初期化）
        
        Returns:
            Tokenizer: Janome形態素解析器
        """
        if self._tokenizer is None:
            self._tokenizer = Tokenizer()
        return self._tokenizer
    
    def extract_nouns(self, text: Union[str, Any]) -> List[str]:
        """テキストから名詞のみを抽出する
        
        Args:
            text (Union[str, Any]): 処理対象のテキスト。文字列以外の場合は空リストを返す。
            
        Returns:
            List[str]: 抽出された名詞のリスト
            
        Note:
            - URLとメンション（@username）は事前に削除される
            - 日本語、英数字、空白以外の文字（記号、絵文字等）は除去される
            - Janome形態素解析器を使用して名詞のみを抽出
            - 正規表現パターンは事前にコンパイルされ、高速処理を実現
        """
        if not isinstance(text, str):
            return []
        
        # 1. URLとメンションを削除（コンパイル済み正規表現を使用）
        cleaned_text: str = self._url_pattern.sub('', text)
        cleaned_text = self._mention_pattern.sub('', cleaned_text)
        
        # 2. 日本語、英数字、空白以外の文字（記号、絵文字など）を全て削除
        cleaned_text = self._clean_pattern.sub('', cleaned_text)
        
        # 空文字列の場合は早期リターン
        if not cleaned_text.strip():
            return []
        
        # 3. 形態素解析
        tokenizer = self._get_tokenizer()
        nouns: List[str] = []
        
        for token in tokenizer.tokenize(cleaned_text):
            part_of_speech: str = token.part_of_speech.split(',')[0]
            if part_of_speech == '名詞':
                nouns.append(token.surface)
        
        return nouns


class ChunkProcessor:
    """大容量データのチャンク分割処理を行うクラス
    
    Attributes:
        text_processor (JapaneseTextProcessor): テキスト処理インスタンス
        chunk_size (int): チャンクサイズ
        output_dir (str): 出力ディレクトリ
    """
    
    def __init__(self, chunk_size: int = 500000, output_dir: str = 'processed_chunks'):
        """ChunkProcessorを初期化する
        
        Args:
            chunk_size (int, optional): チャンクサイズ. Defaults to 500000.
            output_dir (str, optional): 出力ディレクトリ. Defaults to 'processed_chunks'.
        """
        self.text_processor = JapaneseTextProcessor()
        self.chunk_size = chunk_size
        self.output_dir = Path(output_dir)
        
        # pandarallelを初期化
        pandarallel.initialize(progress_bar=True)
    
    def _setup_output_directory(self) -> None:
        """出力ディレクトリを作成する"""
        self.output_dir.mkdir(exist_ok=True)
    
    def _get_chunk_path(self, chunk_index: int) -> Path:
        """チャンクファイルのパスを取得する
        
        Args:
            chunk_index (int): チャンクのインデックス
            
        Returns:
            Path: チャンクファイルのパス
        """
        return self.output_dir / f'chunk_{chunk_index}.pkl'
    
    def _is_chunk_processed(self, chunk_index: int) -> bool:
        """チャンクが既に処理済みかどうかを確認する
        
        Args:
            chunk_index (int): チャンクのインデックス
            
        Returns:
            bool: 処理済みの場合True
        """
        return self._get_chunk_path(chunk_index).exists()
    
    def _process_chunk(self, chunk_df: pd.DataFrame, chunk_index: int, total_chunks: int) -> None:
        """単一チャンクを処理する
        
        Args:
            chunk_df (pd.DataFrame): 処理対象のチャンクデータ
            chunk_index (int): チャンクのインデックス
            total_chunks (int): 総チャンク数
        """
        print(f"\\n--- チャンク {chunk_index + 1}/{total_chunks} の処理を開始 ---")
        
        # 並列処理で名詞抽出
        processed_series: pd.Series = chunk_df['text'].parallel_apply(
            self.text_processor.extract_nouns
        )
        chunk_df['processedtext'] = processed_series
        
        # 元のテキストカラムを削除して保存
        processed_chunk: pd.DataFrame = chunk_df.drop('text', axis=1)
        
        chunk_path = self._get_chunk_path(chunk_index)
        processed_chunk.to_pickle(chunk_path)
        
        print(f"チャンク {chunk_index + 1} を {chunk_path} に保存しました。")
    
    def process_dataframe(self, df: pd.DataFrame) -> None:
        """DataFrameをチャンクに分割して処理する
        
        Args:
            df (pd.DataFrame): 処理対象のDataFrame。'text'カラムが必要。
            
        Raises:
            ValueError: 'text'カラムが存在しない場合
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrameには'text'カラムが必要です")
        
        # 出力ディレクトリをセットアップ
        self._setup_output_directory()
        
        print(f"合計 {len(df)} 行のデータを処理します。")
        print(f"チャンクサイズ: {self.chunk_size:,}")
        
        total_chunks = (len(df) - 1) // self.chunk_size + 1
        
        for i, chunk_start in enumerate(range(0, len(df), self.chunk_size)):
            # 既に処理済みのチャンクをスキップ
            if self._is_chunk_processed(i):
                chunk_path = self._get_chunk_path(i)
                print(f"--- チャンク {i + 1} は処理済みのためスキップします --- ({chunk_path})")
                continue
            
            # チャンクを取得
            chunk_end: int = min(chunk_start + self.chunk_size, len(df))
            chunk_df: pd.DataFrame = df.iloc[chunk_start:chunk_end].copy()
            
            # チャンクを処理
            self._process_chunk(chunk_df, i, total_chunks)
        
        print("\\n全てのチャンク処理が完了しました。")
    
    def load_processed_chunks(self) -> pd.DataFrame:
        """処理済みチャンクを読み込んで結合する
        
        Returns:
            pd.DataFrame: 結合された処理済みデータ
            
        Raises:
            FileNotFoundError: チャンクファイルが見つからない場合
        """
        chunk_files = sorted(self.output_dir.glob('chunk_*.pkl'))
        
        if not chunk_files:
            raise FileNotFoundError(f"チャンクファイルが見つかりません: {self.output_dir}")
        
        print(f"{len(chunk_files)} 個のチャンクファイルを結合します...")
        
        chunks: List[pd.DataFrame] = []
        for chunk_file in chunk_files:
            chunk_df = pd.read_pickle(chunk_file)
            chunks.append(chunk_df)
        
        combined_df = pd.concat(chunks, ignore_index=True)
        print(f"結合完了: {len(combined_df)} 行")
        
        return combined_df


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """pklファイルからデータを読み込む
    
    Args:
        file_path (Union[str, Path]): データファイルのパス
        
    Returns:
        pd.DataFrame: 読み込まれたデータ
        
    Raises:
        FileNotFoundError: ファイルが見つからない場合
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    
    print(f"データファイルの読み込みを開始します: {file_path}")
    df = pd.read_pickle(file_path)
    print(f"読み込み完了: {len(df)} 行")
    
    return df


def main(data_path:str = '/mnt/ExtreamSSD/ja_df.pkl') -> None:
    """メイン処理関数"""
    try:
        # データを読み込み
        df = load_data(data_path)
        
        # チャンク処理器を初期化
        processor = ChunkProcessor(
            chunk_size=500000,
            output_dir='processed_chunks'
        )
        
        # データを処理
        processor.process_dataframe(df)
        
        print("\\n処理が正常に完了しました。")
        
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        exit(1)
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
        exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='日本語テキストの前処理を行う')
    parser.add_argument('--data_path', type=str, default='/mnt/ExtreamSSD/ja_df.pkl', help='データファイルのパス')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.data_path)