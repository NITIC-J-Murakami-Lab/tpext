"""英語テキストの前処理を行うモジュール

このモジュールは英語のCSVファイルを効率的に処理するために、
spaCyを使用してレンマ化（見出し語化）と重要な単語の抽出を行う。

主な機能:
- 英語テキストのクリーニングとレンマ化
- 複数CSVファイルの並列処理
- エラーハンドリングと処理継続
- 処理済みファイルのスキップ機能

Classes:
    EnglishTextProcessor: 英語テキストの前処理を行うクラス
    CSVFileProcessor: CSVファイルの一括処理を行うクラス
"""
from typing import List, Optional, Union, Any
import pandas as pd
import spacy
from pandarallel import pandarallel
import re
import os
import glob
from pathlib import Path
import argparse
import logging
import sys


# 公開API（import時に見えるもの）
__all__ = [
    'EnglishTextProcessor',
    'CSVFileProcessor'
]

# ログ設定（デフォルト）
logger = logging.getLogger(__name__)


class EnglishTextProcessor:
    """英語テキストの前処理を行うクラス
    
    Attributes:
        _nlp (Optional[spacy.Language]): spaCy英語モデル
        _url_pattern (re.Pattern): URL削除用の正規表現パターン
        _mention_pattern (re.Pattern): メンション削除用の正規表現パターン
        _clean_pattern (re.Pattern): 英字以外の文字削除用の正規表現パターン
    """
    
    def __init__(self, model_name: str = 'en_core_web_sm'):
        """EnglishTextProcessorを初期化する
        
        Args:
            model_name (str, optional): spaCyモデル名. Defaults to 'en_core_web_sm'.
        """
        self.model_name = model_name
        self._nlp: Optional[spacy.Language] = None
        
        # 正規表現パターンをコンパイル（高速化のため）
        self._url_pattern = re.compile(r'https?://[^\s]+')
        self._mention_pattern = re.compile(r'@[a-zA-Z0-9_]+')
        self._clean_pattern = re.compile(r'[^a-zA-Z\s]')
    
    def _get_nlp(self) -> spacy.Language:
        """spaCyモデルを取得する（遅延初期化）
        
        Returns:
            spacy.Language: spaCy英語モデル
            
        Raises:
            ImportError: spaCyモデルが見つからない場合
        """
        if self._nlp is None:
            try:
                self._nlp = spacy.load(
                    self.model_name,
                    disable=['parser', 'ner']  # 高速化のため不要な機能を無効化
                )
                logger.info(f"spaCyモデル '{self.model_name}' を読み込みました")
            except OSError as e:
                raise ImportError(
                    f"spaCyモデル '{self.model_name}' が見つかりません。"
                    f"以下のコマンドでインストールしてください:\n"
                    f"python -m spacy download {self.model_name}"
                ) from e
        return self._nlp
    
    def lemmatize_text(self, text: Union[str, Any]) -> List[str]:
        """英語テキストをクリーニングし、重要な単語をレンマ化する
        
        Args:
            text (Union[str, Any]): 処理対象のテキスト。文字列以外の場合は空リストを返す。
            
        Returns:
            List[str]: レンマ化された重要な単語のリスト
            
        Note:
            - URLとメンション（@username）は事前に削除される
            - 英字と空白以外の文字は除去される
            - ストップワードは除去される
            - 名詞、固有名詞、動詞、形容詞のみを抽出
            - 全て小文字に変換してからレンマ化を実行
        """
        if not isinstance(text, str):
            return []
        
        # 1. 小文字に変換
        text = text.lower()
        
        # 2. URLとメンションを削除（コンパイル済み正規表現を使用）
        cleaned_text: str = self._url_pattern.sub('', text)
        cleaned_text = self._mention_pattern.sub('', cleaned_text)
        
        # 3. 英字と空白以外の文字を削除
        cleaned_text = self._clean_pattern.sub('', cleaned_text)
        
        # 空文字列の場合は早期リターン
        if not cleaned_text.strip():
            return []
        
        # 4. spaCyによる処理
        nlp = self._get_nlp()
        doc = nlp(cleaned_text)
        
        lemmatized_words: List[str] = [
            token.lemma_ for token in doc 
            if not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']
            and len(token.lemma_) > 1  # 1文字の単語を除外
        ]
        
        return lemmatized_words


class CSVFileProcessor:
    """CSVファイルの一括処理を行うクラス
    
    Attributes:
        text_processor (EnglishTextProcessor): テキスト処理インスタンス
        input_dir (Path): 入力ディレクトリのパス
        output_dir (Path): 出力ディレクトリのパス
        text_column (str): 処理対象のテキストカラム名
    """
    
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path] = 'processed_pkls_en',
        text_column: str = 'text',
        spacy_model: str = 'en_core_web_sm'
    ):
        """CSVFileProcessorを初期化する
        
        Args:
            input_dir (Union[str, Path]): 入力ディレクトリのパス
            output_dir (Union[str, Path], optional): 出力ディレクトリ. Defaults to 'processed_pkls_en'.
            text_column (str, optional): 処理対象のテキストカラム名. Defaults to 'text'.
            spacy_model (str, optional): spaCyモデル名. Defaults to 'en_core_web_sm'.
        """
        self.text_processor = EnglishTextProcessor(spacy_model)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.text_column = text_column
        
        # 入力ディレクトリの存在確認
        if not self.input_dir.exists():
            raise FileNotFoundError(f"入力ディレクトリが見つかりません: {self.input_dir}")
        
        # pandarallelを初期化
        pandarallel.initialize(progress_bar=True)
    
    def _setup_output_directory(self) -> None:
        """出力ディレクトリを作成する"""
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"出力ディレクトリを作成しました: {self.output_dir}")
    
    def _get_csv_files(self) -> List[Path]:
        """入力ディレクトリからCSVファイルのリストを取得する
        
        Returns:
            List[Path]: CSVファイルのパスリスト
            
        Raises:
            FileNotFoundError: CSVファイルが見つからない場合
        """
        csv_files = list(self.input_dir.glob('*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(
                f"ディレクトリ '{self.input_dir}' 内にCSVファイルが見つかりません"
            )
        
        logger.info(f"{len(csv_files)} 個のCSVファイルを検出しました")
        return sorted(csv_files)
    
    def _get_output_path(self, csv_file: Path) -> Path:
        """CSVファイルに対応する出力パスを生成する
        
        Args:
            csv_file (Path): 入力CSVファイルのパス
            
        Returns:
            Path: 出力pklファイルのパス
        """
        output_base_name = csv_file.stem
        return self.output_dir / f"{output_base_name}_processed.pkl"
    
    def _is_file_processed(self, output_path: Path) -> bool:
        """ファイルが既に処理済みかどうかを確認する
        
        Args:
            output_path (Path): 出力ファイルのパス
            
        Returns:
            bool: 処理済みの場合True
        """
        return output_path.exists()
    
    def _process_single_file(self, csv_file: Path, output_path: Path) -> bool:
        """単一のCSVファイルを処理する
        
        Args:
            csv_file (Path): 入力CSVファイルのパス
            output_path (Path): 出力pklファイルのパス
            
        Returns:
            bool: 処理が成功した場合True
        """
        try:
            logger.info(f"処理開始: {csv_file.name}")
            
            # CSVファイルを読み込み
            df = pd.read_csv(csv_file)
            
            # テキストカラムの存在確認
            if self.text_column not in df.columns:
                logger.warning(
                    f"'{self.text_column}' カラムが見つかりません: {csv_file.name}"
                )
                return False
            
            # 並列処理でテキストを処理
            df['processed_text'] = df[self.text_column].parallel_apply(
                self.text_processor.lemmatize_text
            )
            
            # 処理済みデータを保存
            df.to_pickle(output_path)
            
            logger.info(f"処理完了: {csv_file.name} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ファイル '{csv_file.name}' の処理中にエラーが発生: {e}")
            return False
    
    def process_files(self) -> None:
        """全CSVファイルを処理する
        
        Raises:
            FileNotFoundError: CSVファイルが見つからない場合
        """
        logger.info("CSVファイルの一括処理を開始します")
        
        # 出力ディレクトリをセットアップ
        self._setup_output_directory()
        
        # CSVファイルのリストを取得
        csv_files = self._get_csv_files()
        
        # 処理統計
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        for i, csv_file in enumerate(csv_files, 1):
            logger.info(f"\\n--- ファイル {i}/{len(csv_files)} の処理: {csv_file.name} ---")
            
            output_path = self._get_output_path(csv_file)
            
            # 既に処理済みのファイルをスキップ
            if self._is_file_processed(output_path):
                logger.info(f"既に処理済みのためスキップ: {output_path}")
                skipped_count += 1
                continue
            
            # ファイルを処理
            if self._process_single_file(csv_file, output_path):
                processed_count += 1
            else:
                failed_count += 1
        
        # 処理結果のサマリを表示
        logger.info("\\n=== 処理結果サマリ ===")
        logger.info(f"処理完了: {processed_count} ファイル")
        logger.info(f"スキップ: {skipped_count} ファイル")
        logger.info(f"失敗: {failed_count} ファイル")
        logger.info(f"総ファイル数: {len(csv_files)} ファイル")
        
        if failed_count > 0:
            logger.warning(f"{failed_count} 個のファイルの処理に失敗しました")
        
        logger.info("全てのファイルの処理が完了しました")
    
    def get_file_info(self) -> pd.DataFrame:
        """CSVファイルの情報を取得する
        
        Returns:
            pd.DataFrame: ファイル情報のDataFrame
        """
        csv_files = self._get_csv_files()
        
        file_info = []
        for csv_file in csv_files:
            try:
                # ファイルサイズを取得
                file_size = csv_file.stat().st_size / (1024 * 1024)  # MB
                
                # 出力パスと処理状態を確認
                output_path = self._get_output_path(csv_file)
                is_processed = self._is_file_processed(output_path)
                
                file_info.append({
                    'ファイル名': csv_file.name,
                    'ファイルサイズ(MB)': round(file_size, 2),
                    '処理状態': '完了' if is_processed else '未処理',
                    '入力パス': str(csv_file),
                    '出力パス': str(output_path)
                })
                
            except Exception as e:
                file_info.append({
                    'ファイル名': csv_file.name,
                    'ファイルサイズ(MB)': 'エラー',
                    '処理状態': 'エラー',
                    '入力パス': str(csv_file),
                    '出力パス': 'エラー',
                    'エラー': str(e)
                })
        
        return pd.DataFrame(file_info)


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
        description='英語CSVファイルのテキスト前処理ツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で実行
  python english_text_processor.py -i /path/to/csv_files

  # カスタム設定で実行
  python english_text_processor.py -i input_dir -o output_dir -c content_column

  # ファイル情報のみ表示
  python english_text_processor.py -i input_dir --info-only

  # デバッグモードで実行
  python english_text_processor.py -i input_dir --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        required=True,
        help='入力CSVファイルが格納されているディレクトリのパス'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='processed_pkls_en',
        help='出力ディレクトリのパス (デフォルト: processed_pkls_en)'
    )
    
    parser.add_argument(
        '-c', '--text-column',
        type=str,
        default='text',
        help='処理対象のテキストカラム名 (デフォルト: text)'
    )
    
    parser.add_argument(
        '-m', '--spacy-model',
        type=str,
        default='en_core_web_sm',
        help='spaCyモデル名 (デフォルト: en_core_web_sm)'
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
        # CSVFileProcessorを初期化
        processor = CSVFileProcessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            text_column=args.text_column,
            spacy_model=args.spacy_model
        )
        
        # ファイル情報のみの場合
        if args.info_only:
            logger.info("=== CSVファイル一覧 ===")
            file_info_df = processor.get_file_info()
            print(file_info_df.to_string(index=False))
            logger.info("ファイル情報の表示が完了しました")
            return
        
        # 通常の処理
        if not args.quiet:
            logger.info("=== CSVファイル一覧 ===")
            file_info_df = processor.get_file_info()
            print(file_info_df.to_string(index=False))
        
        # ファイル処理を実行
        processor.process_files()
        
        logger.info("全ての処理が正常に完了しました")
        
        # 処理結果のサマリを表示
        if not args.quiet:
            logger.info(f"出力ディレクトリ: {args.output_dir}")
        
    except FileNotFoundError as e:
        logger.error(f"ファイルエラー: {e}")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"依存関係エラー: {e}")
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