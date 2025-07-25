"""テキスト前処理パッケージ

このパッケージは日本語と英語のテキスト前処理機能を提供する。
主要な機能には統計分析、データ結合、閲覧機能が含まれる。

モジュール:
- japanese_text_processor: 日本語テキスト前処理
- english_text_processor: 英語テキスト前処理  
- processed_data_combiner: データ結合機能
- data_statistics: 統計分析機能
- data_viewer: データ閲覧機能
- csv_inspector: CSV検査機能

使用例:
    # 日本語テキスト処理
    from tpext.preprocess import JapaneseTextProcessor
    processor = JapaneseTextProcessor()
    
    # 英語テキスト処理
    from tpext.preprocess import EnglishTextProcessor
    processor = EnglishTextProcessor()
    
    # 統計分析
    from tpext.preprocess import analyze_processed_data
    result = analyze_processed_data('data.pkl')
    
    # データ閲覧
    from tpext.preprocess import view_data
    data = view_data('data.pkl')
"""

# 日本語テキスト処理
from .japanese_text_processor import (
    JapaneseTextProcessor,
    ChunkProcessor,
    load_data
)

# 英語テキスト処理
from .english_text_processor import (
    EnglishTextProcessor,
    CSVFileProcessor
)

# データ結合
from .processed_data_combiner import (
    DataFrameCombiner,
    combine_pickle_files
)

# 統計分析
from .data_statistics import (
    DataStatisticsAnalyzer,
    analyze_processed_data
)

# データ閲覧
from .data_viewer import (
    DataViewer,
    view_data
)

# CSV検査
from .csv_inspector import (
    CSVInspector,
    inspect_csv,
    find_csv_files
)

# パッケージレベルの公開API
__all__ = [
    # 日本語テキスト処理
    'JapaneseTextProcessor',
    'ChunkProcessor',
    'load_data',
    
    # 英語テキスト処理
    'EnglishTextProcessor',
    'CSVFileProcessor',
    
    # データ結合
    'DataFrameCombiner',
    'combine_pickle_files',
    
    # 統計分析
    'DataStatisticsAnalyzer',
    'analyze_processed_data',
    
    # データ閲覧
    'DataViewer',
    'view_data',
    
    # CSV検査
    'CSVInspector',
    'inspect_csv',
    'find_csv_files'
]

# バージョン情報
__version__ = "1.0.0"
__author__ = "Cursor AI Code Editor"

# パッケージ情報
__description__ = "テキスト前処理と拡張機能のPythonパッケージ"
__license__ = "MIT"