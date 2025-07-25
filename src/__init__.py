"""TPEXT - Text Processing and Extension Tools

大規模テキストデータ処理のためのPythonパッケージ
日本語と英語のテキスト前処理、統計分析、閲覧機能を含む

パッケージ構成:
- preprocess: テキスト前処理とデータ処理
- utils: 共通ユーティリティ関数（将来拡張予定）

主な機能:
- 日本語テキストの分かち書きと名詞抽出
- 英語テキストの言語処理機能
- 大規模データの統計分析
- データファイルの閲覧とサンプリング
- ログ機能と性能計測
- CSV検査の自動化
- データ閲覧の拡張機能

使用例:
    # 基本的な使用
    import tpext
    
    # 日本語テキスト処理
    from tpext.preprocess import JapaneseTextProcessor
    jp_processor = JapaneseTextProcessor()
    nouns = jp_processor.extract_nouns("これは日本語のテキストです")
    
    # 英語テキスト処理
    from tpext.preprocess import EnglishTextProcessor
    en_processor = EnglishTextProcessor()
    lemmas = en_processor.lemmatize_text("This is an English text.")
    
    # 統計分析
    from tpext.preprocess import analyze_processed_data
    stats = analyze_processed_data('processed_data.pkl')
    
    # データ閲覧
    from tpext.preprocess import view_data
    data = view_data('data.pkl')
"""

# サブパッケージをインポート
from . import preprocess

# 公開API（よく使用される機能をトップレベルで利用可能に）
__all__ = [
    'preprocess',
]