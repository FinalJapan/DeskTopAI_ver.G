# 🎙️ DeskTopAI - 音声AIアシスタント

**Python 3.12** | 音声入力 → AI応答 → 音声出力の完全な音声AIアシスタント

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen)

## 📋 目次

- [概要](#概要)
- [主な機能](#主な機能)
- [システム要件](#システム要件)
- [インストール](#インストール)
- [設定](#設定)
- [使用方法](#使用方法)
- [プロジェクト構成](#プロジェクト構成)
- [トラブルシューティング](#トラブルシューティング)
- [ライセンス](#ライセンス)

## 🎯 概要

DeskTopAIは、音声入力からAI応答までを完全に自動化したデスクトップAIアシスタントです。

### 処理フロー
```
🎤 音声入力 → 🧠 Whisper文字起こし → 🤖 Gemini AI処理 → 🔊 音声合成 → 🎵 音声出力
```

### 特徴
- **リアルタイム音声認識**: Whisper AIによる高精度文字起こし
- **最新AI応答**: Google Gemini 2.0 Flashによる自然な会話
- **音声合成**: 自然な音声での応答
- **記憶機能**: 会話の内容を記憶・参照
- **Web検索**: リアルタイム情報取得
- **直感的GUI**: シンプルで使いやすいインターフェース

## ✨ 主な機能

### 🎙️ 音声認識・応答
- **音声入力**: マイクボタンまたはF2キーで録音開始
- **リアルタイム処理**: 音声を即座にテキストに変換
- **AI応答**: Gemini 2.0 Flashによる自然な回答生成
- **音声出力**: 合成音声での応答再生

### 🧠 記憶・学習機能
- **記憶保存**: 「覚えて〜」で情報を記憶
- **記憶削除**: 「これは忘れて〜」で記憶を削除
- **記憶参照**: 「〜って覚えてる？」で記憶を確認
- **永続化**: 記憶はJSONファイルに保存

### 🌐 Web連携機能
- **リアルタイム検索**: DuckDuckGoによるWeb検索
- **ニュース取得**: RSSフィードからの最新ニュース
- **天気情報**: OpenWeather APIによる天気取得
- **ブラウザ要約**: Webページの内容要約

### 🎨 ユーザーインターフェース
- **シンプルGUI**: Tkinterベースの直感的な操作
- **リアルタイム表示**: 処理状況をリアルタイム表示
- **キーボードショートカット**: F2（録音）、ESC（終了）
- **視覚的フィードバック**: 音声レベルに応じたアニメーション

## 💻 システム要件

### 必須要件
- **OS**: Windows 10/11, macOS 11+, Linux Ubuntu 20.04+
- **Python**: 3.10以上（推奨: 3.12）
- **メモリ**: 4GB以上（推奨: 8GB以上）
- **ストレージ**: 2GB以上の空き容量

### 推奨要件
- **CPU**: 4コア以上（音声処理のため）
- **GPU**: NVIDIA GPU（CUDA対応、オプション）
- **インターネット**: 安定した接続（AI API使用のため）

### 必要なAPIキー
- **Google Gemini API**: [Google AI Studio](https://aistudio.google.com/app/apikey)
- **OpenWeather API**: [OpenWeatherMap](https://openweathermap.org/api)（オプション）

## 🚀 インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/あなたのユーザー名/DeskTopAI.git
cd DeskTopAI
```

### 2. Python仮想環境の作成

```bash
# Python 3.12で仮想環境を作成
py -3.12 -m venv venv_ai

# 仮想環境を有効化
# Windows PowerShell
.\venv_ai\Scripts\Activate.ps1

# Windows コマンドプロンプト
venv_ai\Scripts\activate.bat

# macOS/Linux
source venv_ai/bin/activate
```

### 3. 依存パッケージのインストール

```bash
# 必要なパッケージをインストール
pip install faster-whisper google-generativeai duckduckgo_search feedparser sounddevice soundfile python-dotenv Flask flask-cors beautifulsoup4 requests keyboard pillow numpy
```

### 4. 環境変数の設定

`.env`ファイルを`DeskTopAI`フォルダに作成：

```env
# Google Gemini APIキー（必須）
GOOGLE_API_KEY=your_google_api_key_here

# OpenWeather APIキー（オプション）
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

## ⚙️ 設定

### APIキーの取得

#### Google Gemini APIキー
1. [Google AI Studio](https://aistudio.google.com/app/apikey)にアクセス
2. Googleアカウントでログイン
3. 「Create API Key」をクリック
4. 生成されたAPIキーをコピー
5. `.env`ファイルの`GOOGLE_API_KEY`に設定

#### OpenWeather APIキー（オプション）
1. [OpenWeatherMap](https://openweathermap.org/api)にアクセス
2. アカウントを作成
3. APIキーを取得
4. `.env`ファイルの`OPENWEATHER_API_KEY`に設定

### 音声設定
- **マイク**: システムのデフォルトマイクが使用されます
- **スピーカー**: システムのデフォルトスピーカーが使用されます
- **音量**: システムの音量設定に従います

## 🎮 使用方法

### 起動方法

#### 方法1: 仮想環境を使用（推奨）
```bash
# 仮想環境を有効化
.\venv_ai\Scripts\Activate.ps1

# アプリケーションを起動
python GUI_Gemini\gui.py
```

#### 方法2: バッチファイルを使用
```bash
# バッチファイルを実行
GUI_Gemini\run_gemini.bat
```

#### 方法3: 直接実行
```bash
# 仮想環境のPythonを直接指定
.\venv_ai\Scripts\python.exe GUI_Gemini\gui.py
```

### 操作方法

#### 基本操作
- **🎤 録音開始**: マイクボタンをクリックまたはF2キーを押す
- **⏹️ 録音停止**: 再度マイクボタンをクリックまたはF2キーを押す
- **❌ 終了**: ✖ボタンをクリックまたはESCキーを押す

#### 音声コマンド例
```
「こんにちは」→ 挨拶の応答
「今日の天気は？」→ 天気情報を取得
「覚えて、私の名前は田中です」→ 情報を記憶
「田中って覚えてる？」→ 記憶を確認
「これは忘れて、田中」→ 記憶を削除
「最新のニュースを教えて」→ ニュースを取得
```

### GUIの説明

#### メイン画面
- **中央のディスク**: 音声レベルに応じてアニメーション
- **ステータス表示**: 現在の処理状況を表示
- **🎤 ボタン**: 音声録音の開始/停止
- **✖ ボタン**: アプリケーションの終了

#### ステータス表示
- `F2 or 🎤 で録音開始` - 待機中
- `🎙️ 録音中 ... F2 でも停止可` - 録音中
- `🤖 Gemini に問い合わせ中 ...` - AI処理中
- `🔊 応答を再生中 ...` - 音声再生中
- `✅ 再生完了 / 待機中` - 完了

## 📁 プロジェクト構成

```
DeskTopAI/
├── GUI_Gemini/              # メインアプリケーション
│   ├── gui.py               # GUIメイン
│   ├── gemini.py            # AI処理エンジン
│   ├── backend.py           # バックエンド処理
│   ├── memory.py            # 記憶管理
│   ├── run_gemini.bat       # 起動スクリプト
│   ├── venv_ai/             # Python仮想環境
│   └── assets/              # 画像・音声ファイル
├── GUI/                     # 旧バージョン（OpenAI版）
├── chrome_Extension/        # ブラウザ拡張機能
├── backup/                  # バックアップファイル
├── .env                     # 環境変数（作成が必要）
├── .gitignore              # Git除外設定
└── README.md               # このファイル
```

### 主要ファイルの説明

#### `gui.py`
- **役割**: メインGUIアプリケーション
- **機能**: ユーザーインターフェース、音声入力、状態管理
- **技術**: Tkinter, PIL, sounddevice

#### `gemini.py`
- **役割**: AI処理エンジン
- **機能**: 音声認識、AI応答生成、音声合成、Web検索
- **技術**: faster-whisper, google-generativeai, Flask

#### `backend.py`
- **役割**: GUIとAI処理の橋渡し
- **機能**: 非同期処理、エラーハンドリング
- **技術**: ThreadPoolExecutor, threading

#### `memory.py`
- **役割**: 記憶管理
- **機能**: 情報の保存・削除・参照
- **技術**: JSON, threading

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. Pythonが認識されない
```bash
# エラー: 'python' は、コマンドレット...として認識されません
```
**解決方法**:
- Pythonを再インストール
- インストール時に「Add Python to PATH」にチェック
- ターミナルを再起動

#### 2. 仮想環境が有効化されない
```bash
# エラー: このシステムではスクリプトの実行が無効になっているため...
```
**解決方法**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 3. パッケージが見つからない
```bash
# エラー: ModuleNotFoundError: No module named 'faster_whisper'
```
**解決方法**:
- 仮想環境が有効化されているか確認
- パッケージを再インストール
```bash
pip install faster-whisper
```

#### 4. CUDAエラー
```bash
# エラー: Could not locate cudnn_ops64_9.dll
```
**解決方法**:
- `gemini.py`の61行目を修正
```python
# 修正前
whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")

# 修正後
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
```

#### 5. APIキーエラー
```bash
# エラー: DEBUG(gui) Gemini = None
```
**解決方法**:
- `.env`ファイルが正しい場所にあるか確認
- APIキーが正しく設定されているか確認
- 環境変数名が`GOOGLE_API_KEY`になっているか確認

#### 6. 音声入力が動作しない
**解決方法**:
- マイクの権限を確認
- 音声デバイスが正しく設定されているか確認
- 仮想環境が有効化されているか確認

### デバッグ方法

#### ログの確認
```bash
# ターミナルでエラーメッセージを確認
python GUI_Gemini\gui.py
```

#### 環境の確認
```bash
# Pythonバージョン確認
python --version

# 仮想環境確認
where python

# パッケージ確認
pip list
```

## 🚀 高度な設定

### パフォーマンス最適化

#### GPU使用（NVIDIA GPUがある場合）
```python
# gemini.py の61行目
whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")
```

#### モデルサイズの調整
```python
# より高速（精度は落ちる）
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# バランス型
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# 高精度（遅い）
whisper_model = WhisperModel("large", device="cpu", compute_type="int8")
```

### カスタマイズ

#### 音声設定の変更
```python
# gemini.py の音声パラメータ
THRESHOLD_START = 0.02   # 録音開始音量
THRESHOLD_STOP = 0.01    # 録音停止音量
SILENCE_DURATION = 1.0   # 無音継続時間
SAMPLE_RATE = 44_100     # サンプリングレート
```

#### GUI設定の変更
```python
# gui.py のGUI設定
WIDTH, HEIGHT = 300, 300  # ウィンドウサイズ
BG_COLOR = "white"        # 背景色
IDLE_RADIUS = 100         # アイドル時の円のサイズ
```

## 📝 開発者向け情報

### アーキテクチャ

```
GUI Layer (gui.py)
    ↓
Backend Layer (backend.py)
    ↓
AI Engine (gemini.py)
    ↓
External APIs (Google Gemini, Whisper, etc.)
```

### 拡張方法

#### 新しい機能の追加
1. `gemini.py`に機能を追加
2. `backend.py`でGUIとの連携を実装
3. `gui.py`でUI要素を追加

#### 新しいAIモデルの追加
1. `gemini.py`のモデル初期化部分を修正
2. 対応するAPIキーを`.env`に追加
3. エラーハンドリングを追加

### テスト

```bash
# 単体テスト
python -m pytest tests/

# 統合テスト
python GUI_Gemini\gui.py
```

## 🤝 貢献

### 貢献方法
1. このリポジトリをフォーク
2. 新しいブランチを作成
3. 変更をコミット
4. プルリクエストを作成

### 報告事項
- バグ報告
- 機能要望
- ドキュメント改善
- コード改善

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 📞 サポート

### 問題報告
- GitHub Issuesで報告
- 詳細なエラーメッセージを含める
- 環境情報（OS、Pythonバージョン）を記載

### コミュニティ
- ディスカッション: GitHub Discussions
- 質問: GitHub Issues

---

## 🎉 まとめ

DeskTopAIは、最新のAI技術を活用した音声アシスタントです。Python 3.12対応により、より安定した動作を実現しています。

**主な特徴**:
- 🎙️ リアルタイム音声認識
- 🤖 最新AI応答（Gemini 2.0 Flash）
- 🧠 記憶・学習機能
- 🌐 Web連携
- 🎨 直感的GUI

**今すぐ始める**:
1. リポジトリをクローン
2. 仮想環境を作成
3. パッケージをインストール
4. APIキーを設定
5. アプリケーションを起動

**Happy Coding!** 🚀
