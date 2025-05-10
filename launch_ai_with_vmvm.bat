@echo off
REM ===============================
REM  Desktop AI Assistant 起動スクリプト
REM ===============================

REM ▼ 1) バッチ自身のフォルダへ移動
cd /d "%~dp0"

REM ▼ 2) AivisSpeech を起動（非同期）
start "" "C:\Users\kanat\AppData\Local\Programs\AivisSpeech\AivisSpeech.exe"

REM ▼ 3) ★ 3秒待機して安定化 ★
REM     TIMEOUT は秒単位で待つコマンド
REM     /NOBREAK を付けるとキー押下で解除されない
TIMEOUT /T 3 /NOBREAK >nul

REM ▼ 4) Python GUI を起動（コンソール非表示）
start "" pythonw "GUI_Gemini\gui.py"

REM ▼ 5) バッチ終了
exit
