"""
backend.py
-----------
GUI から使う “黒子” モジュール。
既存 gpt.py（音声録音 → Whisper → GPT → 合成 → 再生）を
AssistantBackend クラスでラップするだけで、ロジックは触らない。
"""

# ======== 既存スクリプトをインポート ========
# ファイル名が gpt.py ならそのまま。変更したら as gpt_core を適宜変える。
import gemini as gemini_core

from concurrent.futures import ThreadPoolExecutor
import threading


class AssistantBackend:
    """
    🎙 音声アシスタントのバックエンド
    --------------------------------
    GUI からは
        backend.record_and_reply(
            on_status_print=gui_callback,
            on_finish=gui_reset_callback
        )
    を呼ぶだけで一連の処理を非同期で実行する。
    """

    def __init__(self):
        # 同時に 1 つだけタスクを走らせる
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()

    # ----------------------------
    # メインハンドラ
    # ----------------------------
    def record_and_reply(
        self,
        on_status_print=lambda txt: None,
        on_finish=lambda: None
    ):
        """
        Parameters
        ----------
        on_status_print : callable(str)  途中経過を GUI に表示するための関数
        on_finish       : callable()     処理完了時に GUI が状態をリセットするための関数
        """

        def task():
            voice_path = None  # 例外時の未定義エラー防止

            try:
                # 1) 録音 -------------------------------------------------
                on_status_print("🎤 録音開始 ...")
                wav_path = gemini_core.smart_record()
                if not wav_path:
                    on_status_print("⚠️ 音声がありません")
                    return

                # 2) GPT 応答生成 ----------------------------------------
                on_status_print("🤖 Gemini に問い合わせ中 ...")
                voice_path = gemini_core.process_audio_and_generate_reply(wav_path)

                # 3) 合成音声を再生 --------------------------------------
                if voice_path:
                    on_status_print("🔊 応答を再生中 ...")
                    gemini_core.play_voice(voice_path)
                else:
                    on_status_print("⚠️ 合成音声が生成できなかったよ")

            except Exception as e:
                # 何か起きても GUI に出してあげる
                on_status_print(f"💥 予期せぬエラー: {e}")

            finally:
                # GUI 側に「全部終わったよ」と伝える
                on_finish()

        # ============ 非同期実行 ============
        with self._lock:               # 多重押下ガード
            self.executor.submit(task)
