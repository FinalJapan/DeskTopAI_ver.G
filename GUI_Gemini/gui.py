import tkinter as tk
import time, threading, sounddevice as sd, numpy as np, os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageTk

# -------------------------------------------------
# ① .env を必ず “ルート” で読む
# -------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]      # ← DeskTopAI/
load_dotenv(ROOT_DIR / ".env", override=False)

import os, inspect
print("DEBUG(gui) file =", inspect.getfile(inspect.currentframe()))
print("DEBUG(gui) Gemini =", os.getenv("GOOGLE_API_KEY"))
print("DEBUG(gui) WEATHER =", os.getenv("OPENWEATHER_API_KEY"))

from backend import AssistantBackend

# デバッグ確認（あとで消してOK）
print("DEBUG OPENAI:", os.getenv("GOOGLE_API_KEY")[:5], "...")

# -------------------------------------------------
# GUI 定数
# -------------------------------------------------
WIDTH, HEIGHT = 300, 300
BG_COLOR  = "white"
IDLE_RADIUS = 100         # 待機中直径 / 2

# -------------------------------------------------
# 画面セットアップ
# -------------------------------------------------
root = tk.Tk()
root.title("Desktop AI Assistant")
root.geometry(f"{WIDTH}x{HEIGHT}")
root.configure(bg=BG_COLOR)

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg=BG_COLOR, highlightthickness=0)
canvas.pack(fill="both", expand=True)

center_x, center_y = WIDTH // 2, HEIGHT // 2 - 40

# -------------------------------------------------
# ② 中央ディスク画像
# -------------------------------------------------
ASSET_DIR = Path(__file__).parent / "assets"
disc_src  = Image.open(ASSET_DIR / "stop.png").convert("RGBA")

def make_disc(diam):
    return ImageTk.PhotoImage(disc_src.resize((diam, diam), Image.LANCZOS))

disc_photo = make_disc(IDLE_RADIUS * 2)
disc_item  = canvas.create_image(center_x, center_y, image=disc_photo)




# -------------------------------------------------
# バックエンド
# -------------------------------------------------
backend = AssistantBackend()
is_recording, volume_level = False, 0.0

def on_mic_pressed():
    global is_recording
    if is_recording:
        return
    start_recording()

def skip_playback(event=None):
    """再生中の音声を即停止してステータスを更新"""
    try:
        sd.stop()                       # ← 音声ストリームを強制停止
        status_var.set("🔇 再生スキップ")
    except Exception as e:
        print("⚠️ スキップ失敗:", e)
        
def start_recording():
    global is_recording
    is_recording = True
    status_var.set("🎙️ 録音中 ... F2 でも停止可")

    backend.record_and_reply(
        on_status_print=status_var.set,
        on_finish=reset_after_playback
    )
    threading.Thread(target=monitor_mic_level, daemon=True).start()

def reset_after_playback():
    global is_recording
    is_recording = False
    status_var.set("✅ 再生完了 / 待機中")

def monitor_mic_level():
    global is_recording, volume_level
    def callback(indata, frames, t, status):
        global volume_level
        volume_level = float(np.sqrt(np.mean(indata ** 2)))
    with sd.InputStream(channels=1, callback=callback):
        while is_recording:
            time.sleep(0.05)

# -------------------------------------------------
# ステータスラベル & ボタンフレーム（下詰めに pack）
# -------------------------------------------------
status_var = tk.StringVar(value="F2 or 🎤 で録音開始")

status_label = tk.Label(root, textvariable=status_var,
                        font=("Segoe UI", 14), bg=BG_COLOR)
status_label.pack(side="bottom", pady=(0, 8))     # ← ここだけで十分

btn_frame = tk.Frame(root, bg=BG_COLOR)
btn_frame.pack(side="bottom", pady=8)

tk.Button(btn_frame, text="🎤", font=("Segoe UI", 28),
          borderwidth=0, command=on_mic_pressed
          ).grid(row=0, column=0, padx=20)

tk.Button(btn_frame, text="✖", font=("Segoe UI", 28),
          borderwidth=0, command=root.destroy
          ).grid(row=0, column=1, padx=20)



# -------------------------------------------------
# アニメーション
# -------------------------------------------------
def animate():
    global disc_photo

    base  = 1.0
    amp   = min(volume_level * 30, 1.0) if is_recording else 0.0
    scale = base + amp                    # 1.0〜2.0

    # ★ここを変更★  短辺 × 45% を上限に
    short_side = min(canvas.winfo_width(), canvas.winfo_height())
    max_r = int(short_side * 0.45)        # 135px までOK (300×0.45)

    radius = max(1, min(int(IDLE_RADIUS * scale), max_r))
    diam   = radius * 2

    disc_photo = make_disc(diam)
    canvas.itemconfig(disc_item, image=disc_photo)
    canvas.coords(disc_item, center_x, center_y)

    root.after(20, animate)

animate()  # アニメーション開始

def on_resize(event):
    global center_x, center_y
    center_x = event.width  // 2
    center_y = event.height // 2 - 40
    canvas.coords(disc_item, center_x, center_y)  # 画像を再センタリング

canvas.bind("<Configure>", on_resize)             # ★追加
canvas.bind("<Button-1>", lambda e: on_mic_pressed())  # クリックで録音開始
canvas.bind("<Button-3>", skip_playback)          # 右クリックでスキップ

# -------------------------------------------------
# キーバインド
# -------------------------------------------------
root.bind("<KeyRelease-F2>", lambda e: on_mic_pressed())
root.bind("<Escape>",        lambda e: root.destroy())

root.mainloop()
