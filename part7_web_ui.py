from flask import Flask, render_template
import os
import cv2
import shutil

app = Flask(__name__)

PROJECT_DIR = r"C:\Users\Bhagyesh\Ai Interactive project"
AVATAR_DIR = os.path.join(PROJECT_DIR, "avatar")
STATIC_DIR = os.path.join(PROJECT_DIR, "static")
TEMPLATES_DIR = os.path.join(PROJECT_DIR, "templates")
VIDEO_PATH = os.path.join(AVATAR_DIR, "talking_loop_cropped.mp4")
IDLE_PATH = os.path.join(STATIC_DIR, "idle.jpg")
STATIC_VIDEO_PATH = os.path.join(STATIC_DIR, "talking_loop_cropped.mp4")


def prepare_static_files():
    """Ensure static video and idle frame exist."""
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(TEMPLATES_DIR, exist_ok=True)

    if not os.path.exists(VIDEO_PATH):
        print(f"[WARNING] Avatar video not found: {VIDEO_PATH}")
        return

    if not os.path.exists(STATIC_VIDEO_PATH):
        shutil.copy(VIDEO_PATH, STATIC_VIDEO_PATH)
        print(f"[INFO] Copied video to static folder: {STATIC_VIDEO_PATH}")

    if not os.path.exists(IDLE_PATH):
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(IDLE_PATH, frame)
            print(f"[INFO] Extracted idle frame: {IDLE_PATH}")
        else:
            print("[WARNING] Failed to extract idle frame from video")
        cap.release()


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    prepare_static_files()
    app.run(host="0.0.0.0", port=5000, debug=True)
