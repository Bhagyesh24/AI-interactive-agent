import cv2
import time
import os
import tkinter as tk

# Paths
AVATAR_VIDEO_PATH = r"C:\Users\Bhagyesh\Ai Interactive project\avatar\talking_loop.mp4"

# Avatar watermark crop
CROP_BOTTOM_PERCENT = 0.13

# Demo Q&A pairs
DEMO_QA = [
    ("What is blockchain?", "Blockchain is a secure, decentralized digital ledger."),
    ("How does AI learn?", "AI learns patterns from large amounts of example data."),
    ("What is a smart contract?", "A smart contract is code that runs automatically on a blockchain."),
]


# --- webcam helpers ---

def open_webcam(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap


def read_webcam_frame(cap, width=480, height=640):
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    return frame


# --- avatar helpers ---

def crop_frame(frame, crop_percent=CROP_BOTTOM_PERCENT):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    crop_h = int(h * (1 - crop_percent))
    return frame[:crop_h, 0:w]


def resize_avatar(frame, target_width=480, target_height=640):
    """Resize avatar to fit panel while preserving aspect ratio, letterboxing if needed."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Create black panel of target size and center the resized frame
    panel = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_off = (target_height - new_h) // 2
    x_off = (target_width - new_w) // 2
    panel[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return panel


# --- caption helpers ---

def create_caption_window(root, width=960, height=120):
    root.title("AI Companion - Captions")
    root.geometry(f"{width}x{height}+100+700")
    root.configure(bg="black")
    root.attributes("-topmost", True)

    frame = tk.Frame(root, bg="black")
    frame.pack(expand=True, fill="both", padx=10, pady=10)

    you_label = tk.Label(
        frame,
        text="You said: ",
        font=("Arial", 16, "bold"),
        fg="white",
        bg="black",
        wraplength=width - 40,
        justify="left",
        anchor="w",
    )
    you_label.pack(fill="x")

    ai_label = tk.Label(
        frame,
        text="AI: ",
        font=("Arial", 16, "bold"),
        fg="white",
        bg="black",
        wraplength=width - 40,
        justify="left",
        anchor="w",
    )
    ai_label.pack(fill="x")

    return you_label, ai_label


def update_captions(you_label, ai_label, question, answer):
    you_label.config(text=f"You said: {question}")
    ai_label.config(text=f"AI: {answer}")


# --- main layout ---

def main():
    # Panel sizes
    WEBCAM_W, WEBCAM_H = 480, 640
    AVATAR_W, AVATAR_H = 480, 640
    CAPTION_H = 120
    TOTAL_W = WEBCAM_W + AVATAR_W
    TOTAL_H = max(WEBCAM_H, AVATAR_H) + CAPTION_H

    cv2.namedWindow("AI Companion - UI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Companion - UI", TOTAL_W, TOTAL_H - CAPTION_H + 80)

    # Open webcam
    cap = open_webcam(0)
    if not cap.isOpened():
        print("[Webcam not found]")
        return

    # Load avatar idle frame
    if not os.path.exists(AVATAR_VIDEO_PATH):
        print(f"[Avatar video not found: {AVATAR_VIDEO_PATH}]")
        return

    avatar_cap = cv2.VideoCapture(AVATAR_VIDEO_PATH)
    ret, idle_frame = avatar_cap.read()
    avatar_cap.release()
    if not ret or idle_frame is None:
        print("[Could not read avatar video]")
        return
    idle_frame = crop_frame(idle_frame)
    idle_frame = resize_avatar(idle_frame, AVATAR_W, AVATAR_H)

    # Open talking video capture (kept open for looping)
    talking_cap = cv2.VideoCapture(AVATAR_VIDEO_PATH)

    # Tkinter captions
    root = tk.Tk()
    you_label, ai_label = create_caption_window(root, width=TOTAL_W, height=CAPTION_H)

    is_talking = False
    last_toggle = time.time()
    qa_index = 0

    # Initial captions
    update_captions(you_label, ai_label, DEMO_QA[0][0], DEMO_QA[0][1])

    print("UI running. Press Q in the CV window to quit.")

    while True:
        now = time.time()

        # Toggle talking every 8 seconds
        if now - last_toggle >= 8:
            is_talking = not is_talking
            last_toggle = now
            if not is_talking:
                # Update captions when returning to idle (new Q&A)
                qa_index = (qa_index + 1) % len(DEMO_QA)
                q, a = DEMO_QA[qa_index]
                update_captions(you_label, ai_label, q, a)
                # Reset talking video to start for next time
                talking_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Read webcam frame
        webcam_frame = read_webcam_frame(cap, WEBCAM_W, WEBCAM_H)
        if webcam_frame is None:
            webcam_frame = np.zeros((WEBCAM_H, WEBCAM_W, 3), dtype=np.uint8)

        # Read avatar frame (idle or talking)
        if is_talking:
            ret, avatar_frame = talking_cap.read()
            if not ret:
                talking_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, avatar_frame = talking_cap.read()
            if ret and avatar_frame is not None:
                avatar_frame = crop_frame(avatar_frame)
                avatar_frame = resize_avatar(avatar_frame, AVATAR_W, AVATAR_H)
            else:
                avatar_frame = idle_frame
        else:
            avatar_frame = idle_frame

        # Combine side by side
        combined = np.hstack([webcam_frame, avatar_frame])
        cv2.imshow("AI Companion - UI", combined)

        # Update tkinter
        root.update_idletasks()
        root.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    talking_cap.release()
    cv2.destroyAllWindows()
    root.destroy()
    print("Done.")


if __name__ == "__main__":
    import numpy as np
    main()
