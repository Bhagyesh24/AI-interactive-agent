import cv2
import time
import os

# Path to the avatar talking-loop video
VIDEO_PATH = r"C:\Users\Bhagyesh\Ai Interactive project\avatar\talking_loop.mp4"

# Watermark crop: remove bottom N% of frame height
CROP_BOTTOM_PERCENT = 0.13  # adjust 0.10 to 0.15 if needed

# --- helpers ---


def crop_frame(frame, crop_percent=CROP_BOTTOM_PERCENT):
    """Crop off the bottom of the frame to remove watermark."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    crop_h = int(h * (1 - crop_percent))
    return frame[:crop_h, 0:w]


def resize_keep_aspect(frame, max_width=640, max_height=720):
    """Resize frame to fit within bounds without distorting aspect ratio."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return frame
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def display_idle(idle_frame, wait_ms=3000):
    """Show the frozen idle frame for a set duration."""
    start = time.time()
    while time.time() - start < wait_ms / 1000.0:
        cv2.imshow("Avatar", idle_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True


def display_talking(video_path, idle_frame, duration_s=6, fps_cap=30):
    """Play the talking-loop video for duration_s, looping if needed, then return to idle."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error: Could not open video {video_path}]")
        return False

    start = time.time()
    frame_time = 1.0 / fps_cap

    while time.time() - start < duration_s:
        ret, frame = cap.read()
        if not ret:
            # loop back to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        frame = crop_frame(frame)
        frame = resize_keep_aspect(frame)
        cv2.imshow("Avatar", frame)
        if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord('q'):
            cap.release()
            return False

    cap.release()

    # brief return to idle
    return display_idle(idle_frame, 500)


# --- main ---

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"[Video not found: {VIDEO_PATH}]")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, first_frame = cap.read()
    cap.release()

    if not ret or first_frame is None:
        print("[Could not read first frame from video]")
        return

    first_frame = crop_frame(first_frame)
    idle_frame = resize_keep_aspect(first_frame)

    print("Avatar window opening. Press Q to quit.")
    print(f"Native crop resolution: {idle_frame.shape[1]}x{idle_frame.shape[0]}")

    # Idle for 3 seconds
    if not display_idle(idle_frame, 3000):
        cv2.destroyAllWindows()
        return

    # Talking for 6 seconds
    if not display_talking(VIDEO_PATH, idle_frame, duration_s=6):
        cv2.destroyAllWindows()
        return

    # Idle again briefly before closing
    display_idle(idle_frame, 2000)

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
