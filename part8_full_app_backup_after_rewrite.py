import os
import time
import json
import subprocess
import tempfile
import shutil
import re
import threading
import queue
import traceback

import cv2
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from flask import Flask, render_template, Response
from groq import Groq
from dotenv import load_dotenv

# ============================ CONFIG ============================

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Whisper / STT config
WHISPER_MODEL = "whisper-large-v3-turbo"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.5
SILENCE_DURATION = 0.7
MAX_RECORD_DURATION = 25
ENERGY_THRESHOLD = 0.025  # Safe default floor; overwritten by calibration

# LLM config
LLM_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 80
SYSTEM_PROMPT = (
    "You must answer in exactly 2 to 3 complete sentences, "
    "using approximately 40 words. Do not give a shorter answer. "
    "Respond in plain spoken sentences only - no bullets, lists, or markdown."
)

MIN_AVG_ENERGY = 0.035

# Piper / TTS config
PROJECT_DIR = r"C:\\Users\\Bhagyesh\\Ai Interactive project"
VOICE_DIR = os.path.join(PROJECT_DIR, "piper-voices")
VOICE_MODEL = os.path.join(VOICE_DIR, "en_US-amy-medium.onnx")

# Find piper executable with explicit fallbacks
PIPER_EXE = None
for candidate in [
    shutil.which("piper"),
    shutil.which("piper.exe"),
    os.path.join(PROJECT_DIR, "venv", "Scripts", "piper.exe"),
    os.path.join(PROJECT_DIR, "piper", "piper.exe"),
]:
    if candidate and os.path.exists(candidate):
        PIPER_EXE = candidate
        break

TTS_LENGTH_SCALE = "1.0"
LEADING_SILENCE_MS = 700

AUDIO_DIR = os.path.join(PROJECT_DIR, "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
AI_REPLY_WAV = os.path.join(AUDIO_DIR, "ai_reply.wav")

# Flask + UI
STATIC_DIR = os.path.join(PROJECT_DIR, "static")
AVATAR_DIR = os.path.join(PROJECT_DIR, "avatar")
IDLE_PATH = os.path.join(STATIC_DIR, "idle.jpg")
VIDEO_PATH = os.path.join(STATIC_DIR, "loop_video.mp4")
STATIC_VIDEO_PATH = os.path.join(STATIC_DIR, "loop_video.mp4")

# Groq client
client = Groq(api_key=GROQ_API_KEY)

# Session memory (tuples: (user_text, ai_text))
conversation_history = []

# SSE queue for frontend updates
ui_queue = queue.Queue()

# Synchronization: frontend signals when TTS audio finishes playing
audio_finished_event = threading.Event()

app = Flask(__name__)

# ============================ HELPERS ============================


def rms_energy(audio):
    audio = audio.flatten()
    if len(audio) == 0:
        return 0.0
    return np.sqrt(np.mean(audio ** 2))


def record_audio_with_silence_detection():
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    silence_chunks_needed = int(SILENCE_DURATION / CHUNK_DURATION)
    max_chunks = int(MAX_RECORD_DURATION / CHUNK_DURATION)

    audio_chunks = []
    silence_counter = 0
    recording_started = False

    print("\n[Listening...] Speak now.")
    ui_queue.put({"state": "listening", "user_text": "", "ai_text": "Listening..."})

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=chunk_samples,
    )

    with stream:
        for chunk_index in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk.copy()
            energy = rms_energy(chunk)
            print(
                f"\r  [listening] chunk {chunk_index:3d} | energy: {energy:.5f} | "
                f"threshold: {ENERGY_THRESHOLD:.5f} | recording: {recording_started}",
                end="",
                flush=True,
            )

            if not recording_started and energy > ENERGY_THRESHOLD:
                recording_started = True
                print("\n[Voice detected, recording...]")

            if recording_started:
                audio_chunks.append(chunk)

                if energy < ENERGY_THRESHOLD:
                    silence_counter += 1
                else:
                    silence_counter = 0

                if silence_counter >= silence_chunks_needed:
                    print("\n[Silence detected, stopping recording.]")
                    break
        print()  # newline after the live loop ends

    if not audio_chunks:
        return None

    audio = np.concatenate(audio_chunks, axis=0).flatten()
    print(f"  -> Recorded {len(audio)/SAMPLE_RATE:.2f}s of audio")
    return audio


def audio_has_clear_speech(audio, sample_rate):
    duration = len(audio) / sample_rate
    avg_energy = rms_energy(audio)
    peak_energy = float(np.max(np.abs(audio)))
    print(f"  -> Audio duration: {duration:.2f}s | avg energy: {avg_energy:.5f} | peak energy: {peak_energy:.5f}")
    if duration < 1.0:
        return False, "too short"
    if avg_energy < MIN_AVG_ENERGY:
        return False, f"too quiet (avg {avg_energy:.5f} < min {MIN_AVG_ENERGY})"
    return True, "ok"


def calibrate_noise_threshold(calibration_seconds=2.0, margin=2.5):
    """Record ambient room noise and set ENERGY_THRESHOLD dynamically."""
    global ENERGY_THRESHOLD
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    chunks_to_read = int(calibration_seconds / CHUNK_DURATION)
    print(f"\n[Calibrating ambient noise for {calibration_seconds}s... please stay quiet.]")
    energies = []
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=chunk_samples,
    )
    with stream:
        for _ in range(chunks_to_read):
            chunk, _ = stream.read(chunk_samples)
            energies.append(rms_energy(chunk))
    ambient = float(np.median(energies))
    ENERGY_THRESHOLD = max(0.025, ambient * margin)
    print(f"[Calibration] ambient energy: {ambient:.5f} | margin: {margin}x | final ENERGY_THRESHOLD: {ENERGY_THRESHOLD:.5f}\n")
    return ENERGY_THRESHOLD


def save_audio_to_wav(audio, wav_path):
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(wav_path, SAMPLE_RATE, audio_int16)


def preprocess_audio(audio, sample_rate):
    if len(audio) == 0:
        return audio

    peak = np.max(np.abs(audio))
    if peak > 0:
        target_peak = 0.70
        audio = audio * (target_peak / peak)

    cutoff = 80.0
    rc = 1.0 / (2.0 * np.pi * cutoff)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)

    filtered = np.zeros_like(audio)
    filtered[0] = audio[0]
    for i in range(1, len(audio)):
        filtered[i] = alpha * (audio[i] - audio[i - 1]) + (1 - alpha) * filtered[i - 1]

    return filtered


def transcribe_with_groq(audio):
    audio = preprocess_audio(audio, SAMPLE_RATE)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        save_audio_to_wav(audio, wav_path)
        with open(wav_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=WHISPER_MODEL,
                language="en",
            )
        return transcription.text.strip()
    finally:
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except OSError:
            pass


def trim_incomplete_sentence(text):
    text = text.strip()
    if not text:
        return text
    if text.endswith((".", "!", "?")):
        return text
    last_period = text.rfind(".")
    last_exclaim = text.rfind("!")
    last_question = text.rfind("?")
    last_end = max(last_period, last_exclaim, last_question)
    if last_end == -1:
        return text
    return text[: last_end + 1].strip()


def fix_word_merging(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-z])", r"\1 \2", text)
    text = " ".join(text.split())
    return text


def get_ai_reply(user_text, history):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_msg, ai_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})
        messages.append({"role": "user", "content": user_text})

        chat_completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=MAX_TOKENS,
        )
        reply = chat_completion.choices[0].message.content.strip()
        reply = reply.replace("*", "").replace("#", "").replace("-", " ").replace("`", "").replace("\n", " ")
        reply = fix_word_merging(reply)
        reply = " ".join(reply.split())
        reply = trim_incomplete_sentence(reply)
        return reply
    except Exception as e:
        return f"Error: {e}"


def speak_text_to_file(text, output_path):
    if not text or not text.strip():
        print("[No text to speak.]")
        return False

    if not os.path.exists(VOICE_MODEL):
        print(f"[Voice model not found: {VOICE_MODEL}]")
        return False

    if PIPER_EXE is None or not os.path.exists(PIPER_EXE):
        print(f"[Error: Piper executable not found at '{PIPER_EXE}'. TTS disabled.]")
        return False

    print(f"[TTS] Using piper executable: {PIPER_EXE}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav_path = tmp.name

    try:
        proc = subprocess.run(
            [
                PIPER_EXE,
                "--model", VOICE_MODEL,
                "--config", VOICE_MODEL + ".json",
                "--output_file", tmp_wav_path,
                "--length_scale", TTS_LENGTH_SCALE,
            ],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if proc.returncode != 0:
            stderr_text = proc.stderr.decode("utf-8", errors="replace")
            print(f"[Piper error (code {proc.returncode}): {stderr_text}]")
            return False

        sample_rate, data = wavfile.read(tmp_wav_path)

        # Add leading silence buffer
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        silence_samples = int(sample_rate * (LEADING_SILENCE_MS / 1000.0))
        silence = np.zeros(silence_samples, dtype=data.dtype)
        data = np.concatenate([silence, data])

        audio_int16 = (data * 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_int16)
        print(f"  -> TTS audio saved to: {output_path}")
        return True

    finally:
        try:
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)
        except OSError:
            pass


# ============================ VOICE LOOP ============================


def voice_pipeline_loop():
    print("=" * 60)
    print(" AI Companion - Full Integrated App (Part 8)")
    print("=" * 60)
    print(f" STT: Groq Whisper | Model: {WHISPER_MODEL}")
    print(f" LLM: Groq | Model: {LLM_MODEL} | max_tokens: {MAX_TOKENS}")
    print(f" TTS: Piper | Voice: en_US-amy-medium | length_scale: {TTS_LENGTH_SCALE}")
    print(f" Piper executable: {PIPER_EXE}")
    print(f" SILENCE_DURATION: {SILENCE_DURATION}s")
    print(f" Audio playback: Browser <audio> only")
    print(" Say 'exit' to stop, or 'forget everything' to reset memory.")
    print("=" * 60)
    print()

    try:
        calibrate_noise_threshold()
    except Exception as e:
        print(f"[Calibration error: {e}. Using default threshold {ENERGY_THRESHOLD}]")

    while True:
        try:
            audio = record_audio_with_silence_detection()

            if audio is None or len(audio) == 0:
                print("[No audio captured, listening again...]")
                continue

            audio_duration = len(audio) / SAMPLE_RATE
            avg_energy = rms_energy(audio)
            print(f"  -> Audio duration: {audio_duration:.2f}s | avg energy: {avg_energy:.5f}")

            clear_speech, reason = audio_has_clear_speech(audio, SAMPLE_RATE)
            if not clear_speech:
                print(f"[No clear speech detected ({reason}), listening again...]")
                continue

            stt_start = time.time()
            transcribed = transcribe_with_groq(audio)
            stt_elapsed = time.time() - stt_start

            if not transcribed:
                print("[No speech detected, listening again...]")
                continue

            print(f"You said: {transcribed}")
            print(f"  -> Transcription time: {stt_elapsed:.2f}s")

            if transcribed.lower() in ("exit", "quit", "stop"):
                print("Goodbye!")
                ui_queue.put({"state": "stopped", "user_text": "", "ai_text": "Goodbye!"})
                break

            if transcribed.lower() in ("forget everything", "new conversation", "clear memory", "reset"):
                conversation_history.clear()
                print("[Memory cleared.]")
                ui_queue.put({"state": "listening", "user_text": "", "ai_text": "Memory cleared. Listening..."})
                continue

            # Generate AI reply
            ai_start = time.time()
            try:
                reply = get_ai_reply(transcribed, conversation_history)
            except Exception as e:
                print(f"[LLM error: {e}]")
                reply = "I'm having trouble connecting to my language model. Please try again."
            ai_elapsed = time.time() - ai_start

            if reply.startswith("Error:"):
                print(f"[AI reply error: {reply}]")
                ui_queue.put({"state": "listening", "user_text": transcribed, "ai_text": reply})
                continue

            reply_word_count = len(reply.split())
            print(f"AI: {reply}")
            print(f"  -> AI reply word count: {reply_word_count} words")
            print(f"  -> AI thinking time: {ai_elapsed:.2f}s")

            # Save turn to memory
            conversation_history.append((transcribed, reply))

            # Generate TTS audio
            tts_start = time.time()
            try:
                tts_ok = speak_text_to_file(reply, AI_REPLY_WAV)
            except Exception as e:
                print(f"[TTS error: {e}]")
                tts_ok = False
            tts_elapsed = time.time() - tts_start

            if not tts_ok:
                print("[TTS failed, continuing without audio.]")
                ui_queue.put({"state": "listening", "user_text": transcribed, "ai_text": reply})
                continue

            print(f"  -> TTS generation time: {tts_elapsed:.2f}s")

            # Notify frontend to play audio and show talking avatar
            ui_queue.put({
                "state": "speaking",
                "user_text": transcribed,
                "ai_text": reply,
                "audio_url": "/static/audio/ai_reply.wav?" + str(int(time.time()))
            })

            # Wait for frontend to confirm audio finished (with timeout fallback)
            print("[Waiting for frontend audio finished signal...]")
            audio_finished_event.clear()
            finished = audio_finished_event.wait(timeout=30)
            if finished:
                print("[Frontend confirmed audio finished. Resuming listening.]")
            else:
                print("[Timeout: no audio finished signal from frontend. Resuming listening.]")

        except Exception as e:
            print(f"\n[UNEXPECTED ERROR in voice_pipeline_loop: {type(e).__name__}: {e}]")
            traceback.print_exc()
            print("[Continuing to listen...]")
            try:
                ui_queue.put({"state": "listening", "user_text": "", "ai_text": "Sorry, something went wrong. I'm listening again."})
            except Exception:
                pass
            continue


# ============================ FLASK ROUTES ============================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/audio_finished", methods=["POST"])
def audio_finished():
    audio_finished_event.set()
    return {"status": "ok"}, 200


@app.route("/events")
def events():
    def generate():
        while True:
            try:
                msg = ui_queue.get(timeout=30)
                data = f"data: {json.dumps(msg)}\n\n"
                yield data
            except queue.Empty:
                yield f"data: {json.dumps({'state': 'keepalive'})}\n\n"
    return Response(generate(), mimetype="text/event-stream")


# ============================ MAIN ============================


def prepare_static_files():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    if not os.path.exists(VIDEO_PATH):
        print(f"[WARNING] Avatar video not found: {VIDEO_PATH}")
    else:
        if VIDEO_PATH != STATIC_VIDEO_PATH and not os.path.exists(STATIC_VIDEO_PATH):
            shutil.copy(VIDEO_PATH, STATIC_VIDEO_PATH)
            print(f"[INFO] Copied video to static: {STATIC_VIDEO_PATH}")

    if not os.path.exists(IDLE_PATH):
        cap = cv2.VideoCapture(VIDEO_PATH)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(IDLE_PATH, frame)
            print(f"[INFO] Extracted idle frame: {IDLE_PATH}")
        cap.release()


if __name__ == "__main__":
    prepare_static_files()
    voice_thread = threading.Thread(target=voice_pipeline_loop, daemon=True)
    voice_thread.start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
