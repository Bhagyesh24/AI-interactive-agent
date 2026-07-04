import os
import sys
import time
import subprocess
import tempfile
import shutil
import re
import winsound
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
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
ENERGY_THRESHOLD = 0.04

# LLM config
LLM_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 80
SYSTEM_PROMPT = (
    "You must answer in exactly 2 to 3 complete sentences, "
    "using approximately 40 words. Do not give a shorter answer. "
    "Respond in plain spoken sentences only - no bullets, lists, or markdown."
)

MIN_AUDIO_DURATION = 0.5  # seconds
MIN_AVG_ENERGY = 0.035  # reject near-silent/background-noise clips

# Piper / TTS config
PROJECT_DIR = r"C:\\Users\\Bhagyesh\\Ai Interactive project"
VOICE_DIR = os.path.join(PROJECT_DIR, "piper-voices")
VOICE_MODEL = os.path.join(VOICE_DIR, "en_US-amy-medium.onnx")
PIPER_EXE = shutil.which("piper")
TTS_LENGTH_SCALE = "1.0"
LEADING_SILENCE_MS = 700

# Groq client
client = Groq(api_key=GROQ_API_KEY)

# Session memory: stores every user/assistant exchange
conversation_history = []

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

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=chunk_samples,
    )

    with stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk.copy()
            energy = rms_energy(chunk)

            if not recording_started and energy > ENERGY_THRESHOLD:
                recording_started = True
                print("[Voice detected, recording...]")

            if recording_started:
                audio_chunks.append(chunk)

                if energy < ENERGY_THRESHOLD:
                    silence_counter += 1
                else:
                    silence_counter = 0

                if silence_counter >= silence_chunks_needed:
                    print("[Silence detected, stopping recording.]")
                    break

    if not audio_chunks:
        return None

    audio = np.concatenate(audio_chunks, axis=0).flatten()
    return audio


def audio_has_clear_speech(audio, sample_rate):
    """Check if recorded audio has enough duration and volume to be real speech."""
    duration = len(audio) / sample_rate
    avg_energy = rms_energy(audio)
    print(f"  -> Audio duration: {duration:.2f}s | avg energy: {avg_energy:.5f}")
    if duration < 0.5:
        return False, "too short"
    if avg_energy < 0.01:
        return False, "too quiet"
    return True, "ok"


def save_audio_to_wav(audio, wav_path):
    """Save float32 audio to 16-bit WAV file."""
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(wav_path, SAMPLE_RATE, audio_int16)


def preprocess_audio(audio, sample_rate):
    """Basic audio preprocessing for Bluetooth/low-quality mics.

    1. Normalize volume to peak -3 dB
    2. High-pass filter to reduce low-frequency rumble/noise
    """
    if len(audio) == 0:
        return audio

    # Normalize: scale to 70% of max int16 range to avoid clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        target_peak = 0.70
        audio = audio * (target_peak / peak)

    # Simple high-pass filter (first-order IIR) at ~80 Hz
    # y[n] = a0 * (y[n-1] + x[n] - x[n-1])
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
    """Send recorded audio to Groq Whisper API and return transcribed text."""
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
    """Fix common word-merging artifacts like 'wellknown' -> 'well known'."""
    # Insert space between lowercase and uppercase: wellKnown -> well Known
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Insert space between lowercase and digit or vice versa if needed
    text = re.sub(r"([a-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-z])", r"\1 \2", text)
    # Collapse multiple spaces
    text = " ".join(text.split())
    return text


def get_ai_reply(user_text, conversation_history):
    """Send full conversation history to Groq LLM and return a reply."""
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in conversation_history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": user_text})

        chat_completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=MAX_TOKENS,
        )
        reply = chat_completion.choices[0].message.content.strip()
        # Safety strip for markdown/list symbols (replace - with space, don't merge words)
        reply = reply.replace("*", "").replace("#", "").replace("-", " ").replace("`", "").replace("\n", " ")
        reply = fix_word_merging(reply)
        reply = " ".join(reply.split())
        reply = trim_incomplete_sentence(reply)
        return reply
    except Exception as e:
        return f"Error: {e}"


def play_audio(audio_data, sample_rate, debug_wav_path=None):
    """Play audio array using Windows native winsound to avoid sounddevice underrun."""
    try:
        # Add 700ms leading silence buffer as safety margin
        silence_samples = int(sample_rate * (LEADING_SILENCE_MS / 1000.0))
        silence = np.zeros(silence_samples, dtype=audio_data.dtype)
        audio_data = np.concatenate([silence, audio_data])

        # Save to a WAV file first (we know Windows can play this perfectly)
        if debug_wav_path is None:
            debug_wav_path = os.path.join(PROJECT_DIR, "debug_last_output.wav")
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(debug_wav_path, sample_rate, audio_int16)

        print(f"  -> Playing via Windows native winsound: {debug_wav_path}")
        winsound.PlaySound(debug_wav_path, winsound.SND_FILENAME)

        # Clean up debug file after playback
        try:
            if os.path.exists(debug_wav_path):
                os.remove(debug_wav_path)
                print("  -> debug_last_output.wav cleaned up")
        except OSError as e:
            print(f"  -> cleanup warning: {e}")
    except Exception as e:
        print(f"[playback failed: {e}]")



def speak_text(text):
    """Convert text to speech using Piper and play it."""
    if not text or not text.strip():
        print("[No text to speak.]")
        return

    if not os.path.exists(VOICE_MODEL):
        print(f"[Voice model not found: {VOICE_MODEL}]")
        return

    if PIPER_EXE is None:
        print("[Error: 'piper' command not found in PATH.]")
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        proc = subprocess.run(
            [
                PIPER_EXE,
                "--model", VOICE_MODEL,
                "--config", VOICE_MODEL + ".json",
                "--output_file", wav_path,
                "--length_scale", TTS_LENGTH_SCALE,
            ],
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if proc.returncode != 0:
            print(f"[Piper error: {proc.stderr.decode('utf-8', errors='replace')}]")
            return

        sample_rate, data = wavfile.read(wav_path)

        # DEBUG: Save raw Piper output before any modification
        debug_path = os.path.join(os.path.dirname(VOICE_MODEL), "debug_last_output.wav")
        try:
            wavfile.write(debug_path, sample_rate, data)
            print(f"  -> DEBUG raw Piper output saved to: {debug_path}")
        except Exception as e:
            print(f"  -> DEBUG save failed: {e}")

        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max

        play_audio(data, sample_rate)

    finally:
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except OSError:
            pass


# ============================ MAIN LOOP ============================


def main():
    print("=" * 60)
    print(" Real-Time AI Companion - Groq API Version (Part 4)")
    print("=" * 60)
    print(f" STT: Groq Whisper | Model: {WHISPER_MODEL}")
    print(f" LLM: Groq | Model: {LLM_MODEL} | max_tokens: {MAX_TOKENS}")
    print(f" TTS: Piper | Voice: en_US-amy-medium | length_scale: {TTS_LENGTH_SCALE}")
    print(f" SILENCE_DURATION: {SILENCE_DURATION}s | ENERGY_THRESHOLD: {ENERGY_THRESHOLD}")
    print(f" Leading silence buffer: {LEADING_SILENCE_MS}ms")
    print(" Say 'exit' at any time to stop.")
    print("=" * 60)

    print("\nReady for conversation.\n")

    turn_count = 1

    while True:
        print(f"--- Turn {turn_count} ---")
        listen_start = time.time()
        audio = record_audio_with_silence_detection()
        listen_elapsed = time.time() - listen_start

        if audio is None or len(audio) == 0:
            print("[No audio captured, listening again...]\n")
            continue

        # 2. Check for clear speech before sending to Whisper
        audio_duration = len(audio) / SAMPLE_RATE
        avg_energy = rms_energy(audio)
        print(f"  -> Audio duration: {audio_duration:.2f}s | avg energy: {avg_energy:.5f}")

        if audio_duration < MIN_AUDIO_DURATION or avg_energy < MIN_AVG_ENERGY:
            print("[No clear speech detected, listening again...]\n")
            continue

        # 3. Valid speech detected - now start an official turn
        turn_count += 1
        print(f"--- Turn {turn_count} ---")
        total_start = time.time()
        print(f"  -> Listening time: {listen_elapsed:.2f}s")
        stt_start = time.time()
        transcribed = transcribe_with_groq(audio)
        stt_elapsed = time.time() - stt_start

        if not transcribed or not clear_speech:
            if not clear_speech:
                print("[No clear speech detected, try again]\n")
            else:
                print("[No speech detected, try again.]\n")
            continue

        turn_count += 1

        print(f"You said: {transcribed}")
        print(f"  -> Transcription time: {stt_elapsed:.2f}s")

        if transcribed.lower() in ("exit", "quit", "stop"):
            print("Goodbye!")
            break

        # Reset memory command
        if transcribed.lower() in ("forget everything", "new conversation", "clear memory", "reset"):
            conversation_history.clear()
            print("[Memory cleared. Starting a new conversation.]\n")
            continue

        # 3. AI reply with Groq (full history)
        llm_start = time.time()
        reply = get_ai_reply(transcribed, conversation_history)
        llm_elapsed = time.time() - llm_start

        print(f"AI: {reply}")
        print(f"  -> AI reply word count: {len(reply.split())} words")
        print(f"  -> AI thinking time: {llm_elapsed:.2f}s")

        if reply.startswith("Error:"):
            print()
            continue

        # Store this turn in memory
        conversation_history.append({"user": transcribed, "assistant": reply})

        # 4. TTS and play
        tts_start = time.time()
        speak_text(reply)
        tts_elapsed = time.time() - tts_start

        total_elapsed = time.time() - total_start

        print("\n" + "=" * 40)
        print("  TURN SUMMARY")
        print(f"  Listening time: {listen_elapsed:.2f}s")
        print(f"  Transcription time: {stt_elapsed:.2f}s")
        print(f"  AI thinking time: {llm_elapsed:.2f}s")
        print(f"  TTS + playback time: {tts_elapsed:.2f}s")
        print(f"  Total turn time: {total_elapsed:.2f}s")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
