import os
import sys
import subprocess
import tempfile
import shutil
import numpy as np
from scipy.io import wavfile

# Path to Piper voice files
VOICE_DIR = r"C:\\Users\\Bhagyesh\\Ai Interactive project\\piper-voices"
VOICE_MODEL = os.path.join(VOICE_DIR, "en_US-amy-medium.onnx")
PIPER_EXE = shutil.which("piper")

def speak_text(text):
    """Convert text to speech using Piper TTS and play the audio."""
    if not text or not text.strip():
        print("No text provided.")
        return

    if not os.path.exists(VOICE_MODEL):
        print(f"Voice model not found: {VOICE_MODEL}")
        return

    if PIPER_EXE is None:
        print("Error: 'piper' command not found in PATH. Make sure piper-tts is installed and in PATH.")
        return

    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        # Run piper to generate speech
        # Piper reads text from stdin and writes WAV to stdout
        print("Generating speech...")
        with open(wav_path, "wb") as wav_file:
            proc = subprocess.run(
                [
                    PIPER_EXE,
                    "--model", VOICE_MODEL,
                    "--config", VOICE_MODEL + ".json",
                    "--output_file", wav_path,
                    "--length_scale", "1.25",
                ],
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        if proc.returncode != 0:
            print(f"Piper error: {proc.stderr.decode('utf-8', errors='replace')}")
            return

        # Add a short silence pad at the start to prevent clipping/skipping
        pad_start_ms = 120  # milliseconds
        sample_rate, data = wavfile.read(wav_path)
        padding_samples = int(sample_rate * (pad_start_ms / 1000.0))
        silence = np.zeros(padding_samples, dtype=data.dtype)
        data = np.concatenate([silence, data])
        wavfile.write(wav_path, sample_rate, data)

        print(f"Audio saved: {wav_path}")

        # Play the audio
        play_audio(wav_path)

        # Delete the WAV file immediately after playback finishes
        try:
            os.remove(wav_path)
            print(f"Cleaned up: {wav_path}")
        except OSError as e:
            print(f"Could not delete WAV file: {e}")

    except Exception as e:
        print(f"Error during speech generation: {e}")
    finally:
        # Ensure cleanup if file still exists for any reason
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
                print(f"Cleaned up leftover file: {wav_path}")
        except OSError:
            pass


def play_audio(wav_path):
    """Play WAV audio using ffplay (via FFmpeg) or a fallback method."""
    # Try ffplay first (FFmpeg is installed and in PATH)
    ffplay = shutil.which("ffplay")
    if ffplay:
        print("Playing audio with ffplay...")
        try:
            subprocess.run(
                [ffplay, "-nodisp", "-autoexit", wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception as e:
            print(f"ffplay failed: {e}")

    # Fallback: use Python's built-in winsound on Windows
    try:
        import winsound
        print("Playing audio with winsound...")
        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
        return
    except ImportError:
        pass

    print("Could not play audio automatically. File saved at:", wav_path)


def main():
    print("=" * 50)
    print("Text-to-Speech Test - Piper TTS")
    print("Type text and press Enter to hear it.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        user_input = input("\\nText: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        speak_text(user_input)


if __name__ == "__main__":
    main()
