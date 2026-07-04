import sys
import time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

MODEL_SIZE = "base"  # Better accuracy than 'tiny'; still fast enough on CPU
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.5
SILENCE_DURATION = 1.0
MAX_RECORD_DURATION = 25
ENERGY_THRESHOLD = 0.015


def get_device():
    """Force CPU for Whisper to keep GPU VRAM free for Ollama/phi3."""
    return "cpu"


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

    print("Listening... Speak now.")

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
                print("Voice detected, recording...")

            if recording_started:
                audio_chunks.append(chunk)

                if energy < ENERGY_THRESHOLD:
                    silence_counter += 1
                else:
                    silence_counter = 0

                if silence_counter >= silence_chunks_needed:
                    print("Silence detected, stopping recording.")
                    break

    if not audio_chunks:
        return None

    audio = np.concatenate(audio_chunks, axis=0).flatten()
    return audio


def main():
    device = get_device()

    # Configure compute_type and CPU threads for maximum speed
    if device == "cuda":
        compute_type = "float16"
        cpu_threads = 0  # ignored on GPU
        num_workers = 1
    else:
        compute_type = "int8"
        cpu_threads = 8  # adjust based on your CPU core count
        num_workers = 2

    print("=" * 50)
    print("Speech-to-Text Test - faster-whisper")
    print(f"Model: {MODEL_SIZE}, Device: {device}, Compute: {compute_type}")
    print("Press Enter to record. Stops on silence or max time.")
    print("Type 'exit' to quit.")
    print("=" * 50)

    print(f"\nLoading Whisper model '{MODEL_SIZE}' on {device}...")
    print("Model is loaded ONCE here and reused for every recording.")

    # Build model kwargs carefully: faster-whisper/CTranslate2 rejects
    # intra_threads=None when device='cuda'.
    model_kwargs = {
        "model_size_or_path": MODEL_SIZE,
        "device": device,
        "compute_type": compute_type,
        "num_workers": num_workers,
    }
    if device != "cuda":
        model_kwargs["cpu_threads"] = cpu_threads

    model = WhisperModel(**model_kwargs)
    print("Model loaded. Ready.")

    while True:
        user_input = input("\nPress Enter to record, or type 'exit': ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        audio = record_audio_with_silence_detection()
        if audio is None or len(audio) == 0:
            print("No audio captured. Try again.")
            continue

        print("Transcribing...")
        start_time = time.time()
        segments, _info = model.transcribe(
            audio,
            language="en",
            beam_size=5,
        )
        text = " ".join([segment.text for segment in segments]).strip()
        elapsed = time.time() - start_time

        if text:
            print(f"You said: {text}")
            print(f"(Transcription took {elapsed:.2f}s)")
        else:
            print("No speech detected. Try again.")


if __name__ == "__main__":
    main()
