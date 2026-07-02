"""
Stage D — voice I/O on the Mac.
  listen()  : record from the mic and transcribe with faster-whisper (STT)
  speak()   : speak text aloud with the macOS built-in `say` (TTS)

First run downloads the small Whisper model (~75 MB). Mic access will prompt
for permission the first time.

Quick test:
    python brain/voice.py
"""

import os
import queue
import sys
import numpy as np

SAMPLE_RATE = 16000          # Whisper expects 16 kHz
_model = None                # lazy-loaded Whisper model

# Robot voice. Swap VOICE for a different macOS voice:
#   "Fred"    – classic robotic computer voice (default)
#   "Zarvox"  – full sci-fi robot (more synthetic)
#   "Daniel"  – calm British AI-assistant
#   "Samantha"– clear neutral
# List all with:  say -v '?'
VOICE = "Fred"
RATE  = 190                  # words per minute


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        # base.en: good accuracy for short commands, fast on CPU
        _model = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _model


# ---------------------------------------------------------------- TTS
def speak(text):
    """Speak text aloud using the macOS `say` command (blocking)."""
    text = (text or "").replace('"', "'")
    if text:
        os.system(f'say -v "{VOICE}" -r {RATE} "{text}"')


# ---------------------------------------------------------------- STT
def listen(seconds=4.0, verbose=True):
    """Record `seconds` of audio from the mic and return the transcript."""
    import sounddevice as sd
    if verbose:
        print(f"🎤 listening for {seconds:.0f}s ... speak now")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()

    segments, _ = _get_model().transcribe(audio, language="en", beam_size=1)
    text = " ".join(s.text for s in segments).strip()
    if verbose:
        print(f"📝 heard: {text!r}")
    return text


# Phrases Whisper invents from silence / background noise — never real commands.
_HALLUCINATIONS = {
    "the", "you", "bye", "bye.", "thank you", "thank you.", "thanks",
    "thanks for watching", "thanks for watching.", ".", "uh", "um", "so",
}


def _clean_transcript(text):
    """Drop Whisper hallucinations (silence → 'The The The The...', etc.)."""
    t = (text or "").strip()
    if len(t) <= 1:
        return ""
    words = t.lower().replace(",", " ").split()
    # repeated single token ("the the the the") is always a hallucination
    if len(words) >= 3 and len(set(words)) == 1:
        return ""
    if t.lower().strip(" .!?") in _HALLUCINATIONS:
        return ""
    return t


def listen_until_silence(max_seconds=6.0, silence_rms=0.01, verbose=True):
    """Record until a short silence is detected (more natural than fixed time)."""
    import sounddevice as sd
    if verbose:
        print("🎤 listening ... speak now")
    q = queue.Queue()

    def cb(indata, frames, time_info, status):
        q.put(indata.copy())

    chunks, silent_run, started = [], 0, False
    block = int(0.1 * SAMPLE_RATE)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=block, callback=cb):
        for _ in range(int(max_seconds / 0.1)):
            data = q.get().flatten()
            chunks.append(data)
            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms > silence_rms:
                started, silent_run = True, 0
            elif started:
                silent_run += 1
                if silent_run >= 6:          # ~0.6 s of trailing silence
                    break

    # Energy gate: if no speech was ever detected, don't transcribe at all —
    # Whisper hallucinates text ("The The The...") from silence.
    if not started:
        return ""

    audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    segments, _ = _get_model().transcribe(audio, language="en", beam_size=1)
    text = _clean_transcript(" ".join(s.text for s in segments))
    if verbose and text:
        print(f"📝 heard: {text!r}")
    return text


if __name__ == "__main__":
    speak("Voice test. Say something after the prompt.")
    heard = listen_until_silence()
    speak(f"You said: {heard}" if heard else "I didn't hear anything.")
