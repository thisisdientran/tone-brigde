from typing import Dict

import librosa
import numpy as np


def _estimate_average_pitch_hz(y: np.ndarray, sr: int) -> float:
    # Using YIN for a simple robust F0 estimate.
    f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    f0 = f0[np.isfinite(f0)]
    if len(f0) == 0:
        return 0.0
    return float(np.mean(f0))


def _estimate_energy(y: np.ndarray) -> float:
    # RMS over full signal as a basic loudness proxy.
    rms = librosa.feature.rms(y=y)[0]
    return float(np.mean(rms))


def _estimate_speaking_rate(y: np.ndarray, sr: int) -> float:
    # Approximate speaking rate from onset density (events/sec).
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    duration = len(y) / float(sr) if sr > 0 else 0.0
    if duration <= 0:
        return 0.0
    return float(len(onsets) / duration)


def extract_vocal_features(audio_path: str) -> Dict[str, float]:
    """
    Extract simple vocal features from original speech audio.
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    if y.size == 0:
        return {
            "average_pitch_hz": 0.0,
            "energy_rms": 0.0,
            "speaking_rate_events_per_sec": 0.0,
        }

    return {
        "average_pitch_hz": _estimate_average_pitch_hz(y, sr),
        "energy_rms": _estimate_energy(y),
        "speaking_rate_events_per_sec": _estimate_speaking_rate(y, sr),
    }


def map_features_to_tts_controls(features: Dict[str, float]) -> Dict[str, float]:
    """
    Map extracted features into simple TTS control parameters.
    These are heuristic and intentionally lightweight for prototype use.
    """
    pitch = features.get("average_pitch_hz", 0.0)
    energy = features.get("energy_rms", 0.0)

    # Pitch mapping: above ~180 Hz gets a slight boost.
    pitch_shift_semitones = 0.8 if pitch > 180 else 0.0

    # Energy mapping:
    # - high energy -> faster
    # - low energy -> slower
    if energy >= 0.08:
        tts_speed = 1.1
    elif energy <= 0.04:
        tts_speed = 0.9
    else:
        tts_speed = 1.0

    return {
        "tts_speed": tts_speed,
        "pitch_shift_semitones": pitch_shift_semitones,
    }
