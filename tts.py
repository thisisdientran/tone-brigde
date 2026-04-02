from pathlib import Path
from typing import Dict, Optional

import librosa
import soundfile as sf
from TTS.api import TTS

from utils import ensure_output_dir


_TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
_TTS_INSTANCE: Optional[TTS] = None


def _get_tts() -> TTS:
    global _TTS_INSTANCE
    if _TTS_INSTANCE is None:
        _TTS_INSTANCE = TTS(model_name=_TTS_MODEL_NAME)
    return _TTS_INSTANCE


def _apply_pitch_shift_if_needed(
    wav_path: str,
    pitch_shift_semitones: float,
) -> None:
    if abs(pitch_shift_semitones) < 1e-6:
        return
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch_shift_semitones)
    sf.write(wav_path, y_shifted, sr)


def synthesize_translated_speech(
    text: str,
    target_language_code: str,
    controls: Dict[str, float],
    speaker_wav: Optional[str] = None,
) -> str:
    """
    Synthesize translated text using Coqui TTS.

    Notes for future extension:
    - Emotion control can be inserted here by selecting expressive models or style tokens.
    - Voice cloning can be improved by passing a high-quality reference `speaker_wav`
      or replacing this backend with a stronger cloning-capable model.
    """
    output_dir = ensure_output_dir()
    output_path = Path(output_dir) / "translated_output.wav"

    tts = _get_tts()
    speed = float(controls.get("tts_speed", 1.0))
    pitch_shift = float(controls.get("pitch_shift_semitones", 0.0))

    # XTTS supports language + optional speaker reference wav.
    # `speed` availability may vary by backend/model; this prototype uses it directly.
    tts.tts_to_file(
        text=text,
        file_path=str(output_path),
        language=target_language_code,
        speaker_wav=speaker_wav,
        speed=speed,
    )

    # Fallback style control: post-process pitch if direct control is unavailable.
    _apply_pitch_shift_if_needed(str(output_path), pitch_shift)
    return str(output_path)
