# ToneBridge (Prototype)

ToneBridge is a local Python prototype that:

1. Accepts an input speech audio file
2. Transcribes it with OpenAI Whisper
3. Translates the transcript to a target language
4. Extracts vocal features from the original audio
5. Maps vocal features to simple TTS style controls
6. Synthesizes translated speech output with Coqui TTS
7. Exposes everything in a minimal Gradio UI

## Project structure

- `app.py` - Gradio UI and pipeline orchestration
- `asr.py` - Whisper transcription
- `translate.py` - text translation
- `features.py` - vocal feature extraction + style mapping
- `tts.py` - Coqui TTS synthesis + pitch-shift fallback
- `utils.py` - shared helpers
- `requirements.txt` - dependencies

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Run

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal.

## Notes on prototype behavior

- Feature extraction currently includes:
  - average pitch (Hz)
  - RMS energy
  - speaking-rate estimate (onset events/sec)
- TTS style mapping is intentionally simple:
  - high pitch -> slight positive pitch shift
  - high energy -> faster speaking speed
  - low energy -> slower speaking speed
- Coqui model controls vary across versions/models. This prototype uses:
  - direct `speed` where available
  - post-processing pitch shift fallback via librosa

## Future extension points

- **Emotion detection**:
  - Add a module that predicts emotion labels from acoustic/prosodic features
  - Route those labels into model-specific style tokens or prompts in `tts.py`
- **Voice cloning**:
  - Improve speaker embedding/reference handling in `tts.py`
  - Optionally replace backend with a dedicated voice-cloning model

## Troubleshooting

- If Coqui TTS model download is slow/fails, retry with stable internet.
- If audio synthesis fails for a language, test with another target language first.
- If you see OpenAI authentication errors, re-check `OPENAI_API_KEY`.
# tone-brigde
