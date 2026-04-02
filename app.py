import traceback
from typing import Tuple

import gradio as gr

from asr import transcribe_audio
from features import extract_vocal_features, map_features_to_tts_controls
from translate import LANGUAGE_OPTIONS, translate_text
from tts import synthesize_translated_speech
from utils import pretty_json


def run_tonebridge(audio_path: str, target_language: str) -> Tuple[str, str, str, str]:
    if not audio_path:
        raise gr.Error("Please upload an audio file.")

    try:
        transcript = transcribe_audio(audio_path)
        translated = translate_text(transcript, target_language)
        features = extract_vocal_features(audio_path)
        controls = map_features_to_tts_controls(features)
        language_code = LANGUAGE_OPTIONS[target_language]

        output_audio = synthesize_translated_speech(
            text=translated,
            target_language_code=language_code,
            controls=controls,
            speaker_wav=audio_path,  # prototype: use source voice as optional reference
        )

        features_view = {
            **features,
            **controls,
            "target_language_code": language_code,
        }

        return transcript, translated, pretty_json(features_view), output_audio
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"Pipeline failed: {exc}") from exc


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ToneBridge Prototype") as demo:
        gr.Markdown("## ToneBridge\nUpload speech, translate it, and synthesize styled output.")

        with gr.Row():
            audio_in = gr.Audio(label="Input Speech Audio", type="filepath")
            target_lang = gr.Dropdown(
                label="Target Language",
                choices=list(LANGUAGE_OPTIONS.keys()),
                value="Spanish",
            )

        run_btn = gr.Button("Run ToneBridge")

        transcript_out = gr.Textbox(label="Transcript", lines=5)
        translated_out = gr.Textbox(label="Translated Text", lines=5)
        features_out = gr.Textbox(label="Detected Features + TTS Controls", lines=12)
        audio_out = gr.Audio(label="Generated Audio Output", type="filepath")

        run_btn.click(
            fn=run_tonebridge,
            inputs=[audio_in, target_lang],
            outputs=[transcript_out, translated_out, features_out, audio_out],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
