from __future__ import annotations

import argparse
import gc
import threading
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

SHARE_LINK = False
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860

MODEL_IDS = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

SPEAKERS = ["Serena", "Vivian", "Ono_Anna", "Sohee", "Aiden", "Dylan", "Eric", "Ryan", "Uncle_Fu"]

current_model: Any | None = None
current_model_type: str | None = None
_MODEL_LOCK = threading.RLock()
_OUTPUT_DIR = tempfile.TemporaryDirectory(prefix="qwen3_tts_local_")

CUSTOM_CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
}
.app-shell {
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    padding: 18px 20px 14px;
    margin: 10px 0 12px;
    background: var(--background-fill-primary);
    box-shadow: 0 10px 35px rgba(15, 23, 42, 0.08);
}
.app-title {
    margin: 0;
    font-size: clamp(1.55rem, 3vw, 2.15rem);
    font-weight: 720;
    letter-spacing: -0.03em;
}
.app-subtitle {
    margin: 5px 0 0;
    color: var(--body-text-color-subdued);
    font-size: 0.94rem;
}
.status-line {
    min-height: 34px;
}
.compact-note p {
    margin: 0.15rem 0;
    font-size: 0.86rem;
    color: var(--body-text-color-subdued);
}
.footer-note {
    text-align: center;
    font-size: 0.82rem;
    color: var(--body-text-color-subdued);
    margin-top: 8px;
}
@media (max-width: 768px) {
    .gradio-container { padding: 8px !important; }
    .app-shell { padding: 14px 14px 10px; margin-top: 4px; }
}
"""

THEME_INIT_JS = """
() => {
    const stored = localStorage.getItem('qwen3tts-theme');
    const dark = stored === 'dark';
    document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
    if (dark) document.body.classList.add('dark');
}
"""

THEME_TOGGLE_JS = """
() => {
    const isDark = document.body.classList.toggle('dark');
    localStorage.setItem('qwen3tts-theme', isDark ? 'dark' : 'light');
}
"""


def _import_runtime():
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    return torch, sf, Qwen3TTSModel


def _trimmed(value: str | None) -> str:
    return (value or "").strip()


def validate_clone_inputs(text: str | None, reference_audio: str | None, transcript: str | None, fast_mode: bool) -> str | None:
    if not _trimmed(text):
        return "Enter text to synthesize."
    if not reference_audio:
        return "Add a reference audio file."
    if not fast_mode and not _trimmed(transcript):
        return "A reference transcript is required when x-vector-only mode is disabled."
    return None


def validate_text_input(text: str | None) -> str | None:
    if not _trimmed(text):
        return "Enter text to synthesize."
    return None


def validate_language(language: str | None) -> str | None:
    if language not in LANGUAGES:
        return "Select a supported language."
    return None


def configure_cuda(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA-capable NVIDIA GPU was detected by PyTorch.")

    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    except (AttributeError, TypeError):
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True


def _preferred_dtype(torch: Any):
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def unload_model() -> str:
    global current_model, current_model_type

    with _MODEL_LOCK:
        current_model = None
        current_model_type = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return "Model unloaded. GPU cache released where supported."


def load_model(model_type: str):
    global current_model, current_model_type

    if model_type not in MODEL_IDS:
        raise ValueError(f"Unknown model type: {model_type}")

    with _MODEL_LOCK:
        if current_model is not None and current_model_type == model_type:
            return current_model

        torch, _, model_class = _import_runtime()
        configure_cuda(torch)

        current_model = None
        current_model_type = None
        gc.collect()
        torch.cuda.empty_cache()

        started = time.perf_counter()
        try:
            model = model_class.from_pretrained(
                MODEL_IDS[model_type],
                dtype=_preferred_dtype(torch),
                device_map="cuda:0",
                attn_implementation="sdpa",
            )
        except Exception:
            current_model = None
            current_model_type = None
            gc.collect()
            torch.cuda.empty_cache()
            raise

        current_model = model
        current_model_type = model_type
        print(f"Loaded {model_type} model in {time.perf_counter() - started:.1f}s")
        return model


def _save_waveform(waveform: Any, sample_rate: int) -> str:
    _, sf, _ = _import_runtime()
    output = Path(_OUTPUT_DIR.name) / f"{uuid.uuid4().hex}.wav"
    sf.write(output, waveform, sample_rate)
    return str(output)


def _friendly_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return f"Error: {message}"


def voice_clone(text: str, reference_audio: str, ref_transcript: str, use_fast_mode: bool, language: str):
    error = validate_clone_inputs(text, reference_audio, ref_transcript, use_fast_mode)
    if error:
        return None, error
    language_error = validate_language(language)
    if language_error:
        return None, language_error

    try:
        torch, _, _ = _import_runtime()
        model = load_model("base")
        transcript = _trimmed(ref_transcript)

        prompt_items = model.create_voice_clone_prompt(
            ref_audio=reference_audio,
            ref_text=None if use_fast_mode else transcript,
            x_vector_only_mode=use_fast_mode,
        )

        with torch.inference_mode():
            wavs, sample_rate = model.generate_voice_clone(
                text=_trimmed(text),
                language=language or "Auto",
                voice_clone_prompt=prompt_items,
            )

        return _save_waveform(wavs[0], sample_rate), "Generated with the Base voice-cloning model."
    except Exception as exc:
        return None, _friendly_error(exc)


def custom_voice(text: str, voice_name: str, instruction: str, language: str):
    error = validate_text_input(text)
    if error:
        return None, error
    if voice_name not in SPEAKERS:
        return None, "Select a supported speaker."
    language_error = validate_language(language)
    if language_error:
        return None, language_error

    try:
        torch, _, _ = _import_runtime()
        model = load_model("custom")
        instruction = _trimmed(instruction)

        kwargs = {
            "text": _trimmed(text),
            "speaker": voice_name,
            "language": language or "Auto",
        }
        if instruction:
            kwargs["instruct"] = instruction

        with torch.inference_mode():
            wavs, sample_rate = model.generate_custom_voice(**kwargs)

        return _save_waveform(wavs[0], sample_rate), f"Generated with speaker {voice_name}."
    except Exception as exc:
        return None, _friendly_error(exc)


def voice_design(text: str, voice_description: str, language: str):
    error = validate_text_input(text)
    if error:
        return None, error
    if not _trimmed(voice_description):
        return None, "Describe the voice you want to generate."
    language_error = validate_language(language)
    if language_error:
        return None, language_error

    try:
        torch, _, _ = _import_runtime()
        model = load_model("design")

        with torch.inference_mode():
            wavs, sample_rate = model.generate_voice_design(
                text=_trimmed(text),
                language=language or "Auto",
                instruct=_trimmed(voice_description),
            )

        return _save_waveform(wavs[0], sample_rate), "Generated with the VoiceDesign model."
    except Exception as exc:
        return None, _friendly_error(exc)


def runtime_summary() -> str:
    try:
        torch, _, _ = _import_runtime()
    except Exception as exc:
        return f"Runtime dependency error: {exc}"

    if not torch.cuda.is_available():
        return "CUDA unavailable. Install a CUDA-enabled PyTorch build and verify the NVIDIA driver."

    name = torch.cuda.get_device_name(0)
    return f"Ready · {name} · models load on first use"


def build_demo():
    import gradio as gr

    with gr.Blocks(
        title="Qwen3-TTS Local",
        analytics_enabled=False,
        fill_width=True,
    ) as demo:
        with gr.Row(equal_height=True):
            gr.HTML(
                """
                <div class="app-shell">
                    <h1 class="app-title">Qwen3-TTS Local</h1>
                    <p class="app-subtitle">Local Gradio interface for voice cloning, preset voices, and voice design on an NVIDIA CUDA GPU.</p>
                </div>
                """
            )
            with gr.Column(scale=0, min_width=150):
                theme_btn = gr.Button("Light / Dark", size="sm")
                unload_btn = gr.Button("Unload model", size="sm")

        runtime = gr.Markdown(value=runtime_summary(), elem_classes=["compact-note"])
        unload_btn.click(fn=unload_model, inputs=None, outputs=runtime, show_progress="hidden")
        theme_btn.click(fn=None, inputs=None, outputs=None, js=THEME_TOGGLE_JS, show_progress="hidden")

        with gr.Tabs():
            with gr.Tab("Voice Cloning"):
                with gr.Row():
                    with gr.Column(scale=6):
                        clone_text = gr.Textbox(label="Text", lines=3, placeholder="Enter text to synthesize")
                        with gr.Row():
                            clone_audio = gr.Audio(label="Reference audio", type="filepath", sources=["upload", "microphone"])
                            clone_language = gr.Dropdown(LANGUAGES, value="Auto", label="Language")
                        clone_transcript = gr.Textbox(
                            label="Reference transcript",
                            lines=2,
                            placeholder="Required for transcript-conditioned cloning",
                        )
                        clone_fast = gr.Checkbox(
                            label="Use x-vector-only mode (transcript optional)",
                            value=True,
                        )
                        clone_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=5):
                        clone_out = gr.Audio(label="Generated audio", buttons=["download"])
                        clone_status = gr.Markdown("Ready.", elem_classes=["status-line", "compact-note"])
                clone_btn.click(
                    voice_clone,
                    inputs=[clone_text, clone_audio, clone_transcript, clone_fast, clone_language],
                    outputs=[clone_out, clone_status],
                    concurrency_id="gpu",
                    concurrency_limit=1,
                )

            with gr.Tab("Custom Voice"):
                with gr.Row():
                    with gr.Column(scale=6):
                        custom_text = gr.Textbox(label="Text", lines=3, placeholder="Enter text to synthesize")
                        with gr.Row():
                            custom_name = gr.Dropdown(SPEAKERS, value="Serena", label="Speaker")
                            custom_language = gr.Dropdown(LANGUAGES, value="Auto", label="Language")
                        custom_inst = gr.Textbox(label="Style instruction", placeholder="Optional: e.g. speak slowly and calmly", lines=2)
                        custom_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=5):
                        custom_out = gr.Audio(label="Generated audio", buttons=["download"])
                        custom_status = gr.Markdown("Ready.", elem_classes=["status-line", "compact-note"])
                custom_btn.click(
                    custom_voice,
                    inputs=[custom_text, custom_name, custom_inst, custom_language],
                    outputs=[custom_out, custom_status],
                    concurrency_id="gpu",
                    concurrency_limit=1,
                )

            with gr.Tab("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=6):
                        design_text = gr.Textbox(label="Text", lines=3, placeholder="Enter text to synthesize")
                        design_desc = gr.Textbox(
                            label="Voice description",
                            placeholder="Describe voice, age, tone, pacing, accent, or delivery",
                            lines=3,
                        )
                        design_language = gr.Dropdown(LANGUAGES, value="Auto", label="Language")
                        design_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=5):
                        design_out = gr.Audio(label="Generated audio", buttons=["download"])
                        design_status = gr.Markdown("Ready.", elem_classes=["status-line", "compact-note"])
                design_btn.click(
                    voice_design,
                    inputs=[design_text, design_desc, design_language],
                    outputs=[design_out, design_status],
                    concurrency_id="gpu",
                    concurrency_limit=1,
                )

        gr.HTML(
            '<div class="footer-note"><a href="https://github.com/abusuraihsakhri/qwen3-tts-local" target="_blank" rel="noopener noreferrer">Repository</a></div>'
        )

    return demo.queue(default_concurrency_limit=1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Qwen3-TTS Gradio interface.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind address (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"TCP port (default: {DEFAULT_PORT})")
    parser.add_argument("--share", action="store_true", default=SHARE_LINK, help="Create a Gradio public share link.")
    parser.add_argument("--inbrowser", action="store_true", help="Open the interface in the default browser.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(runtime_summary())
    demo = build_demo()
    import gradio as gr

    theme = gr.themes.Default(primary_hue="blue", neutral_hue="slate")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        show_error=True,
        theme=theme,
        css=CUSTOM_CSS,
        js=THEME_INIT_JS,
    )


if __name__ == "__main__":
    main()
