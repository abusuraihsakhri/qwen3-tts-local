# Qwen3-TTS Local

Local Gradio interface for the Qwen3-TTS 1.7B models, with voice cloning, the built-in CustomVoice speakers, and free-form voice design.

The application runs inference on a local NVIDIA CUDA GPU. It is not a GitHub Pages application: model inference requires Python, CUDA, and multi-gigabyte model weights.

## Features

- **Voice cloning:** synthesize speech from a reference recording using the Qwen3-TTS Base model.
- **Custom Voice:** use the nine speakers exposed by the Qwen3-TTS CustomVoice model, with optional style instructions.
- **Voice Design:** describe the requested voice characteristics in natural language.
- **Language selection:** Auto plus the ten languages documented by Qwen3-TTS.
- **Local-first UI:** binds to `127.0.0.1` by default, disables Gradio analytics, and does not create a public share link unless requested.
- **GPU memory control:** only one model is retained at a time, inference is serialized, and the loaded model can be unloaded from the interface.

## Requirements

- Python 3.10 or newer.
- NVIDIA GPU and a CUDA-enabled PyTorch build.
- Enough disk space for the selected Qwen3-TTS model weights and Hugging Face cache.
- Internet access for initial package installation and model download. After the required files are cached, inference itself is local.

Qwen3-TTS currently publishes `qwen-tts 0.1.1`. The upstream project documents the 1.7B Base, CustomVoice, and VoiceDesign model IDs used here.

> **Upstream dependency note:** `qwen-tts 0.1.1` pins `transformers==4.57.3`. That Transformers release is covered by CVE-2026-1839 in the `Trainer` checkpoint restore path. This application does not use `Trainer` or restore training checkpoints. Do not substitute untrusted local model/checkpoint directories, and update `qwen-tts` when the upstream package publishes a compatible fixed dependency set.

## Installation

```bash
git clone https://github.com/abusuraihsakhri/qwen3-tts-local.git
cd qwen3-tts-local

python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install a CUDA-enabled PyTorch build that matches your driver/toolkit using the official PyTorch selector, then install the application dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify CUDA before starting the application:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

If audio preprocessing reports a SoX-related error, install the SoX executable for your operating system and ensure it is on `PATH`.

## Run

```bash
python qwen3-tts-local.py --inbrowser
```

The default address is `http://127.0.0.1:7860`.

Useful options:

```text
--host HOST       Bind address (default: 127.0.0.1)
--port PORT       TCP port (default: 7860)
--share           Create a Gradio public share link
--inbrowser       Open the interface in the default browser
```

Model weights are downloaded on first use if they are not already available in the local Hugging Face cache. Switching tabs can therefore trigger a different model download and GPU model load.

## Voice-cloning modes

With **x-vector-only mode** enabled, the reference transcript is optional. With it disabled, the application uses transcript-conditioned cloning and requires the exact transcript of the reference audio.

Use voice cloning only with recordings you have permission to use.

## Privacy and network behavior

Text and uploaded/recorded audio are processed by the local Python process. Gradio analytics are disabled and the server binds to localhost by default. Network access is still required when dependencies or model files need to be downloaded.

`--share` intentionally changes the exposure model by creating a public Gradio share URL. Do not use it for sensitive input unless that exposure is acceptable.

## Development and tests

The unit tests exercise input validation, CUDA checks, model-state recovery, model caching, and command-line defaults without downloading model weights.

```bash
python -m py_compile qwen3-tts-local.py
python -m unittest discover -s tests -v
```

A lightweight UI smoke test requires Gradio but does not require CUDA or Qwen model weights:

```bash
python tests/smoke_ui.py
```

GitHub Actions runs these checks on pushes and pull requests.

## Technology

- Qwen3-TTS / `qwen-tts`
- PyTorch with CUDA
- Gradio
- SoundFile

## Browser compatibility

The interface targets current desktop and mobile versions of Chromium-based browsers, Firefox, and Safari. Actual synthesis runs on the host machine, not in the browser.

## GitHub Pages

GitHub Pages is intentionally not configured. It can only host static client-side assets and cannot provide the CUDA/Python runtime required by this application. Pyodide/PyScript are not practical here because the Qwen3-TTS model stack depends on PyTorch CUDA and large model weights.

## License

This repository currently does not include a `LICENSE` file. Unless a license is added, normal copyright restrictions apply.

## Upstream references

- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- Qwen3-TTS package: https://pypi.org/project/qwen-tts/
- PyTorch installation selector: https://pytorch.org/get-started/locally/
