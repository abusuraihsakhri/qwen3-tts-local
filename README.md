# Qwen3-TTS Local

**A lightweight, local Python wrapper for Qwen3-TTS (1.7B) using Gradio.**

Run high-quality voice cloning and text-to-speech generation directly on your own NVIDIA GPU without relying on cloud APIs or paying for subscriptions.

## 🚀 Features

* **Local Privacy:** Everything runs offline on your machine. No audio is uploaded to the cloud.
* **Voice Cloning:** Clone any voice using just a 3-second audio reference file.
* **Custom Presets:** Access 9 pre-defined high-quality character voices (e.g., Serena, Aiden, Ono Anna).
* **Voice Design:** Create unique voices by describing them in plain text (e.g., *"A deep, authoritative male voice speaking slowly"*).
* **Optimized:** Includes TF32 precision settings for faster inference on NVIDIA RTX 30-series cards and newer.

---

## 🛠️ Prerequisites

Before installing, ensure you have the following:

### 1. Hardware
* **NVIDIA GPU:** This script requires a GPU with CUDA support.
* **VRAM:** Minimum **6GB VRAM** recommended.

### 2. Python
* **Python 3.10** or newer installed.

### 3. SoX (Sound eXchange) - *Crucial for Windows Users*
This script relies on SoX to process audio. If you do not install this, audio generation will fail.

1.  Download SoX v14.4.2 from SourceForge:
    * [**Download SoX (SourceForge)**](https://sourceforge.net/projects/sox/files/sox/14.4.2/)
    * *Note: Download the file ending in `win32.exe`.*
2.  Run the installer. **Copy the destination path** during installation (usually `C:\Program Files (x86)\sox-14-4-2`).
3.  **Add to System PATH:**
    * Open Windows Search → Type "env" → Select "Edit the system environment variables".
    * Click **Environment Variables**.
    * Under "System variables", find **Path** and click **Edit**.
    * Click **New** and paste the path you copied (e.g., `C:\Program Files (x86)\sox-14-4-2`).
    * Click OK on all windows.

---

## 📦 Installation

### Step 1: Clone or Download this Repository
If you have Git installed:
```bash
git clone [https://github.com/abusuraihsakhri/qwen3-tts-local.git](https://github.com/abusuraihsakhri/qwen3-tts-local.git)
cd qwen3-tts-local
