import gradio as gr
from qwen_tts import Qwen3TTSModel
import torch
import soundfile as sf
import tempfile
import gc
import time
import os

# --- Configuration ---
# Only change this if you want to access it from other devices (set to True)
SHARE_LINK = False 

# Global variables
current_model = None
current_model_type = None

# Enable PyTorch optimizations
# Note: If these cause errors on your specific GPU, comment them out.
try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
except Exception as e:
    print(f"Warning: Could not set TF32 precision: {e}")

if torch.cuda.is_available():
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    print("❌ No NVIDIA GPU detected. This script requires CUDA.")
    exit()

def load_model(model_type):
    """Load model with SDPA optimization"""
    global current_model, current_model_type

    if current_model_type == model_type:
        print(f"✅ Using cached {model_type} model")
        return current_model

    if current_model is not None:
        print(f"Unloading {current_model_type} model...")
        del current_model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Loading {model_type} model (1.7B)...")
    start = time.time()

    try:
        if model_type == "base":
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        elif model_type == "custom":
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        elif model_type == "design":
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

        current_model = Qwen3TTSModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa"
        )

        current_model_type = model_type
        load_time = time.time() - start

        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"✅ Loaded in {load_time:.1f}s | GPU: {allocated:.2f}GB")

        return current_model

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None

def voice_clone(text, reference_audio, ref_transcript, use_fast_mode):
    if not text or not reference_audio:
        return None
    try:
        model = load_model("base")
        if model is None: return None
        
        print("Generating voice clone...")
        if use_fast_mode or not ref_transcript:
            prompt_items = model.create_voice_clone_prompt(ref_audio=reference_audio, x_vector_only_mode=True)
        else:
            prompt_items = model.create_voice_clone_prompt(ref_audio=reference_audio, ref_text=ref_transcript, x_vector_only_mode=False)

        with torch.inference_mode():
            wavs, sr = model.generate_voice_clone(text=text, voice_clone_prompt=prompt_items)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)
        return temp_file.name
    except Exception as e:
        print(f"Error: {e}")
        return None

def custom_voice(text, voice_name, instruction):
    if not text: return None
    try:
        model = load_model("custom")
        if model is None: return None
        
        print(f"Generating custom voice: {voice_name}...")
        with torch.inference_mode():
            if instruction and instruction.strip():
                wavs, sr = model.generate_custom_voice(text=text, speaker=voice_name, instruct=instruction)
            else:
                wavs, sr = model.generate_custom_voice(text=text, speaker=voice_name)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)
        return temp_file.name
    except Exception as e:
        print(f"Error: {e}")
        return None

def voice_design(text, voice_description):
    if not text or not voice_description: return None
    try:
        model = load_model("design")
        if model is None: return None
        
        print("Designing voice...")
        with torch.inference_mode():
            wavs, sr = model.generate_voice_design(text=text, instruct=voice_description)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)
        return temp_file.name
    except Exception as e:
        print(f"Error: {e}")
        return None

# Gradio Interface
with gr.Blocks(title="Qwen3-TTS Local") as demo:
    gr.Markdown("# Qwen3-TTS Local Version")

    with gr.Tab("Voice Cloning"):
        with gr.Row():
            with gr.Column():
                clone_text = gr.Textbox(label="Text", lines=4)
                clone_audio = gr.Audio(label="Reference Audio", type="filepath")
                clone_transcript = gr.Textbox(label="Transcript (Optional)", lines=2)
                clone_fast = gr.Checkbox(label="Fast Mode", value=True)
                clone_btn = gr.Button("Generate")
            with gr.Column():
                clone_out = gr.Audio(label="Result")
        clone_btn.click(voice_clone, inputs=[clone_text, clone_audio, clone_transcript, clone_fast], outputs=clone_out)

    with gr.Tab("Custom Voice"):
        with gr.Row():
            with gr.Column():
                custom_text = gr.Textbox(label="Text", lines=4)
                custom_name = gr.Dropdown(choices=["serena", "vivian", "ono_anna", "sohee", "aiden", "dylan", "eric", "ryan", "uncle_fu"], value="serena", label="Voice")
                custom_inst = gr.Textbox(label="Instruction", placeholder="e.g. speak slowly", lines=1)
                custom_btn = gr.Button("Generate")
            with gr.Column():
                custom_out = gr.Audio(label="Result")
        custom_btn.click(custom_voice, inputs=[custom_text, custom_name, custom_inst], outputs=custom_out)

    with gr.Tab("Voice Design"):
        with gr.Row():
            with gr.Column():
                design_text = gr.Textbox(label="Text", lines=4)
                design_desc = gr.Textbox(label="Description", placeholder="A deep male voice...", lines=2)
                design_btn = gr.Button("Generate")
            with gr.Column():
                design_out = gr.Audio(label="Result")
        design_btn.click(voice_design, inputs=[design_text, design_desc], outputs=design_out)

if __name__ == "__main__":
    print("Starting server... Open the link below in your browser.")
    demo.launch(share=SHARE_LINK)