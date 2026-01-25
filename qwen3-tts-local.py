import gradio as gr
from qwen_tts import Qwen3TTSModel
import torch
import soundfile as sf
import tempfile
import gc
import time

SHARE_LINK = False

custom_css = """
.header-container {
    background: #2b2d31;
    padding: 35px;
    border-radius: 16px;
    margin-bottom: 30px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.header-title {
    font-size: 2.5em;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.5px;
    color: #ffffff;
}
.header-subtitle {
    font-size: 1.0em;
    opacity: 0.85;
    margin-top: 10px;
    font-weight: 300;
    color: #b8bcc8;
}
.header-author {
    font-size: 0.75em;
    opacity: 0.6;
    margin-top: 5px;
    font-weight: 300;
    color: #8a8e9a;
}
.footer {
    text-align: center;
    margin-top: 30px;
    padding-top: 15px;
    border-top: 1px solid #e5e7eb;
    font-size: 0.9em;
    color: #6b7280;
}
.footer a {
    color: #667eea;
    text-decoration: none;
    font-weight: 500;
}
.footer a:hover {
    text-decoration: underline;
}
"""

current_model = None
current_model_type = None

try:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
except Exception as e:
    print(f"Warning: {e}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No CUDA GPU detected")
    exit()

def load_model(model_type):
    global current_model, current_model_type

    if current_model_type == model_type:
        return current_model

    if current_model is not None:
        del current_model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Loading {model_type} model...")
    start = time.time()

    try:
        models = {
            "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        }

        current_model = Qwen3TTSModel.from_pretrained(
            models[model_type],
            torch_dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa"
        )

        current_model_type = model_type
        print(f"Loaded in {time.time() - start:.1f}s")
        return current_model

    except Exception as e:
        print(f"Error: {e}")
        return None

def voice_clone(text, reference_audio, ref_transcript, use_fast_mode):
    if not text or not reference_audio:
        return None
    try:
        model = load_model("base")
        if model is None:
            return None
        
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
    if not text:
        return None
    try:
        model = load_model("custom")
        if model is None:
            return None
        
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
    if not text or not voice_description:
        return None
    try:
        model = load_model("design")
        if model is None:
            return None
        
        with torch.inference_mode():
            wavs, sr = model.generate_voice_design(text=text, instruct=voice_description)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)
        return temp_file.name
    except Exception as e:
        print(f"Error: {e}")
        return None

with gr.Blocks(title="Qwen3-TTS", css=custom_css) as demo:
    
    gr.HTML("""
        <div class="header-container">
            <h1 class="header-title">Qwen3-TTS</h1>
            <div class="header-subtitle">A local text-to-speech engine based on Qwen 3</div>
            <div class="header-author">by Dr. Abu Suraih Sakhri</div>
        </div>
    """)

    with gr.Tab("Voice Cloning"):
        with gr.Row():
            with gr.Column():
                clone_text = gr.Textbox(label="Text", lines=4, placeholder="Enter text to synthesize")
                clone_audio = gr.Audio(label="Reference Audio", type="filepath")
                clone_transcript = gr.Textbox(label="Transcript (Optional)", lines=2, placeholder="Transcript of reference audio")
                clone_fast = gr.Checkbox(label="Fast Mode", value=True)
                clone_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                clone_out = gr.Audio(label="Output")
        clone_btn.click(voice_clone, inputs=[clone_text, clone_audio, clone_transcript, clone_fast], outputs=clone_out)

    with gr.Tab("Custom Voice"):
        with gr.Row():
            with gr.Column():
                custom_text = gr.Textbox(label="Text", lines=4, placeholder="Enter text to synthesize")
                custom_name = gr.Dropdown(
                    choices=["serena", "vivian", "ono_anna", "sohee", "aiden", "dylan", "eric", "ryan", "uncle_fu"], 
                    value="serena", 
                    label="Speaker"
                )
                custom_inst = gr.Textbox(label="Style (Optional)", placeholder="e.g. speak slowly, whisper", lines=1)
                custom_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                custom_out = gr.Audio(label="Output")
        custom_btn.click(custom_voice, inputs=[custom_text, custom_name, custom_inst], outputs=custom_out)

    with gr.Tab("Voice Design"):
        with gr.Row():
            with gr.Column():
                design_text = gr.Textbox(label="Text", lines=4, placeholder="Enter text to synthesize")
                design_desc = gr.Textbox(label="Voice Description", placeholder="Describe the desired voice characteristics", lines=2)
                design_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                design_out = gr.Audio(label="Output")
        design_btn.click(voice_design, inputs=[design_text, design_desc], outputs=design_out)
        
    gr.HTML("""
        <div class="footer">
            <a href="https://github.com/abusuraihsakhri/qwen3-tts-local" target="_blank">github.com/abusuraihsakhri/qwen3-tts-local</a>
        </div>
    """)

if __name__ == "__main__":
    demo.launch(share=SHARE_LINK)