import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "qwen3-tts-local.py"
spec = importlib.util.spec_from_file_location("qwen3_tts_local_smoke", APP_PATH)
app = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(app)

# build_demo() catches missing runtime dependencies in its status line.
demo = app.build_demo()
assert demo.config.get("components"), "Gradio UI did not produce components"
print(f"UI smoke test passed with {len(demo.config['components'])} components")
