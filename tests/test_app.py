import importlib.util
import pathlib
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "qwen3-tts-local.py"

spec = importlib.util.spec_from_file_location("qwen3_tts_local", APP_PATH)
app = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(app)


class FakeCuda:
    def __init__(self, available=True, bf16=True):
        self.available = available
        self.bf16 = bf16
        self.empty_cache_calls = 0

    def is_available(self):
        return self.available

    def is_bf16_supported(self):
        return self.bf16

    def empty_cache(self):
        self.empty_cache_calls += 1

    def get_device_name(self, index):
        return "Fake GPU"


class FakePrecision:
    fp32_precision = None
    allow_tf32 = False


class FakeBackends:
    class Cudnn:
        benchmark = False
        conv = FakePrecision()
        allow_tf32 = False

    class Cuda:
        matmul = FakePrecision()

    cudnn = Cudnn()
    cuda = Cuda()


class FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"

    def __init__(self, available=True, bf16=True):
        self.cuda = FakeCuda(available, bf16)
        self.backends = FakeBackends()


class AppTests(unittest.TestCase):
    def setUp(self):
        app.current_model = None
        app.current_model_type = None

    def test_clone_validation(self):
        self.assertEqual(app.validate_clone_inputs("", None, "", True), "Enter text to synthesize.")
        self.assertEqual(app.validate_clone_inputs("hello", None, "", True), "Add a reference audio file.")
        self.assertIsNotNone(app.validate_clone_inputs("hello", "ref.wav", "", False))
        self.assertIsNone(app.validate_clone_inputs("hello", "ref.wav", "", True))

    def test_language_validation(self):
        self.assertIsNone(app.validate_language("Auto"))
        self.assertIsNone(app.validate_language("German"))
        self.assertEqual(app.validate_language("Klingon"), "Select a supported language.")

    def test_preferred_dtype(self):
        self.assertEqual(app._preferred_dtype(FakeTorch(bf16=True)), "bf16")
        self.assertEqual(app._preferred_dtype(FakeTorch(bf16=False)), "fp16")

    def test_configure_cuda_rejects_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA-capable"):
            app.configure_cuda(FakeTorch(available=False))

    def test_failed_model_load_resets_global_state(self):
        fake_torch = FakeTorch()

        class BrokenModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError("load failed")

        app.current_model = object()
        app.current_model_type = "custom"
        with mock.patch.object(app, "_import_runtime", return_value=(fake_torch, object(), BrokenModel)):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                app.load_model("base")

        self.assertIsNone(app.current_model)
        self.assertIsNone(app.current_model_type)

    def test_successful_model_load_uses_current_api_and_cache(self):
        fake_torch = FakeTorch()
        sentinel = object()

        class GoodModel:
            calls = []

            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                cls.calls.append((model_id, kwargs))
                return sentinel

        with mock.patch.object(app, "_import_runtime", return_value=(fake_torch, object(), GoodModel)) as runtime:
            first = app.load_model("base")
            second = app.load_model("base")

        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        self.assertEqual(GoodModel.calls[0][0], app.MODEL_IDS["base"])
        self.assertEqual(GoodModel.calls[0][1]["dtype"], "bf16")
        self.assertNotIn("torch_dtype", GoodModel.calls[0][1])
        self.assertEqual(runtime.call_count, 1)

    def test_unknown_model_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model type"):
            app.load_model("not-a-model")

    def test_parse_args_defaults_are_local(self):
        args = app.parse_args([])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 7860)
        self.assertFalse(args.share)


if __name__ == "__main__":
    unittest.main()
