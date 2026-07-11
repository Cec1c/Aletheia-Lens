import importlib
import inspect
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import config
import main


ROOT = Path(__file__).resolve().parents[1]


def _valid_runtime_detection():
    masks = np.zeros((512, 512, 1), dtype=np.bool_)
    masks[160:352, 160:352, 0] = True
    return {
        "rois": np.array([[153, 160, 359, 352]], dtype=np.int32),
        "class_ids": np.array([2], dtype=np.int32),
        "scores": np.array([0.99], dtype=np.float32),
        "masks": masks,
    }


def _nonconstant_output(shape, dtype):
    values = np.linspace(0, 1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    if np.issubdtype(np.dtype(dtype), np.integer):
        return (values * 255).round().astype(dtype)
    return values.astype(dtype)


class _FakeSession:
    def __init__(self, active_providers):
        self._active_providers = active_providers

    def get_providers(self):
        return list(self._active_providers)


class _FakeOrt:
    def __init__(self, available_providers, active_providers, preload_error=None):
        self.available_providers = available_providers
        self.active_providers = active_providers
        self.preload_error = preload_error
        self.preload_calls = []
        self.session_calls = []

    def get_available_providers(self):
        return list(self.available_providers)

    def preload_dlls(self, **kwargs):
        self.preload_calls.append(kwargs)
        if self.preload_error:
            raise RuntimeError(self.preload_error)

    def InferenceSession(self, model_path, sess_options=None, providers=None):
        self.session_calls.append(
            {
                "model_path": model_path,
                "sess_options": sess_options,
                "providers": providers,
            }
        )
        return _FakeSession(self.active_providers)


class RuntimeConfigurationTests(unittest.TestCase):
    def test_runtime_model_paths_use_onnx_files(self):
        self.assertEqual(Path(config.mrcnn_model).name, "weights.onnx")
        self.assertEqual(Path(config.mrcnn_model).parent.name, "mrcnn")
        self.assertEqual(Path(config.esrgan_model).name, "4x-Fatal-Pixels.onnx")
        self.assertEqual(Path(config.esrgan_model).parent.name, "esrgan")

    def test_resource_path_uses_pyinstaller_runtime_root(self):
        packaged_root = ROOT / "packaged-runtime"
        had_meipass = hasattr(sys, "_MEIPASS")
        previous_meipass = getattr(sys, "_MEIPASS", None)
        setattr(sys, "_MEIPASS", str(packaged_root))
        reloaded = importlib.reload(config)
        try:
            self.assertEqual(
                Path(reloaded.resource_path("models/example.onnx")),
                packaged_root / "models/example.onnx",
            )
        finally:
            if had_meipass:
                setattr(sys, "_MEIPASS", previous_meipass)
            else:
                delattr(sys, "_MEIPASS")
            importlib.reload(config)

    def test_session_manager_reports_cuda_fallback_to_cpu(self):
        runtime = importlib.import_module("onnx_runtime")
        fake_ort = _FakeOrt(
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            active_providers=["CPUExecutionProvider"],
        )
        manager = runtime.OnnxRuntimeManager(fake_ort, runtime_root=ROOT)

        manager.create_session("model.onnx")
        status = manager.status()

        self.assertEqual(
            fake_ort.session_calls[0]["providers"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.assertFalse(status["cuda_active"])
        self.assertEqual(status["device"], "CPU")
        self.assertIn("CUDA 12", status["guidance"])
        self.assertIn("cuDNN 9", status["guidance"])

    def test_session_manager_skips_cuda_when_runtime_preload_fails(self):
        runtime = importlib.import_module("onnx_runtime")
        fake_ort = _FakeOrt(
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            active_providers=["CPUExecutionProvider"],
            preload_error="missing CUDA runtime DLLs",
        )
        manager = runtime.OnnxRuntimeManager(fake_ort, runtime_root=ROOT)

        manager.create_session("model.onnx")
        status = manager.status()

        self.assertEqual(fake_ort.session_calls[0]["providers"], ["CPUExecutionProvider"])
        self.assertIn("missing CUDA runtime DLLs", status["guidance"])

    def test_session_manager_honors_an_explicit_cpu_only_policy(self):
        runtime = importlib.import_module("onnx_runtime")
        fake_ort = _FakeOrt(
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            active_providers=["CPUExecutionProvider"],
        )
        manager = runtime.OnnxRuntimeManager(ort_module=fake_ort)

        self.assertIn("providers", inspect.signature(manager.create_session).parameters)
        manager.create_session("legacy.onnx", providers=["CPUExecutionProvider"])
        status = manager.status()

        self.assertEqual(fake_ort.session_calls[0]["providers"], ["CPUExecutionProvider"])
        self.assertEqual(status["intentional_cpu_models"], ["legacy.onnx"])
        self.assertEqual(fake_ort.preload_calls, [])

    def test_cpu_only_runtime_does_not_report_expected_cpu_execution_as_fallback(self):
        runtime = importlib.import_module("onnx_runtime")
        fake_ort = _FakeOrt(
            available_providers=["CPUExecutionProvider"],
            active_providers=["CPUExecutionProvider"],
        )
        manager = runtime.OnnxRuntimeManager(ort_module=fake_ort)

        manager.create_session("model.onnx")
        status = manager.status()

        self.assertEqual(status["unexpected_cpu_models"], [])

    def test_source_runtime_registers_nvidia_wheel_dll_directories_before_preload(self):
        runtime = importlib.import_module("onnx_runtime")
        fake_ort = _FakeOrt(
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            active_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        with tempfile.TemporaryDirectory() as site_root:
            site_root = Path(site_root)
            cudnn_bin = site_root / "nvidia" / "cudnn" / "bin"
            cublas_bin = site_root / "nvidia" / "cublas" / "bin"
            cudnn_bin.mkdir(parents=True)
            cublas_bin.mkdir(parents=True)
            expected_cudnn_bin = cudnn_bin.resolve()
            expected_cublas_bin = cublas_bin.resolve()

            events = []
            handles = []

            def add_dll_directory(path):
                events.append(("add", Path(path)))
                handle = object()
                handles.append(handle)
                return handle

            def preload_dlls(**kwargs):
                events.append(("preload", kwargs))

            fake_ort.preload_dlls = preload_dlls
            with (
                mock.patch("site.getsitepackages", return_value=[str(site_root)]),
                mock.patch("site.getusersitepackages", return_value=""),
                mock.patch("os.add_dll_directory", create=True, side_effect=add_dll_directory),
            ):
                manager = runtime.OnnxRuntimeManager(fake_ort, runtime_root=ROOT)
                manager.create_session("model.onnx")

        self.assertEqual(
            events,
            [
                ("add", expected_cublas_bin),
                ("add", expected_cudnn_bin),
                ("preload", {}),
            ],
        )
        self.assertEqual(manager._dll_directory_handles, handles)

    def test_packaged_runtime_registers_bundle_root_for_lazy_cudnn_dlls(self):
        runtime = importlib.import_module("onnx_runtime")
        fake_ort = _FakeOrt(
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            active_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        with tempfile.TemporaryDirectory() as runtime_root:
            runtime_root = Path(runtime_root)
            handle = object()
            with (
                mock.patch.object(sys, "frozen", True, create=True),
                mock.patch("site.getsitepackages", return_value=[]),
                mock.patch("site.getusersitepackages", return_value=""),
                mock.patch("os.add_dll_directory", create=True, return_value=handle) as add_directory,
            ):
                manager = runtime.OnnxRuntimeManager(fake_ort, runtime_root=runtime_root)
                manager.create_session("model.onnx")

        add_directory.assert_called_once_with(str(runtime_root))
        self.assertEqual(fake_ort.preload_calls, [{"directory": str(runtime_root)}])
        self.assertEqual(manager._dll_directory_handles, [handle])

    def test_runtime_status_refreshes_providers_after_execution_fallback(self):
        runtime = importlib.import_module("onnx_runtime")
        fake_ort = _FakeOrt(
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            active_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        manager = runtime.OnnxRuntimeManager(fake_ort, runtime_root=ROOT)

        session = manager.create_session("model.onnx")
        session._active_providers = ["CPUExecutionProvider"]
        status = manager.status()

        self.assertFalse(status["cuda_active"])
        self.assertEqual(status["device"], "CPU")
        self.assertEqual(status["active_providers"], ["CPUExecutionProvider"])

    def test_runtime_status_reports_mixed_model_providers(self):
        runtime = importlib.import_module("onnx_runtime")
        fake_ort = _FakeOrt(
            available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            active_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        manager = runtime.OnnxRuntimeManager(fake_ort, runtime_root=ROOT)

        manager.create_session("cuda.onnx")
        cpu_session = manager.create_session("cpu.onnx")
        cpu_session._active_providers = ["CPUExecutionProvider"]
        status = manager.status()

        self.assertEqual(status["device"], "MIXED")
        self.assertTrue(status["cuda_active"])
        self.assertEqual(
            status["session_providers"],
            {
                "cuda.onnx": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "cpu.onnx": ["CPUExecutionProvider"],
            },
        )

    def test_app_logs_runtime_provider_changes(self):
        self.assertTrue(hasattr(main.DeepCreampyApp, "log_runtime_status"))
        app = main.DeepCreampyApp.__new__(main.DeepCreampyApp)
        app.runtime_status_signature = None
        app.log_message = mock.Mock()
        mixed_status = {
            "device": "MIXED",
            "guidance": "部分模型已回退 CPU",
            "session_providers": {
                "cuda.onnx": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "cpu.onnx": ["CPUExecutionProvider"],
            },
        }
        cpu_status = {
            "device": "CPU",
            "guidance": "CUDA provider 未激活",
            "session_providers": {
                "cuda.onnx": ["CPUExecutionProvider"],
                "cpu.onnx": ["CPUExecutionProvider"],
            },
        }

        with mock.patch(
            "onnx_runtime.get_runtime_status",
            side_effect=[mixed_status, mixed_status, cpu_status],
        ):
            app.log_runtime_status(force=True)
            app.log_runtime_status()
            app.log_runtime_status()

        messages = [call.args[0] for call in app.log_message.call_args_list]
        self.assertIn("部分模型使用 CUDA", messages[0])
        self.assertIn("cpu.onnx", messages[0])
        self.assertIn("部分模型已回退 CPU", messages[1])
        self.assertIn("当前使用 CPU", messages[2])
        self.assertIn("CUDA provider 未激活", messages[3])

    def test_app_distinguishes_intentional_cpu_models_from_fallbacks(self):
        app = main.DeepCreampyApp.__new__(main.DeepCreampyApp)
        app.runtime_status_signature = None
        app.log_message = mock.Mock()
        status = {
            "device": "MIXED",
            "guidance": "DeepCreamPy 遗留模型固定使用 CPU",
            "session_providers": {
                "weights.onnx": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "bar.onnx": ["CPUExecutionProvider"],
            },
            "intentional_cpu_models": ["bar.onnx"],
            "unexpected_cpu_models": [],
        }

        with mock.patch("onnx_runtime.get_runtime_status", return_value=status):
            app.log_runtime_status(force=True)

        first_message = app.log_message.call_args_list[0].args[0]
        self.assertIn("兼容性 CPU", first_message)
        self.assertNotIn("CPU 回退", first_message)

    def test_runtime_smoke_cli_runs_all_models_without_starting_gui(self):
        self.assertTrue(hasattr(main, "run_runtime_smoke_test"))
        session = _FakeSession(["CUDAExecutionProvider", "CPUExecutionProvider"])
        fake_detector = types.SimpleNamespace(
            _onnx_session=session,
            detect_image=mock.Mock(return_value=_valid_runtime_detection()),
        )
        fake_esrgan = types.SimpleNamespace(
            _esrgan_session=session,
            _run_esrgan_onnx=mock.Mock(
                return_value=_nonconstant_output((64, 64, 3), np.uint8)
            ),
        )
        fake_predict = types.SimpleNamespace(
            session_bar=session,
            session_mosaic=session,
            predict=mock.Mock(
                return_value=_nonconstant_output((256, 256, 3), np.float32)
            ),
        )
        runtime_status = {
            "device": "MIXED",
            "cuda_available": True,
            "session_providers": {
                str(config.mrcnn_model): session.get_providers(),
                str(config.esrgan_model): session.get_providers(),
                str(config.deepcreampy_bar_model): ["CPUExecutionProvider"],
                str(config.deepcreampy_mosaic_model): ["CPUExecutionProvider"],
            },
        }

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "detector": fake_detector,
                    "esrgan": fake_esrgan,
                    "predict": fake_predict,
                },
            ),
            mock.patch("onnx_runtime.get_runtime_status", return_value=runtime_status),
        ):
            try:
                result = main.run_runtime_smoke_test(
                    require_cuda=True,
                    require_cuda_provider=True,
                )
            except RuntimeError as exc:
                self.fail(f"Intentional CPU-only models must not fail CUDA smoke checks: {exc}")

        self.assertEqual(result, runtime_status)
        fake_detector.detect_image.assert_called_once()
        fake_esrgan._run_esrgan_onnx.assert_called_once()
        self.assertEqual(fake_predict.predict.call_count, 2)
        detector_input = fake_detector.detect_image.call_args.args[0]
        self.assertEqual(detector_input.shape, (512, 512, 3))
        self.assertGreater(detector_input[160:352, 160:352].std(), 0)

    def test_deepcreampy_legacy_models_are_explicitly_cpu_pinned(self):
        predict_source = (ROOT / "predict.py").read_text(encoding="utf-8")

        self.assertEqual(
            predict_source.count('providers=["CPUExecutionProvider"]'),
            2,
        )

    def test_runtime_smoke_rejects_invalid_model_outputs(self):
        session = _FakeSession(["CPUExecutionProvider"])
        fake_detector = types.SimpleNamespace(
            _onnx_session=session,
            detect_image=mock.Mock(return_value=_valid_runtime_detection()),
        )
        fake_esrgan = types.SimpleNamespace(
            _esrgan_session=session,
            _run_esrgan_onnx=mock.Mock(
                return_value=np.zeros((1, 1, 3), dtype=np.uint8)
            ),
        )
        fake_predict = types.SimpleNamespace(
            session_bar=session,
            session_mosaic=session,
            predict=mock.Mock(
                return_value=np.zeros((256, 256, 3), dtype=np.float32)
            ),
        )
        runtime_status = {
            "device": "CPU",
            "cuda_available": False,
            "session_providers": {},
        }

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "detector": fake_detector,
                    "esrgan": fake_esrgan,
                    "predict": fake_predict,
                },
            ),
            mock.patch("onnx_runtime.get_runtime_status", return_value=runtime_status),
        ):
            with self.assertRaisesRegex(RuntimeError, "ESRGAN"):
                main.run_runtime_smoke_test()

        with (
            mock.patch.object(main, "run_runtime_smoke_test", return_value=runtime_status) as smoke,
            mock.patch.object(main, "DeepCreampyApp") as app_class,
        ):
            exit_code = main.main(
                [
                    "--runtime-smoke-test",
                    "--require-cuda-provider",
                    "--require-cuda",
                ]
            )

        self.assertEqual(exit_code, 0)
        smoke.assert_called_once_with(
            require_cuda=True,
            require_cuda_provider=True,
        )
        app_class.assert_not_called()

    def test_runtime_smoke_rejects_semantically_degenerate_outputs(self):
        with self.assertRaisesRegex(RuntimeError, "ESRGAN.*degenerate"):
            main._validate_runtime_smoke_outputs(
                _valid_runtime_detection(),
                np.zeros((64, 64, 3), dtype=np.uint8),
                _nonconstant_output((256, 256, 3), np.float32),
                _nonconstant_output((256, 256, 3), np.float32),
            )

    def test_runtime_smoke_rejects_a_misaligned_detection_mask(self):
        detection = _valid_runtime_detection()
        detection["masks"][:] = False
        detection["masks"][0:32, 0:32, 0] = True

        with self.assertRaisesRegex(RuntimeError, "mask IoU"):
            main._validate_runtime_smoke_outputs(
                detection,
                _nonconstant_output((64, 64, 3), np.uint8),
                _nonconstant_output((256, 256, 3), np.float32),
                _nonconstant_output((256, 256, 3), np.float32),
            )

    def test_runtime_smoke_cli_writes_failure_report_for_windowed_builds(self):
        with tempfile.TemporaryDirectory() as root:
            report_path = Path(root) / "smoke.txt"
            with (
                mock.patch.object(
                    main,
                    "run_runtime_smoke_test",
                    side_effect=RuntimeError("packaged failure"),
                ),
                mock.patch.dict(
                    "os.environ",
                    {"ALETHEIA_SMOKE_REPORT": str(report_path)},
                    clear=False,
                ),
            ):
                exit_code = main.main(["--runtime-smoke-test"])

            self.assertEqual(exit_code, 1)
            self.assertTrue(report_path.exists())
            self.assertIn("packaged failure", report_path.read_text(encoding="utf-8"))

    def test_cpu_and_gpu_dependencies_are_split(self):
        cpu_requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        gpu_requirements = (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")

        self.assertIn("onnxruntime==1.23.2", cpu_requirements)
        self.assertNotIn("onnxruntime-gpu", cpu_requirements.replace("_", "-"))
        self.assertIn(
            "onnxruntime-gpu[cuda,cudnn]==1.23.2",
            gpu_requirements.replace("_", "-"),
        )
        for dependency in (
            "nvidia-cuda-nvrtc-cu12==12.9.86",
            "nvidia-cuda-runtime-cu12==12.9.79",
            "nvidia-cublas-cu12==12.9.2.10",
            "nvidia-cufft-cu12==11.4.1.4",
            "nvidia-curand-cu12==10.3.10.19",
            "nvidia-cudnn-cu12==9.24.0.43",
        ):
            self.assertIn(dependency, gpu_requirements)

    def test_pyinstaller_collects_external_onnx_data(self):
        spec = (ROOT / "main.spec").read_text(encoding="utf-8")
        self.assertIn("*.onnx.data", spec)
        self.assertNotIn("pakversion.txt", spec)

    def test_pyinstaller_collects_numpy_runtime_extensions(self):
        spec = (ROOT / "main.spec").read_text(encoding="utf-8")

        self.assertIn("numpy.core._multiarray_tests", spec)

    def test_readme_documents_current_runtime_models_and_gpu_install(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("requirements-gpu.txt", readme)
        self.assertIn("weights.onnx", readme)
        self.assertIn("4x-Fatal-Pixels.onnx.data", readme)
        self.assertIn("源码运行 RAR", readme)
        self.assertIn("会话已启用 CUDA", readme)
        self.assertIn("压缩包模式", readme)
        self.assertNotIn("weights.h5", readme)
        self.assertNotIn("4x-Fatal-Pixels.pth", readme)


if __name__ == "__main__":
    unittest.main()
