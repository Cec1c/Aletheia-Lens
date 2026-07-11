"""Shared ONNX Runtime session setup and provider diagnostics."""

from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
import io
import os
import site
import sys
import threading

import onnxruntime as ort


CUDA_GUIDANCE = (
    "CUDA provider 未激活。源码运行请安装 requirements-gpu.txt；"
    "该环境需要 CUDA 12.x、cuDNN 9.x 与最新 MSVC 运行库。"
)


def _default_runtime_root():
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


class OnnxRuntimeManager:
    """Create sessions consistently and record the providers actually in use."""

    def __init__(self, ort_module=ort, runtime_root=None):
        self._ort = ort_module
        self._runtime_root = Path(runtime_root or _default_runtime_root())
        self._prepared = False
        self._cuda_ready = None
        self._preload_error = None
        self._dll_directory_errors = []
        self._dll_directory_handles = []
        self._sessions = {}
        self._intentional_cpu_models = set()
        self._lock = threading.Lock()

    def _available_providers(self):
        return list(self._ort.get_available_providers())

    def _candidate_dll_directories(self):
        if not callable(getattr(os, "add_dll_directory", None)):
            return []

        if getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"):
            return [self._runtime_root] if self._runtime_root.is_dir() else []

        roots = []
        try:
            roots.extend(site.getsitepackages())
        except (AttributeError, OSError):
            pass
        try:
            user_site = site.getusersitepackages()
            if user_site:
                roots.append(user_site)
        except (AttributeError, OSError):
            pass

        directories = []
        seen = set()
        for root in roots:
            nvidia_root = Path(root) / "nvidia"
            if not nvidia_root.is_dir():
                continue
            for directory in sorted(nvidia_root.glob("*/bin")):
                if not directory.is_dir():
                    continue
                resolved = directory.resolve()
                key = str(resolved).casefold()
                if key not in seen:
                    seen.add(key)
                    directories.append(resolved)
        return directories

    def _register_dll_directories(self):
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if not callable(add_dll_directory):
            return

        for directory in self._candidate_dll_directories():
            try:
                # Keep each handle alive for the process lifetime. cuDNN 9 loads
                # engine sub-libraries lazily during the first convolution.
                self._dll_directory_handles.append(add_dll_directory(str(directory)))
            except OSError as exc:
                self._dll_directory_errors.append(f"{directory}: {exc}")

    def _prepare_cuda(self):
        if self._prepared:
            return

        self._prepared = True
        available = self._available_providers()
        preload = getattr(self._ort, "preload_dlls", None)
        if "CUDAExecutionProvider" not in available:
            self._cuda_ready = False
            return
        if preload is None:
            self._cuda_ready = True
            return

        self._register_dll_directories()

        kwargs = {}
        if getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"):
            kwargs["directory"] = str(self._runtime_root)

        try:
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                preload(**kwargs)
            preload_output = output.getvalue()
            failure_lines = [
                line.strip()
                for line in preload_output.splitlines()
                if "Failed to load" in line or "not installed" in line
            ]
            if failure_lines:
                self._cuda_ready = False
                self._preload_error = failure_lines[0]
            else:
                self._cuda_ready = True
        except Exception as exc:  # Session creation still provides CPU fallback.
            self._cuda_ready = False
            self._preload_error = str(exc)

    def preferred_providers(self):
        available = self._available_providers()
        providers = []
        if "CUDAExecutionProvider" in available and self._cuda_ready is not False:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def create_session(self, model_path, session_options=None, providers=None):
        requested_providers = None if providers is None else list(providers)
        intentional_cpu_only = requested_providers is not None and (
            "CUDAExecutionProvider" not in requested_providers
        )
        with self._lock:
            if requested_providers is None or "CUDAExecutionProvider" in requested_providers:
                self._prepare_cuda()
            selected_providers = requested_providers or self.preferred_providers()

        if session_options is None and hasattr(self._ort, "SessionOptions"):
            session_options = self._ort.SessionOptions()
            session_options.log_severity_level = 3

        session = self._ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=selected_providers,
        )
        active = list(session.get_providers())
        with self._lock:
            model_key = str(model_path)
            self._sessions[model_key] = session
            if intentional_cpu_only:
                self._intentional_cpu_models.add(model_key)
            else:
                self._intentional_cpu_models.discard(model_key)

        device = "CUDA" if "CUDAExecutionProvider" in active else "CPU"
        print(f"[ONNX Runtime] {Path(model_path).name}: {device} ({', '.join(active)})")
        return session

    def status(self):
        available = self._available_providers()
        with self._lock:
            sessions = dict(self._sessions)
            intentional_cpu_models = sorted(
                model_path
                for model_path in self._intentional_cpu_models
                if model_path in sessions
            )
        session_providers = {
            model_path: list(session.get_providers())
            for model_path, session in sessions.items()
        }
        active = sorted({provider for providers in session_providers.values() for provider in providers})
        cuda_available = "CUDAExecutionProvider" in available
        cuda_session_count = sum(
            "CUDAExecutionProvider" in providers
            for providers in session_providers.values()
        )
        cuda_active = cuda_session_count > 0
        if session_providers and cuda_session_count == len(session_providers):
            device = "CUDA"
        elif cuda_active:
            device = "MIXED"
        else:
            device = "CPU"

        intentional_cpu_set = set(intentional_cpu_models)
        unexpected_cpu_models = (
            sorted(
                model_path
                for model_path, providers in session_providers.items()
                if model_path not in intentional_cpu_set
                and "CUDAExecutionProvider" not in providers
            )
            if cuda_available
            else []
        )

        guidance = ""
        if cuda_available and unexpected_cpu_models and device == "CPU":
            guidance = CUDA_GUIDANCE
            if self._preload_error:
                guidance = f"{guidance} 预加载错误: {self._preload_error}"
            elif self._dll_directory_errors:
                guidance = f"{guidance} DLL 搜索路径错误: {self._dll_directory_errors[0]}"
        elif device == "MIXED" and unexpected_cpu_models:
            guidance = "部分 ONNX 模型已回退 CPU；程序会继续运行，但 GPU 加速不完整。"
        elif intentional_cpu_models:
            cpu_names = ", ".join(Path(path).name for path in intentional_cpu_models)
            guidance = (
                "DeepCreamPy 遗留模型为避免 CUDA 数值异常而固定使用 CPU: "
                f"{cpu_names}"
            )

        return {
            "device": device,
            "cuda_available": cuda_available,
            "cuda_active": cuda_active,
            "available_providers": available,
            "active_providers": active,
            "session_providers": session_providers,
            "intentional_cpu_models": intentional_cpu_models,
            "unexpected_cpu_models": unexpected_cpu_models,
            "guidance": guidance,
        }


_manager = OnnxRuntimeManager()


def create_session(model_path, session_options=None, providers=None):
    return _manager.create_session(model_path, session_options, providers)


def get_runtime_status():
    return _manager.status()
