from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "build-and-release.yml"
MODEL_MANIFEST = ROOT / ".github" / "model-assets.sha256"
MODEL_PUBLISH_SCRIPT = ROOT / "tools" / "publish_model_release.ps1"


class ReleaseWorkflowTests(unittest.TestCase):
    def test_workflow_builds_cpu_and_cuda12_variants(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("matrix:", workflow)
        self.assertIn("cpu", workflow)
        self.assertIn("cuda12", workflow)
        self.assertIn("requirements-gpu.txt", workflow)
        self.assertIn("requirements.txt", workflow)

    def test_workflow_forces_utf8_for_chinese_test_output(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn('PYTHONUTF8: "1"', workflow)
        self.assertIn('PYTHONIOENCODING: "utf-8"', workflow)

    def test_workflow_downloads_only_runtime_onnx_assets(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("weights.onnx", workflow)
        self.assertIn("4x-Fatal-Pixels.onnx", workflow)
        self.assertIn("4x-Fatal-Pixels.onnx.data", workflow)
        self.assertNotIn("weights.h5", workflow)
        self.assertNotIn("4x-Fatal-Pixels.pth", workflow)

    def test_release_is_created_once_after_matrix_builds(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("release:", workflow)
        self.assertIn("needs: build", workflow)
        self.assertRegex(workflow, r"actions/download-artifact@[0-9a-f]{40}")
        self.assertEqual(
            len(re.findall(r"softprops/action-gh-release@[0-9a-f]{40}", workflow)),
            1,
        )

    def test_workflow_uses_least_privilege_and_pins_actions(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertIn("persist-credentials: false", workflow)
        release_section = workflow.split("\n  release:", maxsplit=1)[1]
        self.assertIn("permissions:\n      contents: write", release_section)

        action_refs = re.findall(r"^\s*uses:\s*(\S+)\s*$", workflow, flags=re.MULTILINE)
        self.assertTrue(action_refs)
        for action_ref in action_refs:
            with self.subTest(action_ref=action_ref):
                self.assertRegex(action_ref, r"@[0-9a-f]{40}$")

    def test_manual_dispatch_can_publish_an_explicit_release_tag(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("release_tag:", workflow)
        self.assertIn("tag_name:", workflow)
        self.assertIn("inputs.release_tag", workflow)

    def test_archives_are_guarded_against_github_size_limits(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("2GB", workflow)
        self.assertIn("compression-level: 0", workflow)

    def test_model_release_is_preflighted_and_assets_are_hash_verified(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("Verify model release exists", workflow)
        self.assertIn(".github/model-assets.sha256", workflow)
        self.assertIn("Get-FileHash", workflow)
        self.assertIn("Model hash mismatch for ${modelPath}:", workflow)

    def test_model_release_has_a_repeatable_non_latest_publish_script(self):
        self.assertTrue(MODEL_MANIFEST.exists())
        self.assertTrue(MODEL_PUBLISH_SCRIPT.exists())
        manifest = MODEL_MANIFEST.read_text(encoding="utf-8")
        script = MODEL_PUBLISH_SCRIPT.read_text(encoding="utf-8")

        for model_path in (
            "models/mrcnn/weights.onnx",
            "models/esrgan/4x-Fatal-Pixels.onnx",
            "models/esrgan/4x-Fatal-Pixels.onnx.data",
        ):
            self.assertIn(model_path, manifest)
        self.assertIn("Get-FileHash", script)
        self.assertIn("VerifyOnly", script)
        self.assertIn("--latest=false", script)
        self.assertIn("--clobber", script)
        self.assertIn("[string]$Repository", script)
        self.assertIn("--repo $Repository", script)
        self.assertIn("allowedModelPaths", script)
        self.assertIn("GetFullPath", script)

    def test_cuda_bundle_has_a_required_dll_manifest_check(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("Verify CUDA bundle DLLs", workflow)
        for dll in (
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "cudart64_12.dll",
            "cufft64_11.dll",
            "cudnn64_9.dll",
            "cudnn_adv64_9.dll",
            "cudnn_graph64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn_engines_tensor_ir64_9.dll",
            "onnxruntime_providers_cuda.dll",
        ):
            self.assertIn(dll, workflow)

    def test_workflow_smoke_tests_the_packaged_executable(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("Smoke-test packaged executable", workflow)
        self.assertIn("--runtime-smoke-test", workflow)
        self.assertIn("Get-Command nvidia-smi", workflow)
        self.assertIn("--require-cuda-provider", workflow)
        self.assertIn("--require-cuda", workflow)
        self.assertIn("ALETHEIA_SMOKE_REPORT", workflow)

    def test_manual_release_tag_cannot_point_to_another_commit(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("fetch-depth: 0", workflow)
        self.assertIn("Validate requested release tag", workflow)
        self.assertIn("refs/tags/$requestedTag^{commit}", workflow)
        self.assertIn("github.sha", workflow)

    def test_release_tag_is_passed_to_powershell_via_environment(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("RELEASE_TAG_INPUT: ${{ inputs.release_tag }}", workflow)
        self.assertIn("SOURCE_REF_NAME: ${{ github.ref_name }}", workflow)
        self.assertIn("CURRENT_SHA: ${{ github.sha }}", workflow)
        self.assertIn("$requestedTag = $env:RELEASE_TAG_INPUT", workflow)
        self.assertNotIn('$requestedTag = "${{ inputs.release_tag }}"', workflow)


if __name__ == "__main__":
    unittest.main()
