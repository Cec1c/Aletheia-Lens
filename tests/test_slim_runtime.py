from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from decensor import find_regions
from tools.green_mask_project_mosaic_resolution import _local_minima_indices


ROOT = Path(__file__).resolve().parents[1]


class SlimRuntimeTests(unittest.TestCase):
    def test_connected_regions_keep_diagonal_pixels_separate(self):
        pixels = np.zeros((3, 3, 3), dtype=np.uint8)
        pixels[0, 0] = [0, 255, 0]
        pixels[1, 1] = [0, 255, 0]

        regions = find_regions(Image.fromarray(pixels), [0, 255, 0])

        normalized = {frozenset(tuple(point) for point in region) for region in regions}
        self.assertEqual(normalized, {frozenset({(0, 0)}), frozenset({(1, 1)})})

    def test_connected_regions_are_sorted_by_size(self):
        pixels = np.zeros((4, 5, 3), dtype=np.uint8)
        pixels[0, 0] = [0, 255, 0]
        pixels[2, 2:5] = [0, 255, 0]

        regions = find_regions(Image.fromarray(pixels), [0, 255, 0])

        self.assertEqual([len(region) for region in regions], [3, 1])

    def test_local_minima_matches_strict_neighbor_comparison(self):
        values = np.array([5, 2, 2, 4, 1, 3, 3, 0, 4])

        np.testing.assert_array_equal(_local_minima_indices(values), [4, 7])

    def test_runtime_dependencies_do_not_include_scipy_or_scikit_image(self):
        common = (ROOT / "requirements-common.txt").read_text(encoding="utf-8").lower()
        spec = (ROOT / "main.spec").read_text(encoding="utf-8").lower()
        analysis_inputs = spec.split("excludes=[", 1)[0]

        self.assertNotIn("scipy", common)
        self.assertNotIn("scikit-image", common)
        self.assertNotIn("'scipy'", analysis_inputs)
        self.assertNotIn("'skimage'", analysis_inputs)
        self.assertIn("'scipy'", spec)
        self.assertIn("'skimage'", spec)

    def test_cuda_build_removes_nested_nvidia_binary_duplicates(self):
        spec = (ROOT / "main.spec").read_text(encoding="utf-8")

        self.assertIn("def remove_nested_nvidia_duplicates", spec)
        self.assertIn(
            "a.binaries = remove_nested_nvidia_duplicates(a.binaries)",
            spec,
        )


if __name__ == "__main__":
    unittest.main()
