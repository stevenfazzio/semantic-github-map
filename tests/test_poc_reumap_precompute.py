"""Tests for pure functions in poc_reumap_precompute.py."""

import importlib
import sys
from pathlib import Path

import numpy as np

# Ensure pipeline/ is on sys.path so config imports work inside the script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

spec = importlib.util.spec_from_file_location("poc_reumap_precompute", "experiments/poc_reumap_precompute.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["poc_reumap_precompute"] = mod
spec.loader.exec_module(mod)

remap_knn_to_subset = mod.remap_knn_to_subset
rescale_coords = mod.rescale_coords


class TestKnnSubsetRemapping:
    def test_basic_remapping(self):
        """Given a 5-node global graph with k=3, remap to a 3-node subset."""
        # Global KNN: node 0's neighbors are [1,2,3], node 2's are [0,1,4], etc.
        knn_indices = np.array([[1, 2, 3], [0, 2, 4], [0, 1, 4], [0, 1, 4], [1, 2, 3]], dtype=np.int32)
        knn_distances = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.15, 0.4],
                [0.2, 0.15, 0.35],
                [0.3, 0.25, 0.5],
                [0.4, 0.35, 0.5],
            ],
            dtype=np.float32,
        )
        # Subset: global indices [0, 2, 4]  → local [0, 1, 2]
        subset = np.array([0, 2, 4])
        ri, rd = remap_knn_to_subset(knn_indices, knn_distances, subset)

        assert ri.shape == (3, 3)
        assert rd.shape == (3, 3)

        # Node 0 (global 0): neighbors [1,2,3] → only 2 is in subset → local [1]
        assert ri[0, 0] == 1  # global 2 → local 1
        assert np.isclose(rd[0, 0], 0.2)
        assert ri[0, 1] == -1  # padded

        # Node 1 (global 2): neighbors [0,1,4] → 0 and 4 in subset → local [0, 2]
        assert ri[1, 0] == 0  # global 0 → local 0
        assert ri[1, 1] == 2  # global 4 → local 2
        assert ri[1, 2] == -1

    def test_empty_subset(self):
        knn_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
        knn_distances = np.array([[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]], dtype=np.float32)
        subset = np.array([0])
        ri, rd = remap_knn_to_subset(knn_indices, knn_distances, subset)
        # Node 0's neighbors [1,2] — neither in subset
        assert np.all(ri == -1)
        assert np.all(np.isinf(rd))


class TestCoordRescaling:
    def test_rescale_preserves_aspect_ratio(self):
        coords = np.array([[0, 0], [10, 0], [10, 5]], dtype=np.float32)
        result = rescale_coords(coords)
        # Larger span is x=10, so scale = 18/10 = 1.8
        # x range: centered [-5,5] * 1.8 = [-9, 9]
        assert result.max() <= 9.0
        assert result.min() >= -9.0

    def test_rescale_symmetric(self):
        coords = np.array([[-1, -1], [1, 1]], dtype=np.float32)
        result = rescale_coords(coords)
        assert np.isclose(result[0, 0], -9.0)
        assert np.isclose(result[1, 0], 9.0)

    def test_rescale_output_dtype(self):
        coords = np.array([[0, 0], [1, 1]], dtype=np.float64)
        result = rescale_coords(coords)
        assert result.dtype == np.float32


class TestBinaryRoundtrip:
    def test_float32_roundtrip(self, tmp_path):
        original = np.random.randn(100, 512).astype(np.float32)
        path = tmp_path / "test.bin"
        original.tofile(path)
        loaded = np.fromfile(path, dtype=np.float32).reshape(100, 512)
        np.testing.assert_array_equal(original, loaded)

    def test_uint16_roundtrip(self, tmp_path):
        original = np.arange(500, dtype=np.uint16).reshape(10, 50)
        path = tmp_path / "test.bin"
        original.tofile(path)
        loaded = np.fromfile(path, dtype=np.uint16).reshape(10, 50)
        np.testing.assert_array_equal(original, loaded)
