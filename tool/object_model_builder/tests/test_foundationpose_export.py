#!/usr/bin/env python3

import unittest

import numpy as np

from tool.object_model_builder.foundationpose_export import validate_mesh


class FakeMesh:
    def __init__(self, dimensions, watertight=True):
        x, y, z = dimensions
        corners = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [x, 0.0, 0.0],
                [0.0, y, 0.0],
                [x, y, 0.0],
                [0.0, 0.0, z],
                [x, 0.0, z],
                [0.0, y, z],
                [x, y, z],
            ],
            dtype=np.float64,
        )
        self.vertices = np.tile(corners, (16, 1))
        self.triangles = np.tile(
            np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64), (64, 1)
        )
        self._watertight = bool(watertight)

    def is_watertight(self):
        return self._watertight


class MeshQualityGateTests(unittest.TestCase):
    def test_rejects_mesh_below_configured_bottle_height(self):
        validation = validate_mesh(
            FakeMesh((0.13, 0.12, 0.088)),
            minimum_dimensions_m=(0.02, 0.02, 0.12),
            maximum_dimensions_m=(0.30, 0.30, 0.50),
            require_watertight=True,
        )
        self.assertFalse(validation.valid)
        self.assertIn("below", validation.reason)

    def test_rejects_open_mesh_when_watertight_is_required(self):
        validation = validate_mesh(
            FakeMesh((0.08, 0.08, 0.20), watertight=False),
            minimum_dimensions_m=(0.02, 0.02, 0.12),
            maximum_dimensions_m=(0.30, 0.30, 0.50),
            require_watertight=True,
        )
        self.assertFalse(validation.valid)
        self.assertEqual(validation.reason, "mesh is not watertight")

    def test_accepts_metric_watertight_mesh_inside_limits(self):
        validation = validate_mesh(
            FakeMesh((0.08, 0.07, 0.22), watertight=True),
            minimum_dimensions_m=(0.02, 0.02, 0.12),
            maximum_dimensions_m=(0.30, 0.30, 0.50),
            require_watertight=True,
        )
        self.assertTrue(validation.valid)


if __name__ == "__main__":
    unittest.main()
