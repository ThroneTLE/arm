"""Pure-numpy tests for the grasp candidate pipeline (no AnyGrasp SDK needed)."""

import unittest

import numpy as np

from tool.grasp_planning.anygrasp_planner import (
    GraspCandidate,
    filter_by_score,
    filter_by_width,
    filter_top_down,
    transform_to_workspace,
)


def _candidate(translation=(0.0, 0.0, 0.5), rpy_deg=(0.0, 0.0, 0.0), width=0.06, score=0.8, depth=0.02, height=0.03):
    from tf.transformations import euler_matrix

    rotation = np.asarray(euler_matrix(*np.deg2rad(rpy_deg), axes="sxyz"), dtype=np.float64)[:3, :3]
    return GraspCandidate(
        translation=np.asarray(translation, dtype=np.float64),
        rotation=rotation,
        width=width,
        score=score,
        depth=depth,
        height=height,
    )


class GraspCandidateTest(unittest.TestCase):
    def test_tip_moves_along_approach(self):
        # RPY (0,0,0) -> X axis is approach; tip = translation + depth * X.
        candidate = _candidate(translation=(0.1, 0.2, 0.5))
        np.testing.assert_allclose(candidate.tip, np.asarray([0.12, 0.2, 0.5]), atol=1e-12)
        np.testing.assert_allclose(candidate.approach, np.asarray([1.0, 0.0, 0.0]), atol=1e-12)

    def test_transform_to_workspace_translation(self):
        candidate = _candidate(translation=(0.1, 0.0, 0.5))
        workspace_from_camera = np.eye(4)
        workspace_from_camera[:3, 3] = [0.3, -0.2, 0.0]
        output = transform_to_workspace([candidate], workspace_from_camera)
        self.assertEqual(len(output), 1)
        np.testing.assert_allclose(output[0].translation, np.asarray([0.4, -0.2, 0.5]), atol=1e-12)
        np.testing.assert_allclose(output[0].tip, np.asarray([0.42, -0.2, 0.5]), atol=1e-12)

    def test_transform_rotation_composes(self):
        from tf.transformations import euler_matrix

        candidate = _candidate(rpy_deg=(0.0, 0.0, 90.0))
        workspace_from_camera = euler_matrix(0.0, 0.0, np.deg2rad(-90.0))
        output = transform_to_workspace([candidate], workspace_from_camera)
        expected = np.eye(4)[:3, :3]
        np.testing.assert_allclose(output[0].rotation, expected, atol=1e-10)

    def test_filter_top_down_keeps_downward_approach(self):
        # Workspace +Z points into the table, so approach must point toward +Z.
        up = np.asarray([0.0, 0.0, -1.0])
        downward = _candidate(translation=(0.0, 0.0, 0.4), rpy_deg=(0.0, 90.0, 0.0))
        # RPY (0, 90, 0): X axis rotates to... compute numerically instead.
        import itertools

        rotations = []
        for pitch in (90.0, -90.0):
            rotations.append(_candidate(rpy_deg=(0.0, pitch, 0.0)))
        kept = filter_top_down(rotations, workspace_up=up, max_deviation_deg=60.0)
        self.assertEqual(len(kept), 1)
        self.assertGreater(float(np.dot(kept[0].approach, np.asarray([0.0, 0.0, 1.0]))), 0.0)

    def test_filter_top_down_rejects_upward_approach(self):
        up = np.asarray([0.0, 0.0, -1.0])
        upward = _candidate(translation=(0.0, 0.0, 0.4), rpy_deg=(180.0, 0.0, 0.0))
        # RPY (180,0,0): X -> -X which is horizontal; check a clearly upward one
        # by building a rotation whose X axis equals workspace -Z (up).
        rotation = np.eye(3)
        rotation[:, 0] = [0.0, 0.0, -1.0]
        rotation[:, 1] = [1.0, 0.0, 0.0]
        rotation[:, 2] = [0.0, 1.0, 0.0]
        candidate = _candidate()
        candidate.rotation = rotation
        kept = filter_top_down([candidate], workspace_up=up, max_deviation_deg=60.0)
        self.assertEqual(len(kept), 0)

    def test_filters(self):
        candidates = [
            _candidate(width=0.03, score=0.9),
            _candidate(width=0.09, score=0.5),
            _candidate(width=0.05, score=0.1),
        ]
        self.assertEqual(len(filter_by_width(candidates, 0.02, 0.08)), 2)
        self.assertEqual(len(filter_by_score(candidates, 0.4)), 2)


if __name__ == "__main__":
    unittest.main()
