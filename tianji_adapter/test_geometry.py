import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
from tianji_adapter.geometry import encode, decode, pose_matrix


class GeometryTest(unittest.TestCase):
    def test_known_nontrivial_transform(self):
        # world pose Rz(90), p=(1,2,3); tool offset +x becomes world +y.
        base = pose_matrix([10, 0, 0], [0, 0, 0, 1])
        tool = pose_matrix([0.5, 0, 0], [0, 0, 0, 1])
        args = dict(model_base_from_world=base, right_tool_from_model_tool=tool)
        q = Rotation.from_euler("z", 90, degrees=True).as_quat()
        value = encode([1, 2, 3], q, **args)
        assert_allclose(value, [11, 2.5, 3, 1, 0, 0, 0, 0, -1], atol=1e-14)
        p, out_q = decode([11, 2.5, 3, 1, 0, 0, 0, 0, -1], **args)
        assert_allclose(p, [1, 2, 3], atol=1e-14)
        assert_allclose(
            Rotation.from_quat(out_q).as_matrix(),
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            atol=1e-14,
        )


class ChunkTest(unittest.TestCase):
    def test_absolute_actions_and_timing(self):
        from tianji_adapter.geometry import ik_targets

        eye = np.eye(4)
        actions = np.array(
            [[1, 2, 3, 0, 0, -1, -1, 0, 0], [4, 5, 6, 0, 0, -1, -1, 0, 0]]
        )
        result = ik_targets(
            actions,
            [0.2, 0.7],
            observation_timestamp_ns=123,
            observation_clock="bag",
            model_base_from_world=eye,
            right_tool_from_model_tool=eye,
        )
        assert_allclose(result[1]["position_m"], [4, 5, 6])
        self.assertEqual(result[1]["nominal_offset_s"], 1 / 15)
        self.assertEqual(result[1]["observation_timestamp_ns"], 123)
        self.assertEqual(result[1]["gripper_position_droid"], 0.7)
        with self.assertRaises(ValueError):
            ik_targets(
                actions,
                [0.2, float("nan")],
                observation_timestamp_ns=123,
                observation_clock="bag",
                model_base_from_world=eye,
                right_tool_from_model_tool=eye,
            )


class AnchoredTest(unittest.TestCase):
    def test_anchor_repeated_absolute_targets(self):
        from tianji_adapter.geometry import anchored_ik_targets, encode_droid_pose

        physical = Rotation.from_euler("xyz", [0.4, -0.3, 0.7])
        obs = encode_droid_pose([0.4, 0.2, 0.8], physical.as_quat())
        tool = pose_matrix([0.1, 0.03, -0.02], Rotation.from_euler("x", 0.2).as_quat())
        world_q = Rotation.from_euler("xyz", [0.2, 0.3, -0.5]).as_quat()
        target_q = Rotation.from_euler("xyz", [-0.2, 0.25, 0.9]).as_quat()
        target = encode_droid_pose([0.5, 0.1, 0.9], target_q)
        targets = anchored_ik_targets(
            obs,
            [1, 2, 3],
            world_q,
            [target, target],
            [0.3, 0.4],
            observation_timestamp_ns=1,
            observation_clock="bag",
            right_tool_from_model_tool=tool,
        )
        observed = pose_matrix([0.4, 0.2, 0.8], physical.as_quat())
        expected = (
            pose_matrix([1, 2, 3], world_q)
            @ tool
            @ np.linalg.inv(observed)
            @ pose_matrix([0.5, 0.1, 0.9], target_q)
            @ np.linalg.inv(tool)
        )
        for item in targets:
            assert_allclose(item["position_m"], expected[:3, 3], atol=1e-12)
            assert_allclose(
                Rotation.from_quat(item["quaternion_xyzw"]).as_matrix(),
                expected[:3, :3],
                atol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
