"""Explicit frame conversion for absolute GR00T EEF poses.

Transforms use T_A_B (B coordinates expressed in A), metres and xyzw.
DROID correction is a representation convention, not a physical TCP offset.
"""

import warnings
import numpy as np
from scipy.spatial.transform import Rotation

DROID_C = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def pose_matrix(position_m, quaternion_xyzw):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    t[:3, 3] = position_m
    return t


def _rigid(t):
    t = np.asarray(t, dtype=float)
    if (
        t.shape != (4, 4)
        or not np.all(np.isfinite(t))
        or not np.allclose(t[3], [0, 0, 0, 1])
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-7)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-7)
    ):
        raise ValueError("expected a proper rigid transform")
    return t


def pack_eef_matrix(position_m, rotation_matrix, rotation_correction):
    r = np.asarray(rotation_matrix) @ np.asarray(rotation_correction)
    return np.r_[position_m, r[:2].reshape(6)]


def _euler(rotation, axes):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Gimbal lock detected.*")
        try:
            return rotation.as_euler(axes)
        except UserWarning as exc:
            raise ValueError("DROID Euler mapping is ambiguous at gimbal lock") from exc


def encode_droid_pose(position_m, quaternion_xyzw):
    """DROID physical pose in its own base frame, with DROID C."""
    physical = Rotation.from_quat(quaternion_xyzw)
    angles = _euler(physical, "xyz")
    model_rotation = Rotation.from_euler("XYZ", angles).as_matrix()
    return pack_eef_matrix(position_m, model_rotation, DROID_C)


def encode(
    position_m, quaternion_xyzw, *, model_base_from_world, right_tool_from_model_tool
):
    """world/right_tool pose to model XYZ + first two rotation rows."""
    base = _rigid(model_base_from_world)
    tool = _rigid(right_tool_from_model_tool)
    t = base @ pose_matrix(position_m, quaternion_xyzw) @ tool
    return encode_droid_pose(t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_quat())


def decode_droid_matrix(eef_9d):
    """Decode policy pose into physical DROID base/tool transform."""
    x = np.asarray(eef_9d, dtype=float)
    if x.shape != (9,) or not np.all(np.isfinite(x)):
        raise ValueError("eef_9d must contain nine finite values")
    a, b = x[3:6], x[6:9]
    if np.linalg.norm(a) < 1e-8:
        raise ValueError("degenerate first rotation row")
    a = a / np.linalg.norm(a)
    b = b - np.dot(a, b) * a
    if np.linalg.norm(b) < 1e-8:
        raise ValueError("degenerate second rotation row")
    b = b / np.linalg.norm(b)
    r_model = np.stack((a, b, np.cross(a, b))) @ DROID_C.T
    angles = _euler(Rotation.from_matrix(r_model), "XYZ")
    r = Rotation.from_euler("xyz", angles).as_matrix()
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, x[:3]
    return t


def decode(eef_9d, *, model_base_from_world, right_tool_from_model_tool):
    model_base_from_world = _rigid(model_base_from_world)
    right_tool_from_model_tool = _rigid(right_tool_from_model_tool)
    t = decode_droid_matrix(eef_9d)
    t = (
        np.linalg.inv(model_base_from_world)
        @ t
        @ np.linalg.inv(right_tool_from_model_tool)
    )
    return t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_quat()


def ik_targets(
    eef_chunk,
    gripper_chunk,
    *,
    observation_timestamp_ns,
    observation_clock,
    model_base_from_world,
    right_tool_from_model_tool,
):
    """Decode absolute actions with nominal offsets from an observed timestamp.

    The timestamp identifies its clock; offset is nominal, not a scheduled command.
    No joint commands are generated. Gripper values are independent passthrough.
    """
    eef = np.asarray(eef_chunk)
    grip = np.asarray(gripper_chunk).reshape(-1)
    if eef.ndim != 2 or eef.shape[1] != 9 or len(eef) != len(grip):
        raise ValueError("expected EEF (N,9) and one gripper value per action")
    if not observation_clock or not np.all(np.isfinite(grip)):
        raise ValueError("observation clock and finite gripper values required")
    world_from_base = np.linalg.inv(_rigid(model_base_from_world))
    model_tool_from_right_tool = np.linalg.inv(_rigid(right_tool_from_model_tool))
    targets = []
    for i, (pose, gripper) in enumerate(zip(eef, grip)):
        t = world_from_base @ decode_droid_matrix(pose) @ model_tool_from_right_tool
        p, q = t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_quat()
        targets.append(
            {
                "index": i,
                "nominal_offset_s": i / 15,
                "observation_timestamp_ns": int(observation_timestamp_ns),
                "observation_clock": observation_clock,
                "position_m": p.tolist(),
                "quaternion_xyzw": q.tolist(),
                "frame_id": "world",
                "tool_frame_id": "right_tool",
                "gripper_position_droid": float(gripper),
            }
        )
    return targets


def nominal_world_right_tool(joint_names, joint_positions, urdf_path):
    """FK from named measured joints; world is APEX convention, not calibration."""
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    q = pin.neutral(model)
    measured = dict(zip(joint_names, joint_positions))
    for side in "LR":
        for i in range(1, 8):
            name = f"Joint{i}_{side}"
            q[model.joints[model.getJointId(name)].idx_q] = measured[name]
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pose = data.oMf[model.getFrameId("right_tool")]
    w = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    return (w @ pose.translation, Rotation.from_matrix(w @ pose.rotation).as_quat())


def anchored_ik_targets(
    observed_eef_9d,
    world_right_tool_position_m,
    world_right_tool_quaternion_xyzw,
    eef_chunk,
    gripper_chunk,
    *,
    observation_timestamp_ns,
    observation_clock,
    right_tool_from_model_tool,
):
    """Conditional relative alignment using one observation anchor; offline only.

    The caller supplies a meaningful cross-robot tool mapping; no mapping is inferred.
    Both observed and predicted absolute poses are first decoded to physical poses.
    """
    model_base_from_model_tool = decode_droid_matrix(observed_eef_9d)
    world_from_model_tool = pose_matrix(
        world_right_tool_position_m, world_right_tool_quaternion_xyzw
    ) @ _rigid(right_tool_from_model_tool)
    model_base_from_world = model_base_from_model_tool @ np.linalg.inv(
        world_from_model_tool
    )
    return ik_targets(
        eef_chunk,
        gripper_chunk,
        observation_timestamp_ns=observation_timestamp_ns,
        observation_clock=observation_clock,
        model_base_from_world=model_base_from_world,
        right_tool_from_model_tool=right_tool_from_model_tool,
    )
