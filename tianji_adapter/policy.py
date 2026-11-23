"""Pack observations and summarize predictions for the shadow policy."""

import numpy as np
from scipy.spatial.transform import Rotation

from tianji_paths import ROOT
from .geometry import decode_droid_matrices

MODEL = ROOT / "models/GR00T-N1.7-3B"
STATE_WIDTHS = {"eef_9d": 9, "joint_position": 7, "gripper_position": 1}


def policy_observation(snapshot, droid_state, instruction, modality):
    """Use caller-supplied, already DROID-encoded state at the snapshot time.

    Video inputs use Tianji's right eye / right wrist as alternate viewpoints.
    This boundary does not infer a Franka joint posture or a jaw width.
    """
    from gr00t.data.utils import parse_observation_gr00t

    flat = {modality["language"].modality_keys[0]: instruction}
    for model_key, camera in (
        ("exterior_image_1_left", "right_eye"),
        ("wrist_image_left", "right_wrist"),
    ):
        video = np.asarray(snapshot[camera])
        if (
            video.ndim != 4
            or video.shape[0] != 2
            or video.shape[-1] != 3
            or video.dtype != np.uint8
        ):
            raise ValueError(f"{camera} must be two RGB uint8 images, oldest first")
        flat[f"video.{model_key}"] = video
    for key, width in STATE_WIDTHS.items():
        state = np.asarray(droid_state[key], dtype=np.float32)
        if state.shape != (width,) or not np.isfinite(state).all():
            raise ValueError(f"{key} must have shape ({width},) and finite values")
        flat[f"state.{key}"] = state[None, :]
    return parse_observation_gr00t(flat, modality)


def summarize(state, chunk, quantiles):
    outside = {}
    for key, values in state.items():
        stats = quantiles[key]
        mask = (values < np.array(stats["q01"])) | (values > np.array(stats["q99"]))
        outside[key] = (1 + np.flatnonzero(mask)).tolist()
    eef = chunk["eef_9d"]
    offsets = 1000 * np.linalg.norm(eef[:, :3] - state["eef_9d"][:3], axis=1)
    droid_rotations = Rotation.from_matrix(
        decode_droid_matrices(np.vstack((state["eef_9d"], eef)))[:, :3, :3]
    )
    turns_deg = np.rad2deg((droid_rotations[0].inv() * droid_rotations[1:]).magnitude())
    # Include the transition from the observed DROID state to the first target.
    positions = np.vstack((state["eef_9d"][:3], eef[:, :3]))
    jumps_mm = 1000 * np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return {
        "outside_state_q01_q99_dimensions_one_based": outside,
        "max_adjacent_translation_mm_including_observation": float(jumps_mm.max()),
        "max_droid_orientation_change_from_observation_deg": float(turns_deg.max()),
        "xyz_distance_from_observation_mm": {
            "first": float(offsets[0]),
            "last": float(offsets[-1]),
            "max": float(offsets.max()),
        },
        "gripper_position_prediction_range": [
            float(chunk["gripper_position"].min()),
            float(chunk["gripper_position"].max()),
        ],
    }
