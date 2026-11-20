"""Pack two-view Tianji snapshots for a DROID-conditioned policy, or run a format probe."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import DROID_C, decode_droid_matrices, pack_eef_matrix
from .observations import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
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


def format_probe_state(snapshot):
    """Reproduce the uncalibrated historical inputs for diagnostics."""
    names = snapshot["joint_names"].tolist()
    q_indices = [names.index(f"Joint{i}_R") for i in range(1, 8)]
    return {
        "eef_9d": pack_eef_matrix(
            snapshot["eef_xyz"],
            Rotation.from_quat(snapshot["eef_xyzw"]).as_matrix(),
            DROID_C,
        ),
        "joint_position": snapshot["joint_positions"][q_indices],
        "gripper_position": np.clip(snapshot["gripper_motor"][:1] / 1.6, 0, 1),
    }


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


def main():
    parser = argparse.ArgumentParser(
        description="在已保存原生快照上运行旧跨具身格式探针，仅输出策略坐标。"
    )
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Keep observation packing usable without model initialization.
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    tag = EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(MODEL), device="cuda")
    policy.model.action_head.num_inference_timesteps = 4
    modality = policy.get_modality_config()
    quantiles = json.loads((MODEL / "statistics.json").read_text())[tag.value]["state"]
    indices = modality["action"].delta_indices
    fps = json.loads((ROOT / "data/droid_sample/meta/info.json").read_text())["fps"]
    result = {
        "scope": "format_probe_only",
        "embodiment_tag": tag.value,
        "instruction": args.instruction,
        "input_assumptions": [
            "right_eye substituted for exterior_1_left",
            "right_wrist substituted for wrist_left",
            "base_link xyz and raw rotation @ C, matching the historical input",
            "Tianji right joints in DROID slots, no correspondence established",
            "clip(motor[0] / 1.6, 0, 1) is a proxy, not physical closure",
        ],
        "sample_hz": fps,
        "nominal_action_offsets_s": [i / fps for i in indices],
        "seed_per_snapshot": 42,
        "denoising_steps": 4,
        "records": [],
    }
    for path in args.snapshots:
        snapshot, meta = load_snapshot(path)
        state = format_probe_state(snapshot)
        obs = policy_observation(snapshot, state, args.instruction, modality)
        torch.manual_seed(42)
        actions, _ = policy.get_action(obs)
        chunk = {k: np.asarray(v)[0] for k, v in actions.items()}
        result["records"].append(
            {
                "snapshot": str(path),
                "observation_timestamp_ns": meta["observation_timestamp_ns"],
                "observation_clock": meta["observation_clock"],
                "state": {k: v.tolist() for k, v in state.items()},
                "diagnostics": summarize(state, chunk, quantiles),
                "api_absolute_actions": {k: v.tolist() for k, v in chunk.items()},
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
