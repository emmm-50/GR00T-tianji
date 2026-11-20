"""Reproduce bag FK comparisons and the two snapshot FK anchors offline."""

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from tianji_adapter.capture_ros import read_bag
from tianji_adapter.geometry import nominal_world_right_tool
from tianji_adapter.observations import load_snapshot

from tianji_paths import BAG, NAMES, ROOT, URDF

EEF, JOINTS = "/tj/info/eef_right", "/tj/joint_states"


def snapshots(paths):
    out = {}
    for path in paths:
        snapshot, meta = load_snapshot(path)
        p, q = nominal_world_right_tool(snapshot["joint_names"], snapshot["joint_positions"], URDF)
        out[str(path)] = {
            "position_m": p.tolist(), "quaternion_xyzw": q.tolist(),
            "frame_id": "world", "tool_frame_id": "right_tool",
            "source": "measured joints and nominal URDF FK",
            "observation_timestamp_ns": meta["observation_timestamp_ns"],
            "observation_clock": meta["observation_clock"],
            "joint_header_stamp_ns": meta["joint_header_stamp_ns"],
        }
    return out


def summarize(values):
    out = {}
    for key, rows in values.items():
        if rows:
            a = np.array(rows)
            out[key] = {
                "n": len(a),
                "median_position_m": float(np.median(a[:, 0])),
                "max_position_m": float(np.max(a[:, 0])),
                "median_rotation_rad": float(np.median(a[:, 1])),
                "max_rotation_rad": float(np.max(a[:, 1])),
            }
    return out


def compare(bag, window_timestamps=None):
    bag = Path(bag) / "data"
    # Keep numeric joint samples only; the second pass pairs by bag timestamps,
    # including equal timestamps independently of serialization ordering.
    stamps, joint_values = [], []
    for _, ts, msg in read_bag(bag, [JOINTS]):
        stamps.append(ts)
        joint_values.append([msg.position[msg.name.index(name)] for name in NAMES])
    model = pin.buildModelFromUrdf(str(URDF))
    data = model.createData()
    indices = [model.joints[model.getJointId(name)].idx_q for name in NAMES]
    frames = ["right_tool", "TCP_Link_R", "Link7_R"]
    world = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    errors = {f"{space}/{frame}": [] for space in ("urdf", "apex") for frame in frames}
    window = {key: [] for key in errors}
    ages, frame_ids = [], set()
    for _, ts, msg in read_bag(bag, [EEF]):
        i = bisect.bisect_right(stamps, ts) - 1
        if i < 0:
            continue
        config = pin.neutral(model)
        config[indices] = joint_values[i]
        pin.forwardKinematics(model, data, config)
        pin.updateFramePlacements(model, data)
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = msg.pose.orientation
        r = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        frame_ids.add(msg.header.frame_id)
        ages.append((ts - stamps[i]) / 1e6)
        for space, w in (("urdf", np.eye(3)), ("apex", world)):
            for frame in frames:
                key = f"{space}/{frame}"
                pose = data.oMf[model.getFrameId(frame)]
                item = [
                    float(np.linalg.norm(p - w @ pose.translation)),
                    float(Rotation.from_matrix(r.T @ w @ pose.rotation).magnitude()),
                ]
                errors[key].append(item)
                if window_timestamps and window_timestamps[0] <= ts <= window_timestamps[-1]:
                    window[key].append(item)
    out = {
        "bag": str(bag.parent),
        "model": "native_ik/robot_model/urdf/Pro-m6-696.urdf",
        "matching": "preceding joint bag timestamp for each eef bag timestamp",
        "eef_frame_ids": sorted(frame_ids),
        "joint_age_ms_median": float(np.median(ages)),
        "joint_age_ms_max": float(np.max(ages)),
        "all": summarize(errors),
        "video_time_window": summarize(window) if window_timestamps else None,
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, default=BAG, required=BAG is None,
                        help="recording root (or set TIANJI_BAG)")
    parser.add_argument("--snapshot", type=Path, action="append", default=[],
                        help="captured NPZ to compare with nominal FK; repeatable")
    parser.add_argument("--window-snapshot", type=Path,
                        help="recording NPZ whose two video timestamps define the comparison window")
    args = parser.parse_args()
    window = None
    if args.window_snapshot:
        _, meta = load_snapshot(args.window_snapshot)
        window = meta["video_bag_timestamps_ns"]
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    fk = compare(args.bag, window)
    (results / "adapter_geometry_fk.json").write_text(json.dumps(fk, indent=2) + "\n")
    if args.snapshot:
        result = snapshots(args.snapshot)
        (results / "adapter_geometry_snapshot_fk.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"bag": fk, "snapshots": len(args.snapshot)}, indent=2))


if __name__ == "__main__":
    main()
