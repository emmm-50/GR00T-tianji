"""Load a native Tianji observation snapshot without ROS or robot access."""

import json
from pathlib import Path
import numpy as np


def load_snapshot(path):
    """Return (arrays, metadata) from a snapshot NPZ and its sidecar JSON.

    Arrays: right_eye and right_wrist [2,H,W,3] uint8 RGB, oldest first;
    eef_xyz [3] metres, eef_xyzw [4], joint_names [N], joint_positions
    [N] radians, gripper_motor [N] raw feedback. Metadata records frame,
    timestamps, source, and the limits of synchronization.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    metadata = json.loads(path.with_suffix(".json").read_text())
    return arrays, metadata
