"""Check both directions of the DROID representation against the original Euler columns."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from tianji_adapter.geometry import DROID_C, decode_droid_matrix, encode_droid_pose

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = {"state": [], "action": []}
    for path in sorted((ROOT / "data/droid_sample/data/chunk-000").glob("*.parquet")):
        table = pq.read_table(path)
        for key, prefix in (("state", "observation.state"), ("action", "action")):
            cart = table[f"{prefix}.cartesian_position"].to_pylist()
            poses = table[f"{prefix}.eef_9d"].to_pylist()
            for raw, eef in zip(cart, poses):
                physical = Rotation.from_euler("xyz", raw[3:])
                encoded = encode_droid_pose(raw[:3], physical.as_quat())
                decoded = Rotation.from_matrix(decode_droid_matrix(eef)[:3, :3])
                # Recover the archive's policy matrix to measure the effect of undoing C alone.
                a, b = np.reshape(eef[3:], (2, 3))
                undo_c = Rotation.from_matrix(
                    np.stack([a, b, np.cross(a, b)]) @ DROID_C.T
                )
                rows[key].append(
                    (
                        np.max(np.abs(encoded - eef)),
                        (physical.inv() * decoded).magnitude(),
                        (physical.inv() * undo_c).magnitude(),
                    )
                )
    result = {}
    for key, values in rows.items():
        encoding_error, rotation_error, undo_c_error = np.asarray(values).T
        result[key] = {
            "samples": len(values),
            "max_encoding_abs": float(encoding_error.max()),
            "max_physical_rotation_error_rad": float(rotation_error.max()),
            "undo_c_only_median_physical_rotation_error_rad": float(
                np.median(undo_c_error)
            ),
        }
    report = json.dumps(result, indent=2, allow_nan=False)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/adapter_geometry_droid.json").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
