"""Check DROID pose encoding and decoding against the public parquet columns."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from tianji_adapter.geometry import decode_droid_matrix, encode_droid_pose

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/droid_sample/data/chunk-000"


def main():
    paths = sorted(DATA.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"No DROID parquet files found in {DATA}")

    for label, prefix in (("state", "observation.state"), ("action", "action")):
        count = 0
        max_encoding_error = 0.0
        max_rotation_error = 0.0
        for path in paths:
            columns = [f"{prefix}.cartesian_position", f"{prefix}.eef_9d"]
            table = pq.read_table(path, columns=columns)
            for raw, eef in zip(*(table[column].to_pylist() for column in columns)):
                physical = Rotation.from_euler("xyz", raw[3:])
                encoded = encode_droid_pose(raw[:3], physical.as_quat())
                decoded = Rotation.from_matrix(decode_droid_matrix(eef)[:3, :3])
                max_encoding_error = max(max_encoding_error, float(np.max(np.abs(encoded - eef))))
                max_rotation_error = max(max_rotation_error, float((physical.inv() * decoded).magnitude()))
                count += 1
        if not count:
            raise SystemExit(f"No {label} rows found in {DATA}")
        print(f"{label}: {count} rows; max encoding error {max_encoding_error:.3g}; "
              f"max rotation error {max_rotation_error:.3g} rad")


if __name__ == "__main__":
    main()
