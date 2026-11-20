"""Configurable paths shared by the Tianji adapters and shadow runner."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EEF_CONTROL = Path(os.environ.get("TIANJI_EEF_CONTROL", ROOT.parent / "pi05" / "eef-control"))
URDF = EEF_CONTROL / "native_ik/robot_model/urdf/Pro-m6-696.urdf"
BAG = Path(os.environ["TIANJI_BAG"]) if os.environ.get("TIANJI_BAG") else None
NAMES = [f"Joint{i}_{side}" for side in ("L", "R") for i in range(1, 8)]

IK_BINARY = Path(os.environ.get("GR00T_IK_BINARY", "/tmp/gr00t-eef-module-build/native_ik"))
IK_CONFIG = EEF_CONTROL / "native_ik" / "ik_tasks.yaml"
