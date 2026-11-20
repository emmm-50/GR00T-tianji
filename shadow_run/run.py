"""Finite GR00T runs with recorded or live read-only snapshots and an in-memory Host."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from tianji_adapter.capture_ros import read_live, recording, save_snapshot
from tianji_paths import BAG, IK_BINARY, IK_CONFIG, NAMES, ROOT, URDF
from tianji_adapter.geometry import nominal_world_right_tool
from .receiver import Receiver, slot_time_ns

NS = 1_000_000_000


def read_worker(worker, log):
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError(f"policy worker exited ({worker.wait()}); see {log}")
    return json.loads(line)


@contextmanager
def model_worker(out):
    log = out / "policy.log"
    # Only the capture/IK process uses the ROS Python and native library paths.
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", NO_ALBUMENTATIONS_UPDATE="1", OMP_NUM_THREADS="2")
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "VIRTUAL_ENV"):
        env.pop(key, None)
    with log.open("w") as stderr, subprocess.Popen(
        ["bash", "-c", "source ./activate.sh >&2 && exec python -m shadow_run.prediction"],
        cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
        text=True, bufsize=1,
    ) as worker:
        try:
            loaded = read_worker(worker, log)
            yield worker, loaded
        finally:
            worker.stdin.close()


def process_block(worker, receiver, args, name, sample, next_obs=None):
    path, arrays, meta, capture_s, saved_s = sample
    observation = meta["observation_timestamp_ns"]
    live = meta["source"] == "live_ros"
    started = time.monotonic_ns()
    p, q = nominal_world_right_tool(arrays["joint_names"], arrays["joint_positions"], URDF)
    joints = dict(zip(arrays["joint_names"], arrays["joint_positions"]))
    positions = [float(joints[k]) for k in NAMES]
    fk_done = time.monotonic_ns()
    request = {"path": str(path), "instruction": args.instruction, "seed": args.seed,
               "fk_pose": [p.tolist(), q.tolist()]}
    worker.stdin.write(json.dumps(request, allow_nan=False) + "\n")
    worker.stdin.flush()
    prediction = read_worker(worker, args.out / "policy.log")
    ready = time.monotonic_ns()
    all_targets = prediction["targets"]
    poses = []
    for i, target in enumerate(all_targets):
        # Preempt the old replay block when the next observation is sampled.
        end = slot_time_ns(observation, i + 1)
        if next_obs is not None and end > next_obs:
            break
        poses.append({"position": target["position_m"], "quaternion_xyzw": target["quaternion_xyzw"]})
    feedback_rx = (meta["feedback_callback_monotonic_ns"] if live else meta["feedback_bag_timestamps_ns"])["joints"]
    received = receiver.run({
        "mode": "live" if live else "virtual", "observation_id": name,
        "observation_time_ns": observation, "ready_ns": ready if live else observation,
        "ik_lead_ns": round(args.ik_lead_ms * 1_000_000) if live else 0,
        "feedback_source_ns": meta["joint_header_stamp_ns"], "feedback_received_ns": feedback_rx,
        "snapshot": {"joint_states": {"positions": positions}}, "targets": poses,
    })
    finished = time.monotonic_ns()
    return {
        "id": name, "snapshot": path.name, "observation": meta, "seed_joints_l_then_r": positions,
        "prediction": prediction, "receiver": received, "superseded_tail": len(all_targets) - len(poses),
        "timing": {"capture_s": capture_s, "snapshot_write_s": saved_s, "fk_s": (fk_done-started)/NS,
                   "worker_roundtrip_s": (ready-fk_done)/NS, "ready_monotonic_ns": ready,
                   "receiver_s": (finished-ready)/NS, "finish_monotonic_ns": finished,
                   "live_observation_to_ready_s": (ready-observation)/NS if live else None,
                   "live_observation_to_finish_s": (finished-observation)/NS if live else None},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="+", default=[], help="increasing frame indices in the orange bag")
    parser.add_argument("--live", type=int, default=0, help="number of fresh ROS captures")
    parser.add_argument("--timeout", type=float, default=12, help="seconds per live capture")
    parser.add_argument("--out", required=True, type=Path, help="new directory for snapshots, log and run.json")
    parser.add_argument("--bag", type=Path, default=BAG)
    parser.add_argument("--video")
    parser.add_argument("--binary", type=Path, default=IK_BINARY)
    parser.add_argument("--config", type=Path, default=IK_CONFIG)
    parser.add_argument("--ik-lead-ms", type=float, default=1000,
                        help="compute live IK this far before its original 15 Hz deadline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction", default="pick the orange and place it on the black mouse pad")
    args = parser.parse_args()
    if args.frames != sorted(set(args.frames)) or args.live < 0 or not (args.frames or args.live):
        parser.error("select increasing recording frames and/or a positive live count")
    if args.frames and args.bag is None:
        parser.error("--bag or TIANJI_BAG is required for recording frames")
    if args.ik_lead_ms < 0:
        parser.error("--ik-lead-ms must be nonnegative")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "instruction": args.instruction,
              "seed_per_observation": args.seed, "live_ik_lead_ms": args.ik_lead_ms, "blocks": []}

    def capture(name, read):
        collected_utc = datetime.now(timezone.utc).isoformat()
        before = time.monotonic_ns()
        arrays, meta = read()
        meta["capture_started_utc"] = collected_utc
        meta["capture_started_monotonic_ns"] = before
        after = time.monotonic_ns()
        path = save_snapshot(args.out / (name + ".npz"), arrays, meta)
        return path, arrays, meta, (after-before)/NS, (time.monotonic_ns()-after)/NS

    def run(name, sample, next_obs=None):
        block = process_block(worker, receiver, args, name, sample, next_obs)
        report["blocks"].append(block)
        (args.out / "run.json").write_text(json.dumps(report, allow_nan=False) + "\n")
        n = sum(row.get("host") is not None for row in block["receiver"]["samples"])
        print(f"{name}: {n} mock right-arm messages", flush=True)
        if block["receiver"]["failure"]:
            raise RuntimeError(block["receiver"]["failure"])

    with model_worker(args.out) as (worker, loaded):
        report["policy_worker"] = loaded
        receiver = Receiver(str(args.binary), str(args.config))
        try:
            recordings = []
            for frame in args.frames:
                name = f"bag-{frame}"
                recordings.append((name, capture(name, lambda: recording(args.bag, frame, args.video))))
            for i, (name, sample) in enumerate(recordings):
                following = recordings[i + 1][1][2]["observation_timestamp_ns"] if i + 1 < len(recordings) else None
                run(name, sample, following)
            for i in range(args.live):
                name = f"live-{i + 1}"
                run(name, capture(name, lambda: read_live(args.timeout)))
        finally:
            receiver.close()


if __name__ == "__main__":
    main()
