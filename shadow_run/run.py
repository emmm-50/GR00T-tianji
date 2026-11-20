"""Finite GR00T runs with recorded or live read-only snapshots and an in-memory Host."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from threading import Condition

import pinocchio as pin

from tianji_adapter.capture_ros import live_capture, recording, save_snapshot
from tianji_paths import BAG, IK_BINARY, IK_CONFIG, NAMES, ROOT, URDF
from tianji_adapter.geometry import nominal_world_right_tool
from .prediction import add_tool_sensitivity
from .receiver import Receiver, slot_time_ns

NS = 1_000_000_000


def read_worker(worker, log, spin=None):
    if spin is not None:
        while not select.select([worker.stdout], [], [], 0)[0]:
            spin()
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


def predict_block(worker, args, fk_model, name, sample, next_obs=None, *, spin=None):
    path, arrays, meta, capture_s, saved_s = sample
    observation = meta["observation_timestamp_ns"]
    live = meta["source"] == "live_ros"
    started = time.monotonic_ns()
    p, q = nominal_world_right_tool(arrays["joint_names"], arrays["joint_positions"], URDF, model=fk_model)
    joints = dict(zip(arrays["joint_names"], arrays["joint_positions"]))
    positions = [float(joints[k]) for k in NAMES]
    fk_done = time.monotonic_ns()
    request = {"path": str(path), "instruction": args.instruction, "seed": args.seed,
               "fk_pose": [p.tolist(), q.tolist()], "defer_tool_sensitivity": True}
    worker.stdin.write(json.dumps(request, allow_nan=False) + "\n")
    worker.stdin.flush()
    prediction = read_worker(worker, args.out / "policy.log", spin)
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
    request = {
        "mode": "live" if live else "virtual", "observation_id": name,
        "observation_time_ns": observation, "ready_ns": ready if live else observation,
        "ik_lead_ns": round(args.ik_lead_ms * 1_000_000) if live else 0,
        "feedback_source_ns": meta["joint_header_stamp_ns"], "feedback_received_ns": feedback_rx,
        "snapshot": {"joint_states": {"positions": positions}}, "targets": poses,
    }
    block = {
        "id": name, "snapshot": path.name, "observation": meta, "seed_joints_l_then_r": positions,
        "prediction": prediction, "superseded_tail": len(all_targets) - len(poses),
        "timing": {"capture_s": capture_s, "snapshot_write_s": saved_s, "fk_s": (fk_done-started)/NS,
                   "worker_roundtrip_s": (ready-fk_done)/NS, "ready_monotonic_ns": ready,
                   "live_observation_to_ready_s": (ready-observation)/NS if live else None},
    }
    return block, request


def receive_block(receiver, prediction, next_request=None):
    block, request = prediction
    started = time.monotonic_ns()
    received = receiver.run(request, next_request=next_request)
    finished = time.monotonic_ns()
    block["receiver"] = received
    if "handoff_to_observation_id" in received:
        block["superseded_tail"] += len(request["targets"]) - len(received["samples"])
    block["timing"].update({
        "receiver_start_monotonic_ns": started, "receiver_s": (finished - started) / NS,
        "finish_monotonic_ns": finished,
        "live_observation_to_finish_s": (finished - request["observation_time_ns"]) / NS
        if request["mode"] == "live" else None,
    })
    count = sum(row.get("host") is not None for row in received["samples"])
    print(f"{block['id']}: {count} mock right-arm messages", flush=True)
    if received["failure"]:
        raise RuntimeError(received["failure"])


def live_blocks(receiver, worker, args, fk_model, capture, blocks):
    # Latest completed prediction only. The producer owns ROS and model I/O;
    # the consumer alone claims targets, advances IK and publishes to the mock.
    changed = Condition()
    latest = None
    current = None
    finished = False

    def consume():
        nonlocal current
        while True:
            with changed:
                changed.wait_for(lambda: latest is not current or finished)
                if latest is None or latest is current:
                    return
                current = latest

            def next_request():
                with changed:
                    if latest is None:
                        raise RuntimeError("live input stopped")
                    return latest[1] if latest is not current else None

            receive_block(receiver, current, next_request)

    with live_capture() as node, ThreadPoolExecutor(max_workers=1) as pool:
        previous_switch = sys.getswitchinterval()
        sys.setswitchinterval(min(previous_switch, 0.001))
        owner = pool.submit(consume)
        try:
            for i in range(args.live):
                if owner.done():
                    owner.result()
                name = f"live-{i + 1}"
                sample = capture(name, lambda: node.read(args.timeout))
                prediction = predict_block(worker, args, fk_model, name, sample, spin=node.spin_once)
                block, _ = prediction
                # A possibly superseded, never claimed block still has an audit record.
                block["receiver"] = {"mode": "live", "samples": [], "failure": None}
                blocks.append(block)
                with changed:
                    if latest is not None and latest is not current:
                        latest[0]["superseded_tail"] += len(latest[1]["targets"])
                    latest = prediction
                    changed.notify()
        except BaseException:
            # Preserve the producer error and stop claiming further slots.
            with changed:
                latest = None
            raise
        finally:
            with changed:
                finished = True
                changed.notify()
            try:
                if latest is not None:
                    owner.result()
            finally:
                sys.setswitchinterval(previous_switch)


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
    parser.add_argument("--ik-lead-ms", type=float, default=35,
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

    fk_model = pin.buildModelFromUrdf(str(URDF))
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
                prediction = predict_block(worker, args, fk_model, name, sample, following)
                report["blocks"].append(prediction[0])
                receive_block(receiver, prediction)
            if args.live:
                report["live_python_switch_interval_s"] = min(sys.getswitchinterval(), 0.001)
                live_blocks(receiver, worker, args, fk_model, capture, report["blocks"])
            for block in report["blocks"]:
                add_tool_sensitivity(block["prediction"])
        finally:
            receiver.close()
            (args.out / "run.json").write_text(json.dumps(report, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
