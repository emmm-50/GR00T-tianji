#!/usr/bin/env python3
"""Read-only ROS capture or extraction of a native snapshot from the orange bag."""

import argparse
from collections import deque
from contextlib import contextmanager
import json
from pathlib import Path
import time
import tempfile
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32MultiArray

from tianji_paths import BAG

TOPICS = {
    "eef": "/tj/info/eef_right",
    "joints": "/tj/joint_states",
    "gripper": "/tj/info/gripper_feedback_R",
}


def stamp(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def native(eef, joints, grip, images):
    p, q = eef.pose.position, eef.pose.orientation
    return {
        "right_eye": images["right_eye"],
        "right_wrist": images["right_wrist"],
        "eef_xyz": np.array([p.x, p.y, p.z], dtype=np.float64),
        "eef_xyzw": np.array([q.x, q.y, q.z, q.w], dtype=np.float64),
        "joint_names": np.array(joints.name),
        "joint_positions": np.array(joints.position, dtype=np.float64),
        "gripper_motor": np.array(grip.data, dtype=np.float64),
    }


def read_bag(path, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {
        t.name: get_message(t.type)
        for t in reader.get_all_topics_and_types()
        if t.name in topics
    }
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    while reader.has_next():
        topic, raw, ts = reader.read_next()
        yield topic, ts, deserialize_message(raw, types[topic])


def split_quad(frame):
    h, w = frame.shape[:2]
    return (
        cv2.cvtColor(frame[: h // 2, w // 2 :], cv2.COLOR_BGR2RGB),
        cv2.cvtColor(frame[h // 2 :, w // 2 :], cv2.COLOR_BGR2RGB),
    )


def recording(bag, step, selected_video=None):
    if bag is None:
        raise ValueError("recording requires --bag or TIANJI_BAG")
    bag = Path(bag)
    videos = sorted(p.parent for p in (bag / "video").glob("*/metadata.yaml"))
    if selected_video is not None:
        video_path = bag / "video" / selected_video
        if video_path not in videos:
            raise ValueError(f"video not found: {video_path}; choices: {videos}")
    elif len(videos) == 1:
        video_path = videos[0]
    else:
        raise ValueError(f"expected one video bag; specify --video from: {videos}")
    if step < 1:
        raise ValueError("frame requires one second of prior video")
    times, headers = [], []
    # Bag payload is H.264 Annex B, not an independently decodable JPEG.
    # Preserve packet order and decode the video stream from its beginning.
    with tempfile.NamedTemporaryFile(suffix=".h264") as raw:
        for i, (_, t, msg) in enumerate(
            read_bag(video_path, ["/recorder/quad_tile/compressed_undistorted"])
        ):
            raw.write(bytes(msg.data))
            times.append(t)
            headers.append(stamp(msg))
            if i >= step:
                break
        if len(times) <= step:
            raise ValueError(f"video ended before frame {step}")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("video bag timestamps must increase")
        cadence = float(np.median(np.diff(times)))
        if times[-1] - times[0] < 1_000_000_000 - cadence:
            raise ValueError(
                "one second of video history unavailable (within one sample period)"
            )
        target = times[-1] - 1_000_000_000
        old = min(range(step), key=lambda i: abs(times[i] - target))
        indices = [old, step]
        ts = [times[i] for i in indices]
        raw.flush()
        cap = cv2.VideoCapture(raw.name)
        frames = []
        for i in range(step + 1):
            ok, img = cap.read()
            if not ok:
                raise ValueError(f"quad H.264 decode ended at frame {i}")
            if i in indices:
                frames.append(img)
        cap.release()
    images = {"right_eye": [], "right_wrist": []}
    for frame in frames:
        eye, wrist = split_quad(frame)
        images["right_eye"].append(eye)
        images["right_wrist"].append(wrist)
    images = {k: np.stack(v) for k, v in images.items()}
    values = {}
    for topic, t, msg in read_bag(bag / "data", TOPICS.values()):
        if t > ts[-1]:
            break
        values[topic] = (t, msg)
    if set(values) != set(TOPICS.values()):
        raise ValueError("missing prior feedback")
    arr = native(*(values[TOPICS[k]][1] for k in ("eef", "joints", "gripper")), images)
    eef, joint = values[TOPICS["eef"]][1], values[TOPICS["joints"]][1]
    meta = {
        "source": "recording",
        "bag": str(bag),
        "video_bag": str(video_path),
        "observation_timestamp_ns": int(ts[-1]),
        "observation_clock": "video bag timestamp",
        "image_history_seconds": (ts[-1] - ts[0]) / 1e9,
        "video_frames": indices,
        "video_bag_timestamps_ns": ts,
        "video_header_stamp_ns": [headers[i] for i in indices],
        "feedback_bag_timestamps_ns": {k: int(values[v][0]) for k, v in TOPICS.items()},
        "ages_ms": {
            "camera": 0.0,
            **{k: (ts[-1] - values[v][0]) / 1e6 for k, v in TOPICS.items()},
        },
        "eef_frame_id": eef.header.frame_id,
        "eef_header_stamp_ns": stamp(eef),
        "joint_header_stamp_ns": stamp(joint),
        "synchronization": "previous bag timestamp; physical exposure and sensor clocks are unverified",
        "camera_mapping": "quad bag H264: top-right right_eye, bottom-right right_wrist",
    }
    return arr, meta


class Capture(Node):
    def __init__(self, quad_topic, hz=15):
        super().__init__("gr00t_observation_capture")
        self.quad_topic = quad_topic
        self.images = deque()
        self.states = {}
        self.period = 1 / hz
        self.latest = None
        self.last_selection_ns = 0
        self.create_timer(self.period, self.select, clock=Clock(clock_type=ClockType.STEADY_TIME))
        for key, typ in (
            ("eef", PoseStamped),
            ("joints", JointState),
            ("gripper", Float32MultiArray),
        ):
            self.create_subscription(
                typ,
                TOPICS[key],
                lambda m, k=key: self.state(k, m),
                qos_profile_sensor_data,
            )
        self.create_subscription(
            CompressedImage, quad_topic, self.image, qos_profile_sensor_data
        )

    def state(self, key, msg):
        self.states[key] = (time.monotonic_ns(), msg)

    def image(self, msg):
        now = time.monotonic_ns()
        self.images.append((now, msg))
        while self.images and now - self.images[0][0] > 5_000_000_000:
            self.images.popleft()

    def select(self):
        """Retain only the latest complete sample. Decode its JPEGs on consumption."""
        now = time.monotonic_ns()
        since = self.last_selection_ns
        if (len(self.states) != len(TOPICS) or not self.images or self.images[-1][0] < since
                or any(rx < since for rx, _ in self.states.values())):
            return
        current = self.images[-1]
        old = next((image for image in reversed(self.images) if image[0] <= current[0] - 1_000_000_000), None)
        if old is not None:
            self.latest = now, (old, current), self.states.copy()
            self.last_selection_ns = now

    def snapshot(self, selected):
        now, chosen, states = selected
        eef, joints, grip = (states[k][1] for k in ("eef", "joints", "gripper"))
        views = []
        for _, msg in chosen:
            bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("invalid quad JPEG image")
            views.append(split_quad(bgr))
        eyes, wrists = zip(*views)
        arrays = native(eef, joints, grip,
                        {"right_eye": np.stack(eyes), "right_wrist": np.stack(wrists)})
        meta = {
            "source": "live_ros",
            "eef_frame_id": eef.header.frame_id,
            "observation_timestamp_ns": now,
            "observation_clock": "local monotonic receive time; exposure time unverified",
            "image_history_seconds": (chosen[1][0] - chosen[0][0]) / 1e9,
            "camera_receive_monotonic_ns": [int(x[0]) for x in chosen],
            "camera_header_stamp_ns": [stamp(x[1]) for x in chosen],
            "quad_topic": self.quad_topic,
            "feedback_callback_monotonic_ns": {k: v[0] for k, v in states.items()},
            "ages_ms": {
                "camera": (now - chosen[1][0]) / 1e6,
                **{k: (now - v[0]) / 1e6 for k, v in states.items()},
            },
            "eef_header_stamp_ns": stamp(eef),
            "joint_header_stamp_ns": stamp(joints),
        }
        return arrays, meta

    def spin_once(self, timeout=0.02):
        self.executor.spin_once(timeout_sec=timeout)

    def read(self, timeout):
        started = time.monotonic_ns()
        deadline = started + int(timeout * 1e9)
        # Choose the nearer sampling tick. Past the midpoint, wait for the
        # next real selection; keep its actual timestamp in either case.
        earliest = started - self.period * 1e9 / 2
        while time.monotonic_ns() < deadline:
            if self.latest is not None and self.latest[0] >= earliest:
                selected, self.latest = self.latest, None
                return self.snapshot(selected)
            self.spin_once()
        raise RuntimeError(f"timed out; states={list(self.states)}, quad_frames={len(self.images)}")


@contextmanager
def live_capture(quad_topic="/quad_tile/jpeg/compressed"):
    rclpy.init()
    node = Capture(quad_topic)
    executor = rclpy.get_global_executor()
    executor.add_node(node)
    try:
        yield node
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


def read_live(timeout, quad_topic="/quad_tile/jpeg/compressed"):
    with live_capture(quad_topic) as node:
        return node.read(timeout)


def save_snapshot(path, arrays, meta, *, compressed=False):
    """Save a snapshot and JSON sidecar, optionally compressing the arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (np.savez_compressed if compressed else np.savez)(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["recording", "ros"], required=True)
    p.add_argument("--frame", type=int, default=330)
    p.add_argument(
        "--bag", type=Path, default=BAG, help="recording root with data/ and video/ (or set TIANJI_BAG)"
    )
    p.add_argument(
        "--video",
        help="video bag directory name under BAG/video (required if multiple)",
    )
    p.add_argument("--quad-topic", default="/quad_tile/jpeg/compressed")
    p.add_argument("--timeout", type=float, default=10)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.suffix != ".npz":
        p.error("--output requires a .npz path for the snapshot and its JSON sidecar")
    if a.source == "recording":
        result = recording(a.bag, a.frame, a.video)
    else:
        result = read_live(a.timeout, a.quad_topic)
    arrays, meta = result
    save_snapshot(a.output, arrays, meta, compressed=True)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
