"""Persistent, read-only GR00T shadow inference from a single Tianji snapshot.

The virtual DROID pose identifies its base with Tianji world and its tool with
right_tool. This is an explicit hypothesis, not a calibrated correspondence.
No output here is a robot command. Joint and gripper inputs are named proxies.
"""
import os
import sys


def _protocol_stream():
    sys.stdout.flush()
    stream = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)
    return stream


# Redirect before importing libraries; native writes to fd 1 also go to stderr.
_protocol = _protocol_stream() if __name__ == '__main__' else None

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tianji_adapter.geometry import (
    anchored_ik_targets, encode_droid_pose,
)
from tianji_adapter.observations import load_snapshot
from tianji_adapter.policy_probe import MODEL, policy_observation, summarize


def _motion(targets, position, quaternion):
    positions = np.vstack([position, *[t['position_m'] for t in targets]])
    rotations = Rotation.from_quat(np.vstack([quaternion, *[t['quaternion_xyzw'] for t in targets]]))
    distance = np.linalg.norm(positions[1:] - positions[0], axis=1) * 1000
    return {
        'translation_from_anchor_mm': {'first': float(distance[0]), 'last': float(distance[-1]), 'max': float(distance.max())},
        'max_adjacent_translation_mm_including_anchor': float((np.linalg.norm(np.diff(positions, axis=0), axis=1) * 1000).max()),
        'max_adjacent_rotation_deg_including_anchor': float(np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude()).max()),
        'max_rotation_from_anchor_deg': float(np.rad2deg((rotations[0].inv() * rotations[1:]).magnitude()).max()),
    }


def remap_targets(result, right_tool_from_model_tool):
    """Reinterpret the *same* prediction with another rigid tool hypothesis.

    The virtual input state and network output stay fixed. Changes in these
    candidates are geometric sensitivity, not a new policy evaluation.
    """
    anchor = result['anchor']
    targets = anchored_ik_targets(
        result['state']['eef_9d'], anchor['position_m'], anchor['quaternion_xyzw'],
        result['actions']['eef_9d'], result['actions']['gripper_position'],
        observation_timestamp_ns=anchor['observation_timestamp_ns'],
        observation_clock=anchor['observation_clock'],
        right_tool_from_model_tool=right_tool_from_model_tool,
    )
    return {'right_tool_from_model_tool': np.asarray(right_tool_from_model_tool).tolist(),
            'targets': targets,
            'motion': _motion(targets, anchor['position_m'], anchor['quaternion_xyzw'])}


def compare_tool_hypotheses(result, hypotheses):
    """Map named 4x4 tool transforms and compare each with the baseline targets."""
    baseline = result['targets']
    bp = np.asarray([t['position_m'] for t in baseline])
    br = Rotation.from_quat([t['quaternion_xyzw'] for t in baseline])
    out = {}
    for name, transform in hypotheses.items():
        mapped = remap_targets(result, transform)
        p = np.asarray([t['position_m'] for t in mapped['targets']])
        r = Rotation.from_quat([t['quaternion_xyzw'] for t in mapped['targets']])
        mapped['difference_from_baseline'] = {
            'max_position_mm': float((1000 * np.linalg.norm(p - bp, axis=1)).max()),
            'max_orientation_deg': float(np.rad2deg((br.inv() * r).magnitude()).max()),
        }
        out[name] = mapped
    return out


def add_tool_sensitivity(result):
    """Add diagnostic tool hypotheses to an existing prediction, using CPU only."""
    started = time.perf_counter()
    offset = np.eye(4)
    offset[2, 3] = 0.05
    rotation = np.eye(4)
    rotation[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    hypotheses = compare_tool_hypotheses(result, {'z_plus_50mm': offset, 'z_plus_90deg': rotation})
    result['tool_sensitivity'] = {
        name: {'right_tool_from_model_tool': item['right_tool_from_model_tool'],
               'difference_from_baseline': item['difference_from_baseline']}
        for name, item in hypotheses.items()
    }
    result['timing_s']['tool_sensitivity'] = time.perf_counter() - started
    return result


class PredictionPolicy:
    def __init__(self, policy, modality, quantiles):
        self.policy = policy
        self.modality = modality
        self.quantiles = quantiles

    def predict(self, snapshot, meta, instruction, *, fk_pose, right_tool_from_model_tool=None, seed=None):
        """Predict a chunk from snapshot arrays and metadata, without robot access.

        fk_pose is (position_m, quaternion_xyzw), computed from this snapshot.
        Optional seed controls torch's RNG. Default does not reset it each call.
        Default tool transform is identity, an unvalidated sensitivity baseline.
        """
        import torch
        start = time.perf_counter()
        names = list(snapshot['joint_names'])
        positions = np.asarray(snapshot['joint_positions'])
        p = np.asarray(fk_pose[0], dtype=float)
        q = np.asarray(fk_pose[1], dtype=float)
        state = {
            'eef_9d': encode_droid_pose(p, q),
            'joint_position': np.asarray([positions[names.index(f'Joint{i}_R')] for i in range(1, 8)]),
            'gripper_position': np.clip(np.asarray(snapshot['gripper_motor'])[:1] / 1.6, 0, 1),
        }
        state = {key: np.asarray(value, dtype=np.float32) for key, value in state.items()}
        obs = policy_observation(snapshot, state, instruction, self.modality)
        prepared = time.perf_counter()
        if seed is not None:
            torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_start = time.perf_counter()
        with torch.inference_mode():
            actions, _ = self.policy.get_action(obs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inferred = time.perf_counter()
        # Deliberately do not expose or use the predicted DROID joint actions.
        chunk = {k: np.asarray(actions[k])[0] for k in ('eef_9d', 'gripper_position')}
        result = {
            'scope': 'shadow_only_no_robot_commands',
            'assumptions': [
                'virtual DROID base is Tianji nominal world; virtual DROID tool is nominal right_tool',
                'right joints by index are Franka-state proxies with no established correspondence',
                'clip(gripper_motor[0]/1.6, 0, 1) is a proxy, not measured DROID closure',
                'right eye and right wrist substitute for DROID cameras',
                'base and physical tool correspondences are unknown; targets are conditional candidates',
            ],
            'anchor': {'position_m': p.tolist(), 'quaternion_xyzw': q.tolist(),
                       'observation_timestamp_ns': int(meta['observation_timestamp_ns']),
                       'observation_clock': meta['observation_clock'],
                       'joint_header_stamp_ns': meta.get('joint_header_stamp_ns'),
                       'source': 'caller-provided nominal FK paired with same snapshot named joints'},
            'state': {k: v.tolist() for k, v in state.items()},
            'actions': {k: v.tolist() for k, v in chunk.items()},
            'diagnostics': summarize(state, chunk, self.quantiles),
        }
        # Message EEF comparison is informative only: its frame and tool convention
        # are not independently established by this single snapshot.
        if 'eef_xyz' in snapshot and 'eef_xyzw' in snapshot:
            result['diagnostics']['snapshot_eef_vs_nominal_fk'] = {
                'position_mm': float(1000 * np.linalg.norm(np.asarray(snapshot['eef_xyz']) - p)),
                'orientation_deg': float(np.rad2deg((Rotation.from_quat(q).inv() * Rotation.from_quat(snapshot['eef_xyzw'])).magnitude())),
                'interpretation': 'numeric comparison; frame/tool correspondence must be verified',
            }
        transform = np.eye(4) if right_tool_from_model_tool is None else right_tool_from_model_tool
        mapped = remap_targets(result, transform)
        result['targets'] = mapped['targets']
        result['right_tool_from_model_tool'] = mapped['right_tool_from_model_tool']
        result['diagnostics']['candidate_motion'] = mapped['motion']
        end = time.perf_counter()
        result['timing_s'] = {'prepare': prepared - start, 'inference': inferred - inference_start,
                              'postprocess': end - inferred, 'total': end - start}
        return result

def load_policy(*, model_path=MODEL, device='cuda', denoising_steps=4, compile_action_head=False):
    """Load the original policy once; retain the returned object for all chunks."""
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    tag = EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
    model_path = Path(model_path)
    policy = Gr00tPolicy(embodiment_tag=tag, model_path=str(model_path), device=device)
    policy.model.action_head.num_inference_timesteps = denoising_steps
    if compile_action_head:
        from gr00t_compile import compile_action_head as enable_compile
        enable_compile(policy)
    quantiles = json.loads((model_path / 'statistics.json').read_text())[tag.value]['state']
    return PredictionPolicy(policy, policy.get_modality_config(), quantiles)


def main():
    """JSONL worker: one load notification, followed by one result per request."""
    output = _protocol if _protocol is not None else _protocol_stream()
    started = time.perf_counter()
    compiled = os.environ.get('GR00T_TORCH_COMPILE') == '1'
    policy = load_policy(compile_action_head=compiled)
    import torch
    print(json.dumps({'loaded': True, 'load_s': time.perf_counter() - started,
                      'compile_action_head': compiled, 'torch_threads': torch.get_num_threads(),
                      'torch_interop_threads': torch.get_num_interop_threads()}), file=output, flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        snapshot, meta = load_snapshot(request['path'])
        result = policy.predict(snapshot, meta, request['instruction'],
                                fk_pose=request['fk_pose'], seed=request.get('seed'))
        if not request.get('defer_tool_sensitivity', False):
            add_tool_sensitivity(result)
        print(json.dumps(result, allow_nan=False), file=output, flush=True)


if __name__ == '__main__':
    main()
