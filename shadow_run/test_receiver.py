#!/usr/bin/env python3
"""Recording FK -> prediction remapping -> native IK -> in-memory Host."""
import json
import os
import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tianji_paths import BAG, EEF_CONTROL, IK_BINARY, IK_CONFIG
sys.path.insert(0, str(EEF_CONTROL / 'native_ik' / 'tests'))
from shadow_run.prediction import remap_targets
from tianji_adapter.geometry import encode_droid_pose, pose_matrix
from shadow_run.receiver import Receiver


class Clock:
    def __init__(self): self.ns = 0
    def __call__(self): return self.ns
    def sleep(self, seconds): self.ns += round(seconds * 1e9)

class FakeSolver:
    def __init__(self, clock): self.clock, self.q, self.inits, self.slow = clock, [], [], False
    def __call__(self, req):
        if req['type'] == 'init':
            self.q = req['q'][:]
            self.inits.append(self.q[:])
            return {'ok': True}
        self.q[7] += 0.001
        if self.slow:
            self.clock.ns += self.slow if type(self.slow) is int else 70_000_000
            self.slow = False
        return {'ok': True, 'jointcmd_right': self.q[7:]}
    def close(self): pass

def req(q, targets, mode='virtual', start=0, ready=0):
    return {'mode': mode, 'observation_id': 1, 'observation_time_ns': start,
            'ready_ns': ready, 'feedback_source_ns': 1, 'feedback_received_ns': 2,
            'snapshot': {'stamp': 3, 'joint_states': {'positions': q}}, 'targets': targets}

class TestReceiver(unittest.TestCase):
    def test_recorded_fk_prediction_remap_ik_host(self):
        from recorded_fk import recorded_states, urdf_fk
        kwargs = {}
        if BAG is not None:
            kwargs['bag'] = str(BAG / 'data')
        if os.environ.get('TIANJI_FIXTURE_PROVENANCE'):
            kwargs['provenance'] = os.environ['TIANJI_FIXTURE_PROVENANCE']
        state, later = recorded_states(**kwargs)[:2]
        poses = [urdf_fk(s['q']) for s in (state, later)]
        tool = np.eye(4)
        tool[:3, :3] = Rotation.from_euler('xyz', [15, -20, 30], degrees=True).as_matrix()
        tool[:3, 3] = [0.03, -0.02, 0.05]
        labels = []
        for position, quat in poses:
            physical = pose_matrix(position, quat) @ tool
            labels.append(encode_droid_pose(physical[:3, 3], Rotation.from_matrix(physical[:3, :3]).as_quat()).tolist())
        prediction = {'anchor': {'position_m': poses[0][0], 'quaternion_xyzw': poses[0][1],
                                  'observation_timestamp_ns': state['video_ns'], 'observation_clock': 'bag'},
                      'state': {'eef_9d': labels[0]},
                      'actions': {'eef_9d': labels, 'gripper_position': [[0], [0]]}}
        mapped = remap_targets(prediction, tool)['targets']
        errors = []
        for actual, (p, q) in zip(mapped, poses):
            pos = np.linalg.norm(np.asarray(actual['position_m']) - p)
            rot = (Rotation.from_quat(actual['quaternion_xyzw']).inv() * Rotation.from_quat(q)).magnitude()
            errors.append([float(pos), float(rot)])
            self.assertLess(pos, 1e-10)
            self.assertLess(rot, 1e-10)
        targets = [{'position': t['position_m'], 'quaternion_xyzw': t['quaternion_xyzw']} for t in mapped]
        receiver = Receiver(str(IK_BINARY), str(IK_CONFIG))
        try:
            result = receiver.run(req(state['q'], targets, start=state['video_ns']))
        finally:
            receiver.close()
        self.assertIsNone(result['failure'], result)
        a, b = result['samples']
        self.assertEqual(a['plan_ns'], state['video_ns'] + 66_000_000)
        self.assertEqual(b['plan_ns'], state['video_ns'] + 132_000_000)
        self.assertLess(a['max_joint_delta_rad'], 1e-5)
        self.assertGreater(b['max_joint_delta_rad'], 1e-8)
        for row in (a, b):
            self.assertEqual(row['host']['ack'], [{'success': True, 'control_mode': 'joint'}])
            self.assertEqual(row['host']['joint_right'], [row['ik']['jointcmd_right']])
            self.assertFalse(row['host']['joint_left'])
            self.assertFalse(row['host']['gripper_right'])
            self.assertLess(row['ik']['position_error_m'], 1e-6)
            self.assertLess(row['ik']['rotation_error_rad'], 1e-7)
        print(json.dumps({'frames': [state['frame'], later['frame']], 'remap_error_m_rad': errors,
                          'hold': {'max_joint_delta_rad': a['max_joint_delta_rad'], 'ik': a['ik']},
                          'move': {'max_joint_delta_rad': b['max_joint_delta_rad'], 'ik': b['ik']}}, allow_nan=False))

    def fake_receiver(self, overshoot=0):
        clock = Clock()
        solver = FakeSolver(clock)
        def sleep(seconds):
            clock.sleep(seconds)
            clock.ns += overshoot
        for p in (patch('shadow_run.receiver.NativeSolver', return_value=solver),
                  patch('shadow_run.receiver.host_mock.load_host', return_value=None),
                  patch('shadow_run.receiver.time.monotonic_ns', clock), patch('shadow_run.receiver.time.sleep', sleep)):
            p.start()
            self.addCleanup(p.stop)
        receiver = Receiver('', '')
        def send(action):
            return ({'ack': [{'success': True, 'control_mode': 'joint'}],
                     'joint_right': [action['jointcmd_right']], 'joint_left': [],
                     'gripper_right': [], 'gripper_left': [], 'ik_requests': 0}, clock())
        receiver._send = send
        return receiver, clock, solver

    def test_sleep_overshoot_drops_stale_candidate(self):
        receiver, clock, solver = self.fake_receiver(overshoot=70_000_000)
        target = {'position': [0, 0, 0], 'quaternion_xyzw': [0, 0, 0, 1]}
        result = receiver.run(req([0.0]*14, [target], 'live'))
        self.assertIsNone(result['failure'], result)
        self.assertEqual(result['samples'][0]['skip_reason'], 'expired_after_wait')
        self.assertIsNone(result['samples'][0]['mock_receive_ns'])
        self.assertEqual(result['samples'][0]['end_ns'], 66_666_667)

    def test_small_sleep_jitter_is_reported_not_dropped(self):
        receiver, clock, solver = self.fake_receiver(overshoot=3_000_000)
        target = {'position': [0, 0, 0], 'quaternion_xyzw': [0, 0, 0, 1]}
        result = receiver.run(req([0.0]*14, [target], 'live'))
        self.assertIsNone(result['failure'], result)
        row = result['samples'][0]
        self.assertIsNone(row['skip_reason'])
        self.assertEqual(row['mock_receive_ns'] - row['end_ns'], 3_000_000)

    def test_lead_selects_original_future_slot_and_waits(self):
        receiver, clock, solver = self.fake_receiver()
        target = {'position': [0, 0, 0], 'quaternion_xyzw': [0, 0, 0, 1]}
        clock.ns = 70_000_000
        solver.slow = 800_000_000
        request = req([0.0]*14, [target]*18, 'live', ready=clock.ns)
        request['ik_lead_ns'] = 1_000_000_000
        result = receiver.run(request)
        self.assertIsNone(result['failure'], result)
        self.assertTrue(result['sparse_sampling'])
        self.assertEqual(result['samples'][0]['skip_reason'], 'expired_before_ik')
        self.assertTrue(all(r['skip_reason'] == 'insufficient_ik_lead' for r in result['samples'][1:16]))
        row = result['samples'][16]
        self.assertEqual(row['sample_ns'], 1_066_666_667)
        self.assertEqual(row['end_ns'], 1_133_333_333)
        self.assertEqual(row['mock_receive_ns'], row['end_ns'])
        self.assertEqual(row['ik_duration_ns'], 800_000_000)
        self.assertEqual(result['samples'][17]['skip_reason'], 'insufficient_ik_lead')
        self.assertEqual(len(solver.inits), 1)

    def test_live_expired_and_reseed_from_sent_and_cross_chunk(self):
        receiver, clock, solver = self.fake_receiver()
        target = {'position': [0, 0, 0], 'quaternion_xyzw': [0, 0, 0, 1]}
        clock.ns = 70_000_000
        result = receiver.run(req([0.0]*14, [target]*3, 'live', ready=clock.ns))
        self.assertIsNone(result['failure'], result)
        self.assertEqual(result['samples'][0]['skip_reason'], 'expired_before_ik')
        sent = [r for r in result['samples'] if r['mock_receive_ns'] is not None]
        self.assertEqual(len(sent), 2)
        self.assertGreaterEqual(sent[1]['mock_receive_ns'] - sent[0]['mock_receive_ns'], 60_000_000)
        newq = [0.0]*7 + [0.5]*7
        start = clock.ns
        next_result = receiver.run(req(newq, [target]*2, 'live', start=start, ready=start))
        self.assertIsNone(next_result['failure'], next_result)
        self.assertIn(newq, solver.inits)
        self.assertGreaterEqual(next_result['samples'][0]['mock_receive_ns'] - sent[-1]['mock_receive_ns'], 60_000_000)
        start = clock.ns
        solver.slow = True
        result = receiver.run(req(newq, [target]*3, 'live', start=start, ready=start))
        self.assertIsNone(result['failure'], result)
        self.assertEqual(result['samples'][0]['skip_reason'], 'expired_after_ik')
        self.assertEqual(solver.inits[-1], newq)

if __name__ == '__main__': unittest.main()
