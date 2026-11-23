#!/usr/bin/env python3
"""Serial offline IK -> installed Host in-memory action receiver; no ROS I/O."""
import asyncio
import sys
import time

from tianji_paths import EEF_CONTROL

MOCK = EEF_CONTROL / 'host_mock'
sys.path.insert(0, str(MOCK))
import host_mock
from ik_adapter import DT_NS, IKAdapter, NativeSolver

PERIOD_NUM = 1_000_000_000
HZ = 15
MIN_RECEIVE_NS = 60_000_000  # 15 Hz with modest scheduling jitter


def slot_time_ns(start, index):
    return start + round(index * PERIOD_NUM / HZ)


def next_slot_ns(origin, earliest):
    index = max(1, (earliest - origin) * HZ // PERIOD_NUM)
    end = slot_time_ns(origin, index)
    return end if end >= earliest else slot_time_ns(origin, index + 1)


class Receiver:
    def __init__(self, binary, config):
        self.host = host_mock.load_host()
        self.solver = NativeSolver(binary, config)
        self.host_loop = asyncio.new_event_loop()
        self.adapter = IKAdapter(self.solver)
        self.last_receive_ns = None

    def close(self):
        self.host_loop.close()
        self.solver.close()

    def _send(self, action):
        received = []
        capture = host_mock.run_action(action, host=self.host, loop=self.host_loop,
                                       on_right_publish=received.append)
        if len(received) != 1:
            raise ValueError('expected one mock right joint publication')
        return capture, received[0]

    def run(self, request, *, next_request=None):
        mode = request['mode']
        if mode not in ('virtual', 'live'):
            raise ValueError('mode must be virtual or live')
        start = request['observation_time_ns']
        ready = request.get('ready_ns', start)
        lead = request.get('ik_lead_ns', 0)
        if not isinstance(lead, int) or lead < 0:
            raise ValueError('ik_lead_ns must be a nonnegative integer')
        snapshot = request['snapshot']
        q = list(snapshot['joint_states']['positions'])
        if len(q) != 14:
            raise ValueError('snapshot must contain 14 joints in left/right order')
        output = {'observation_id': request['observation_id'], 'mode': mode,
                  'ready_ns': ready, 'ik_lead_ns': lead,
                  'samples': [], 'failure': None}
        feedback = dict(feedback_source_ns=request.get('feedback_source_ns'),
                        feedback_received_ns=request.get('feedback_received_ns'))
        try:
            reseed = True
            previous_right = q[7:]
            for i, target in enumerate(request['targets']):
                sample = slot_time_ns(start, i)
                end = slot_time_ns(start, i + 1)
                # Decide before claiming a slot; an in-flight slot is never preempted.
                if mode == 'live' and next_request is not None:
                    pending = next_request()
                    if pending is not None:
                        new_start = pending['observation_time_ns']
                        new_lead = pending.get('ik_lead_ns', 0)
                        now = max(time.monotonic_ns(), pending.get('ready_ns', new_start))
                        earliest = now + max(new_lead, DT_NS)
                        if self.last_receive_ns is not None:
                            earliest = max(earliest, self.last_receive_ns + MIN_RECEIVE_NS)
                        new_end = next_slot_ns(new_start, earliest)
                        horizon = slot_time_ns(new_start, len(pending['targets']))
                        if new_end <= horizon and end > new_end - max(MIN_RECEIVE_NS, new_lead):
                            output['handoff_to_observation_id'] = pending['observation_id']
                            output['handoff_end_ns'] = new_end
                            break
                row = {'index': i, 'sample_ns': sample, 'end_ns': end,
                       'plan_ns': None, 'mock_receive_ns': None, 'skip_reason': None}
                output['samples'].append(row)
                if set(target) != {'position', 'quaternion_xyzw'}:
                    raise ValueError('targets must contain only position and quaternion_xyzw')
                now = max(ready, time.monotonic_ns()) if mode == 'live' else ready
                if mode == 'live' and end <= now:
                    row['skip_reason'] = 'expired_before_ik'
                    reseed = True
                    continue
                if mode == 'live' and end < now + lead:
                    row['skip_reason'] = 'insufficient_ik_lead'
                    reseed = True
                    continue
                if mode == 'live' and end - now < DT_NS:
                    row['skip_reason'] = 'insufficient_time_before_ik'
                    reseed = True
                    continue
                if mode == 'live' and self.last_receive_ns is not None and end < self.last_receive_ns + MIN_RECEIVE_NS:
                    row['skip_reason'] = 'minimum_receive_interval'
                    reseed = True
                    continue
                # Claim point: do not inspect the mailbox again until this slot finishes.
                row['claimed_monotonic_ns'] = time.monotonic_ns() if mode == 'live' else None
                if reseed:
                    reset = {'stamp': snapshot.get('stamp'), 'joint_states': {'positions': q[:7] + previous_right}}
                    t0 = time.monotonic_ns()
                    self.adapter.initialize(reset, plan_start_ns=max(sample, now) if mode == 'live' else start, **feedback)
                    row['reset_duration_ns'] = time.monotonic_ns() - t0
                    reseed = False
                    if mode == 'live' and end - time.monotonic_ns() < DT_NS:
                        row['skip_reason'] = 'insufficient_time_after_init'
                        reseed = True
                        continue
                t0 = time.monotonic_ns()
                result = self.adapter.step(target, observation_id=request['observation_id'],
                                           observation_time_ns=start, target_time_ns=end,
                                           stop_requested=(lambda: time.monotonic_ns() >= end)
                                           if mode == 'live' else None)
                row['ik_duration_ns'] = time.monotonic_ns() - t0
                if result is None:
                    row['skip_reason'] = 'expired_during_ik'
                    reseed = True
                    continue
                row['plan_ns'] = result['plan_time_ns']
                row['ik_steps'] = result['steps']
                row['ik'] = result['ik']
                if mode == 'live' and time.monotonic_ns() >= end:
                    row['skip_reason'] = 'expired_after_ik'
                    # Do not continue from an unsent solver candidate.
                    reseed = True
                    continue
                if mode == 'live':
                    delay = end - time.monotonic_ns()
                    if delay > 0:
                        time.sleep(delay / 1e9)
                    if time.monotonic_ns() >= end + round(PERIOD_NUM / HZ):
                        row['skip_reason'] = 'expired_after_wait'
                        reseed = True
                        continue
                t0 = time.monotonic_ns()
                capture, actual = self._send(result['action'])
                row['host_start_monotonic_ns'] = t0
                row['host_duration_ns'] = time.monotonic_ns() - t0
                row['host'] = capture
                # Publication has already happened even if its acknowledgement fails.
                row['mock_receive_ns'] = actual if mode == 'live' else end
                row['mock_receive_monotonic_ns'] = actual
                if mode == 'live':
                    self.last_receive_ns = actual
                if capture['ack'] != [{'success': True, 'control_mode': 'joint'}] or capture['joint_right'] != [result['action']['jointcmd_right']]:
                    raise ValueError('mock Host rejected or changed right joint command')
                if capture['joint_left'] or capture['ik_requests'] or capture['gripper_left'] or capture['gripper_right']:
                    raise ValueError('unexpected mock Host output')
                right = capture['joint_right'][0]
                row['max_joint_delta_rad'] = max(abs(a-b) for a, b in zip(right, previous_right))
                previous_right = right
        except Exception as exc:
            output['failure'] = f'{type(exc).__name__}: {exc}'
        return output
