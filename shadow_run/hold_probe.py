#!/usr/bin/env python3
"""Offline hold of each received live target; no ROS, Host, GPU or robot I/O.

Run from the project root with the native IK dependencies on LD_LIBRARY_PATH:
  /usr/bin/python3 -B -m shadow_run.hold_probe /path/to/private-run.json.gz --output /tmp/shadow_hold.json
"""
import argparse
import gzip
import json
from pathlib import Path
import time
import sys

from tianji_paths import EEF_CONTROL, IK_BINARY, IK_CONFIG

sys.path.insert(0, str(EEF_CONTROL / 'host_mock'))
from ik_adapter import IKAdapter, NativeSolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=IK_BINARY)
    parser.add_argument('--config', type=Path, default=IK_CONFIG)
    args = parser.parse_args()
    opener = gzip.open if args.run.suffix == '.gz' else open
    with opener(args.run, 'rt', encoding='utf-8') as stream:
        run = json.load(stream)
    report = {
        'source_run': str(args.run),
        'method': 'Each received live target held from its own same-observation 14-joint seed; nominal 15 Hz intervals, 2 ms integration. Offline model only.',
        'binary': str(args.binary), 'config': str(args.config), 'blocks': [],
    }
    solver = NativeSolver(str(args.binary), str(args.config))
    try:
        adapter = IKAdapter(solver)
        for block in run['blocks']:
            if not block['id'].startswith('live-'):
                continue
            received = [row for row in block['receiver']['samples'] if row.get('host') is not None]
            for row in received:
                index = row['index']
                pose = block['prediction']['targets'][index]
                target = {'position': pose['position_m'], 'quaternion_xyzw': pose['quaternion_xyzw']}
                adapter.initialize({'joint_states': {'positions': block['seed_joints_l_then_r']}},
                                   feedback_source_ns=None, feedback_received_ns=None, plan_start_ns=0)
                record = {'id': block['id'], 'target_index': index,
                          'recorded_single_interval': {
                              'position_error_m': row['ik']['position_error_m'],
                              'rotation_error_rad': row['ik']['rotation_error_rad']},
                          'checkpoints': []}
                started = time.monotonic()
                for interval in range(1, 11):
                    result = adapter.step(target, observation_id=block['id'], observation_time_ns=0,
                                          target_time_ns=round(interval * 1_000_000_000 / 15))
                    if interval in (1, 3, 5, 10):
                        record['checkpoints'].append({
                            'interval': interval, 'integrated_s': result['plan_time_ns'] / 1e9,
                            'position_error_m': result['ik']['position_error_m'],
                            'rotation_error_rad': result['ik']['rotation_error_rad'],
                        })
                record['wall_s'] = time.monotonic() - started
                report['blocks'].append(record)
    finally:
        solver.close()
        solver.process.stdout.close()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
