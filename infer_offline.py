#!/usr/bin/env python3
"""Offline inference only. NPZ fields: video.<key> (N,H,W,3) RGB uint8,
state.<key> (N,D) float32 and scalar language. No robot connections."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))

import numpy as np
import torch
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.state_action.pose import EndEffectorPose
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.types import ActionRepresentation
from gr00t.data.utils import parse_modality_configs, parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy

TAG = EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT


def memory():
    with open('/proc/self/status') as f:
        rss = next(int(line.split()[1]) * 1024 for line in f if line.startswith('VmRSS:'))
    return {'cuda_allocated_bytes': torch.cuda.memory_allocated(),
            'cuda_reserved_bytes': torch.cuda.memory_reserved(),
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'process_rss_bytes': rss}


def sample_observation(episode, step, modality):
    configs = {k: v for k, v in modality.items() if k in ('video', 'state', 'language')}
    data = extract_step_data(episode, step, configs, TAG, allow_padding=True)
    flat = {f'state.{k}': v for k, v in data.states.items()}
    flat.update({f'video.{k}': np.asarray(v, dtype=np.uint8) for k, v in data.images.items()})
    flat.update({k: data.text for k in configs['language'].modality_keys})
    return parse_observation_gr00t(flat, configs)


def npz_observation(data, step, modality, lengths=None):
    flat = {}
    for group in ('video', 'state'):
        cfg = modality[group]
        for key in cfg.modality_keys:
            arr = data[f'{group}.{key}']
            if len(arr) == 0:
                raise ValueError(f"Observation series {group}.{key} is empty")
            if lengths is not None:
                lengths.append(len(arr))
            indices = np.clip(step + np.asarray(cfg.delta_indices), 0, len(arr) - 1)
            flat[f'{group}.{key}'] = arr[indices].astype(
                np.uint8 if group == 'video' else np.float32)
    flat.update({k: str(data['language'].item()) for k in modality['language'].modality_keys})
    return parse_observation_gr00t(flat, modality)


def roundtrip(proc, cfg, episode, step):
    state = {k: np.asarray(episode[f'state.{k}'].iloc[step], dtype=np.float64)
             for k in cfg['state'].modality_keys}
    action_cfg = cfg['action']
    indices = np.clip(step + np.asarray(action_cfg.delta_indices), 0, len(episode) - 1)
    raw_actions = {k: np.stack([np.asarray(episode[f'action.{k}'].iloc[i], dtype=np.float64) for i in indices])
                   for k in action_cfg.modality_keys}
    result = {}
    for key, ac in zip(action_cfg.modality_keys, action_cfg.action_configs):
        if ac.rep != ActionRepresentation.RELATIVE:
            continue
        absolute = raw_actions[key]
        ref = state[ac.state_key or key]
        relative = proc._convert_to_relative_action(absolute, ref, ac.type, ac.format)
        restored = proc._convert_to_absolute_action(relative, ref, ac.type, ac.format)
        item = {'max_abs_error': float(np.max(np.abs(restored - absolute))),
                'reference': ref.tolist(), 'first_absolute': absolute[0].tolist(),
                'first_relative': relative[0].tolist(), 'first_reconstructed': restored[0].tolist()}
        if key == 'eef_9d':
            rp = EndEffectorPose.from_action_format(ref, ac.format)
            ap = EndEffectorPose.from_action_format(absolute[0], ac.format)
            dp = EndEffectorPose.from_action_format(relative[0], ac.format)
            item['matrix_relative_max_abs_error'] = float(
                np.max(np.abs(np.linalg.inv(rp.homogeneous) @ ap.homogeneous - dp.homogeneous)))
            item['rotation_rows_max_abs_error'] = float(
                np.max(np.abs(rp.homogeneous[:2, :3].reshape(6) - ref[3:])))
        result[key] = item
    raw_state = {k: v[None, :] for k, v in state.items()}
    normalized = proc.apply_action(raw_actions, TAG.value, state=raw_state)
    decoded = proc.unapply_action(normalized, TAG.value, state=raw_state)
    result['processor_roundtrip'] = {
        k: float(np.max(np.abs(decoded[k] - raw_actions[k]))) for k in raw_actions}
    result['normalized_boundary_fraction'] = {
        k: float(np.mean(np.abs(v) >= 1)) for k, v in normalized.items()}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset', type=Path)
    source.add_argument('--observations', type=Path)
    p.add_argument('--model', type=Path, default=ROOT / 'models/GR00T-N1.7-3B')
    p.add_argument('--episode', type=int, default=1)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--count', type=int, default=3)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--denoising-steps', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-only', action='store_true',
                   help='CPU sample and checkpoint processor numerical check, no model or tokenizer')
    a = p.parse_args()
    if a.start < 0 or a.count < 1 or a.stride < 1 or a.warmup < 0:
        p.error('start >= 0, count >= 1, stride >= 1, warmup >= 0 required')
    if a.check_only:
        if not a.dataset:
            p.error('--check-only requires --dataset')
        raw = json.loads((a.model / 'processor_config.json').read_text())['processor_kwargs']
        stats = json.loads((a.model / 'statistics.json').read_text())
        configs = parse_modality_configs(raw['modality_configs'])
        kwargs = {k: raw[k] for k in ('use_percentiles', 'clip_outliers',
                                      'apply_sincos_state_encoding', 'use_relative_action') if k in raw}
        proc = StateActionProcessor(
            modality_configs=raw['modality_configs'], statistics=stats, **kwargs)
        loader = LeRobotEpisodeLoader(dataset_path=a.dataset, modality_configs=configs[TAG.value])
        episode = loader[a.episode]
        if a.start >= len(episode):
            raise ValueError(f'Start {a.start} exceeds trajectory length {len(episode)}')
        result = {'model': str(a.model), 'dataset': str(a.dataset), 'episode': a.episode, 'step': a.start,
                  'video_key_mapping': loader._video_key_mapping,
                  'processor_settings': kwargs, 'roundtrip': roundtrip(proc, configs[TAG.value], episode, a.start)}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, ensure_ascii=False,
                            indent=2, allow_nan=False) + chr(10))
        print(f'Wrote {a.output}', flush=True)
        return
    if a.denoising_steps < 1:
        p.error("--denoising-steps must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    policy = Gr00tPolicy(embodiment_tag=TAG, model_path=str(a.model), device='cuda')
    policy.model.action_head.num_inference_timesteps = a.denoising_steps
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    modality = policy.get_modality_config()
    if a.dataset:
        loader = LeRobotEpisodeLoader(dataset_path=a.dataset, modality_configs=modality)
        episode = loader[a.episode]
        length = len(episode)
        def make_obs(step): return sample_observation(episode, step, loader.modality_configs)
    else:
        data = np.load(a.observations, allow_pickle=False)
        lengths = []
        first_obs = npz_observation(data, a.start, modality, lengths)
        if len(set(lengths)) != 1:
            raise ValueError(f"Observation series lengths differ: {lengths}")
        length = lengths[0]
        def make_obs(step): return npz_observation(data, step, modality)
    steps = [a.start + i * a.stride for i in range(a.count)]
    if steps[-1] >= length:
        raise ValueError(f'Indices {steps} exceed trajectory length {length}')
    if a.dataset:
        first_obs = make_obs(steps[0])
    for _ in range(a.warmup):
        policy.get_action(first_obs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    records = []
    for i, step in enumerate(steps):
        obs = first_obs if i == 0 else make_obs(step)
        torch.cuda.synchronize()
        start = time.perf_counter()
        actions, _ = policy.get_action(obs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        records.append({'step': step, 'inference_seconds': elapsed,
                        'embodiment_tag': TAG.value,
                        'api_action_shapes': {k: list(v.shape) for k, v in actions.items()},
                        'state': {k: v[0].tolist() for k, v in obs['state'].items()},
                        'language': obs['language'],
                        'action_chunk': {k: v[0].tolist() for k, v in actions.items()}, 'memory': memory()})
        print(f'step={step} inference={elapsed:.3f}s', flush=True)
    times = [r['inference_seconds'] for r in records]
    result = {'model': str(a.model), 'dataset': str(a.dataset) if a.dataset else None,
              'observations': str(a.observations) if a.observations else None,
              'episode': a.episode if a.dataset else None, 'seed': a.seed,
              'denoising_steps': a.denoising_steps, 'warmup_calls': a.warmup,
              'load_seconds': load_seconds, 'cuda_device': torch.cuda.get_device_name(),
              'modality': {k: {'keys': v.modality_keys, 'delta_indices': v.delta_indices}
                           for k, v in modality.items() if k in ('video', 'state', 'action', 'language')},
              'video_key_mapping': loader._video_key_mapping if a.dataset else None,
              'timing_seconds': {'mean': float(np.mean(times)),
                                 'median': float(np.median(times)), 'p90': float(np.percentile(times, 90))}, 'records': records}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + chr(10))
    print(f'Wrote {a.output}', flush=True)


if __name__ == '__main__':
    main()
