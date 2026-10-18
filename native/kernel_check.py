#!/usr/bin/env python3
"""Small native INT8 dispatch/correctness check, not a model quality evaluation.

Uses affine 8-bit group-64 BF16 weights and JEV-relevant dimensions. The short
case must remain identical; eligible cases must be finite, within the upstream
5% relative-L2 tolerance, and at least one must differ to detect a silently
ignored feature flag. This establishes activity and a small numerical gate,
not general accuracy, calibration, or deployment safety.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(report, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20261019)
    args = parser.parse_args()
    import mlx.core as mx

    try:
        metal_version = importlib.metadata.version('mlx-metal')
    except importlib.metadata.PackageNotFoundError:
        metal_version = None
    previous_flag = os.environ.get('MLX_QMM_INT8')
    report = {
        'schema_version': 1,
        'started_at': datetime.now(timezone.utc).isoformat(),
        'mlx_version': importlib.metadata.version('mlx'),
        'mlx_metal_distribution_version': metal_version,
        'mlx_module_file': mx.__file__,
        'python_executable': sys.executable,
        'macos': platform.mac_ver()[0],
        'device': mx.device_info(),
        'seed': args.seed,
        'dtype': 'bfloat16',
        'weight_quantization': {'mode': 'affine', 'bits': 8, 'group_size': 64},
        'flag_values': ['0', '1'],
        'previous_flag': previous_flag,
        'relative_l2_tolerance': 0.05,
        'method': 'Fresh QMM operations built after setting MLX_QMM_INT8, fully evaluated before the next flag switch. Identical seeded tensors per comparison. Relative errors calculated in float32. This is a small kernel gate, not general model accuracy or probability calibration validation.',
        'cases': [],
    }
    save(args.out, report)
    eligible_changed = False
    try:
        if not mx.metal.is_available():
            raise RuntimeError('Metal GPU is required to test the native staged path')
        mx.set_default_device(mx.gpu)
        mx.random.seed(args.seed)
        # N=2048 covers an attention projection; N=17408 an MLP projection.
        # The longest M case uses the smaller N to keep scratch memory bounded.
        shapes = [(43, 2048, 5120), (76, 2048, 5120),
                  (1076, 2048, 5120), (76, 17408, 5120)]
        for m, n, k in shapes:
            weight = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
            x = (mx.random.normal((m, k)) * 0.25).astype(mx.bfloat16)
            packed, scales, biases = mx.quantize(weight, group_size=64, bits=8)
            mx.eval(packed, scales, biases, x)
            del weight
            outputs = []
            for flag in ('0', '1'):
                mx.synchronize()
                os.environ['MLX_QMM_INT8'] = flag
                y = mx.quantized_matmul(x, packed, scales, biases,
                                       transpose=True, group_size=64, bits=8,
                                       mode='affine')
                mx.eval(y)
                mx.synchronize()
                outputs.append(y)
            a, b = [value.astype(mx.float32) for value in outputs]
            delta = b - a
            max_abs = mx.max(mx.abs(delta)).item()
            relative_l2 = (mx.sqrt(mx.sum(delta * delta)) /
                           mx.maximum(mx.sqrt(mx.sum(a * a)), 1e-30)).item()
            finite = bool(mx.all(mx.isfinite(a)).item() and mx.all(mx.isfinite(b)).item())
            identical = bool(mx.array_equal(outputs[0].view(mx.uint16), outputs[1].view(mx.uint16)).item())
            eligible = m >= 64
            row = {
                'M': m, 'N': n, 'K': k, 'expected_staged_eligible': eligible,
                'all_finite': finite, 'bitwise_identical': identical,
                'max_abs_error': max_abs, 'relative_l2_error': relative_l2,
                'within_relative_l2_tolerance': math.isfinite(relative_l2) and relative_l2 < 0.05,
                'mlx_active_bytes': mx.get_active_memory(),
                'mlx_peak_bytes': mx.get_peak_memory(),
            }
            report['cases'].append(row)
            save(args.out, report)
            print(json.dumps(row), flush=True)
            if not finite or not row['within_relative_l2_tolerance']:
                raise AssertionError(f'Numerical gate failed for {(m,n,k)}')
            if not eligible and not identical:
                raise AssertionError('Short-input fallback differs between flags')
            eligible_changed |= eligible and not identical
            del a, b, delta, outputs, y, x, packed, scales, biases
        if not eligible_changed:
            raise AssertionError('No eligible output changed; the INT8 flag may be ignored')
        report['eligible_output_changed'] = eligible_changed
        # A wide BF16 attention head exercises the split-D JIT template that
        # failed with the upstream 0.32.4 argument-list mismatch. Causal Q/K
        # length >=1024 is required for the fused D=256 path in this release.
        q = (mx.random.normal((1, 2, 1024, 256)) * 0.25).astype(mx.bfloat16)
        k = (mx.random.normal((1, 2, 1024, 256)) * 0.25).astype(mx.bfloat16)
        v = (mx.random.normal((1, 2, 1024, 256)) * 0.25).astype(mx.bfloat16)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=256 ** -0.5, mask='causal')
        mx.eval(y)
        scores = (q.astype(mx.float32) @ mx.swapaxes(k.astype(mx.float32), -1, -2)) * (256 ** -0.5)
        causal = mx.arange(1024)[:, None] >= mx.arange(1024)[None, :]
        reference = mx.softmax(mx.where(causal, scores, -float('inf')), axis=-1) @ v.astype(mx.float32)
        delta = y.astype(mx.float32) - reference
        relative_l2 = (mx.sqrt(mx.sum(delta * delta)) /
                       mx.maximum(mx.sqrt(mx.sum(reference * reference)), 1e-30)).item()
        attention = {'shape': list(q.shape), 'mask': 'causal', 'dtype': 'bfloat16',
                     'expected_path': 'attention_nax_dsplit',
                     'all_finite': bool(mx.all(mx.isfinite(y)).item()),
                     'relative_l2_error': relative_l2,
                     'passed': math.isfinite(relative_l2) and relative_l2 < 0.05}
        report['attention_check'] = attention
        save(args.out, report)
        print(json.dumps({'attention_check': attention}), flush=True)
        if not attention['all_finite'] or not attention['passed']:
            raise AssertionError('Wide-head attention JIT numerical gate failed')
        report['passed'] = True
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(args.out, report)
    except BaseException as exc:
        report['passed'] = False
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        save(args.out, report)
        raise
    finally:
        mx.synchronize()
        if previous_flag is None:
            os.environ.pop('MLX_QMM_INT8', None)
        else:
            os.environ['MLX_QMM_INT8'] = previous_flag


if __name__ == '__main__':
    main()
