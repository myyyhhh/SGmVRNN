#!/usr/bin/env python3
"""
run_new_experiment.py — 新数据完整实验：3 次重复 × 4 变体 × 2 故障类型

用法:
    python baselines/run_new_experiment.py --train   # 训练（各变体训练一次）
    python baselines/run_new_experiment.py --test    # 测试（3 runs × 2 场景）
    python baselines/run_new_experiment.py --eval    # 评估 + 统计报告
    python baselines/run_new_experiment.py --all     # 全套
"""
import argparse
import glob as globmod
import os
import subprocess
import sys
import time
from datetime import datetime
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')

# All scenarios to test
SCENARIOS = [
    ('cpu_stress_r1', 'recommendationservice'),
    ('cpu_stress_r2', 'recommendationservice'),
    ('cpu_stress_r3', 'recommendationservice'),
    ('pod_kill_r1', 'frontend'),
    ('pod_kill_r2', 'frontend'),
    ('pod_kill_r3', 'frontend'),
]

VARIANT_CONFIGS = {
    'lstm_ndt': {
        'script': os.path.join(SCRIPT_DIR, 'lstm_ndt.py'),
        'model_args': '--T 20 --lr 0.001',
        'data_root': '../OnlineBoutique_data/processed_seq_v2',
        'checkpoint_tag': 'lstm_ndt',
        'checkpoint_arg': '--checkpoint_name',
    },
    'vae_no_rnn': {
        'script': os.path.join(SCRIPT_DIR, 'vae_no_rnn.py'),
        'model_args': '--T 20 --lr 0.0002',
        'data_root': '../OnlineBoutique_data/processed_seq_v2',
        'checkpoint_tag': 'vae_no_rnn',
        'checkpoint_arg': '--checkpoint_prefix',
    },
    'cont_vrnn': {
        'script': os.path.join(SCRIPT_DIR, 'continuous_vrnn.py'),
        'model_args': '--T 20 --lr 0.0002',
        'data_root': '../OnlineBoutique_data/processed_seq_v2',
        'checkpoint_tag': 'cont_vrnn',
        'checkpoint_arg': '--checkpoint_prefix',
    },
    't10': {
        'script': os.path.join(PROJECT_DIR, 'SGmVRNN', 'trainer.py'),
        'test_script': os.path.join(PROJECT_DIR, 'SGmVRNN', 'tester.py'),
        'model_args': '--T 10 --learning_rate 0.0002',
        'data_root': '../OnlineBoutique_data/processed_seq_v2_T10',
        'checkpoint_tag': 'catdim5_zdim10_cdim20_hdim20_winsize1_T10_l1',
        'checkpoint_arg': '',
        'is_orig': True,
    },
}

OUTPUT_ROOT = os.path.join(SCRIPT_DIR, 'baselines_new')
MODELS_DIR = os.path.join(OUTPUT_ROOT, 'models')
TESTER_DIR = os.path.join(OUTPUT_ROOT, 'log_tester')
EVAL_DIR = os.path.join(OUTPUT_ROOT, 'eval_results')
CPU_STRESS_RUN = 'cpu_stress_r1'  # all runs share same training data


def run_cmd(cmd, cwd=None, desc=''):
    print(f'  {desc}...', end='', flush=True)
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=7200, cwd=cwd or PROJECT_DIR)
    ok = proc.returncode == 0
    print(f' {"✅" if ok else "❌"} ({time.time()-start:.0f}s)')
    if not ok:
        print(f'    stderr: {proc.stderr[-200:]}')
    return ok


def train_variant(variant, gpu_id=0, epochs=50, seed=None):
    cfg = VARIANT_CONFIGS[variant]
    suffix = f'_seed{seed}' if seed else ''
    model_dir = os.path.join(MODELS_DIR, f'{variant}{suffix}')
    script = cfg['script']

    for svc in ['recommendationservice', 'checkoutservice', 'frontend']:
        data_path = os.path.join(PROJECT_DIR, cfg['data_root'],
                                 CPU_STRESS_RUN, svc, 'train')
        if not os.path.exists(data_path):
            print(f'  ⏭️  {variant}/{svc}: 数据不存在')
            continue

        svc_model_dir = os.path.join(model_dir, svc)
        os.makedirs(svc_model_dir, exist_ok=True)

        if cfg.get('is_orig'):  # t10 uses original trainer
            cmd = [
                sys.executable, script,
                '--dataset_path', data_path,
                '--gpu_id', str(gpu_id), '--n', '7',
                '--checkpoints_path', svc_model_dir,
                '--log_path', os.path.join(MODELS_DIR, 'train_logs', svc),
                '--epochs', str(epochs), '--batch_size', '128',
                '--learning_rate', '0.0002', '--T', '10',
            ]
        else:
            cmd = [
                sys.executable, script,
                '--mode', 'train',
                '--dataset_path', data_path,
                '--gpu_id', str(gpu_id), '--n', '7', '--T', '20',
                '--batch_size', '128', '--epochs', str(epochs),
                '--lr', '0.0002',
                '--checkpoints_path', svc_model_dir,
                cfg['checkpoint_arg'], cfg['checkpoint_tag'],
                '--log_path', os.path.join(MODELS_DIR, 'train_logs', svc),
            ]
        if seed is not None:
            cmd += ['--seed', str(seed)]

        run_cmd(cmd, desc=f'{variant}/{svc}')


def test_variant(variant, gpu_id=0, epochs=50, seed=None):
    cfg = VARIANT_CONFIGS[variant]
    suffix = f'_seed{seed}' if seed else ''
    model_dir = os.path.join(MODELS_DIR, f'{variant}{suffix}')

    for scenario, target_svc in SCENARIOS:
        data_path = os.path.join(PROJECT_DIR, cfg['data_root'],
                                 scenario, target_svc, 'test')
        if not os.path.exists(data_path):
            print(f'  ⏭️  {variant}/{scenario}: 数据不存在')
            continue

        log_dir = os.path.join(TESTER_DIR, f'{variant}{suffix}', scenario)
        os.makedirs(log_dir, exist_ok=True)

        script = cfg.get('test_script', cfg['script'])
        if cfg.get('is_orig'):
            log_file = f'{cfg["checkpoint_tag"]}_epochs{epochs}_loss'
            cmd = [
                sys.executable, script,
                '--dataset_path', data_path,
                '--gpu_id', str(gpu_id), '--n', '7', '--T', '10',
                '--batch_size', '1',
                '--checkpoints_path', os.path.join(model_dir, target_svc),
                '--log_path', log_dir,
                '--log_file', log_file,
                '--start_epoch', str(epochs),
            ]
        else:
            log_file = f'{cfg["checkpoint_tag"]}_epochs{epochs}_loss.txt'
            cmd = [
                sys.executable, script,
                '--mode', 'test',
                '--dataset_path', data_path,
                '--gpu_id', str(gpu_id), '--n', '7', '--T', '20',
                '--batch_size', '1',
                '--checkpoints_path', os.path.join(model_dir, target_svc),
                cfg['checkpoint_arg'], cfg['checkpoint_tag'],
                '--log_path', log_dir,
                '--log_file', log_file,
                '--start_epoch', str(epochs),
            ]
        run_cmd(cmd, desc=f'{variant}/{scenario}')


def evaluate():
    sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'SGmVRNN'))
    from evaluate_pot import bf_search

    os.makedirs(EVAL_DIR, exist_ok=True)
    variants = list(VARIANT_CONFIGS.keys())
    seeds = [42, 123, 999]

    # Collect all results across all seeds × runs
    # all_results: {variant: {scenario: [f1_from_seed1, f1_from_seed2, ...]}}
    all_results = {}
    for variant in variants:
        all_results[variant] = {}
        for scenario, target_svc in SCENARIOS:
            f1s = []
            for seed in seeds:
                log_dir = os.path.join(TESTER_DIR, f'{variant}_seed{seed}', scenario)
                matches = globmod.glob(os.path.join(log_dir, '*_loss.txt'))
                if not matches:
                    continue
                data = np.loadtxt(matches[0], delimiter=',', dtype=str, unpack=False)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                scores = data[:, 1].astype(np.float64)
                labels = np.array([1 if l == 'Anomaly' else 0 for l in data[:, 2]])
                if labels.sum() > 0:
                    (best_f1, *_), _ = bf_search(
                        scores, labels, start=-200, end=50, step_num=500, verbose=False)
                    f1s.append(best_f1)
                else:
                    f1s.append(0.0)
            all_results[variant][scenario] = f1s

    # Group by fault type
    fault_groups = {'cpu_stress': ['cpu_stress_r1', 'cpu_stress_r2', 'cpu_stress_r3'],
                    'pod_kill': ['pod_kill_r1', 'pod_kill_r2', 'pod_kill_r3']}

    print()
    print('=' * 80)
    print('统计结果 (mean ± std — 3 seeds × 3 runs = 9 values)')
    print('=' * 80)

    header = f'{"Method":<16s}'
    for group in fault_groups:
        header += f' {group:>20s}'
    print(header)
    print('-' * 80)

    for variant in variants:
        row = f'{variant:<16s}'
        for group, runs in fault_groups.items():
            all_f1s = []
            for r in runs:
                all_f1s.extend(all_results[variant].get(r, []))
            if all_f1s:
                row += f' {np.mean(all_f1s):.3f}±{np.std(all_f1s):.3f}'
            else:
                row += f' {"N/A":>20s}'
        print(row)
    print('=' * 80)

    # Generate comparison chart
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        group_names = ['CPU Stress', 'Pod Kill']

        for idx, (group, runs) in enumerate(fault_groups.items()):
            ax = axes[idx]
            v_names, means, stds = [], [], []
            for v in variants:
                all_f1s = []
                for r in runs:
                    all_f1s.extend(all_results[v].get(r, []))
                if all_f1s:
                    v_names.append(v)
                    means.append(np.mean(all_f1s))
                    stds.append(np.std(all_f1s))

            x = range(len(v_names))
            ax.bar(x, means, yerr=stds, capsize=5, color='steelblue')
            ax.set_xticks(list(x))
            ax.set_xticklabels(v_names, rotation=30, ha='right', fontsize=8)
            ax.set_title(f'{group_names[idx]} (3 seeds × 3 runs)')
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('F1')
            ax.grid(axis='y', alpha=0.3)
            for i, (m, s) in enumerate(zip(means, stds)):
                if m > 0:
                    ax.text(i, m + s + 0.02, f'{m:.2f}±{s:.2f}',
                            ha='center', fontsize=7)

        plt.suptitle('Seeded Repeated Experiments (3 seeds × 3 runs)', fontweight='bold')
        plt.tight_layout()
        path = os.path.join(EVAL_DIR, 'comparison_new_data.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f'\n📊 对比图: {path}')
    except Exception as e:
        print(f'  ⚠️  图表生成失败: {e}')

    print(f'\n✅ 评估完成！')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seeds', type=int, nargs='*', default=[42, 123, 999],
                        help='随机种子列表（默认: 42 123 999）')
    parser.add_argument('--variants', nargs='*',
                        default=list(VARIANT_CONFIGS.keys()))
    args = parser.parse_args()

    do_train = args.train or args.all
    do_test = args.test or args.all
    do_eval = args.eval or args.all

    for variant in args.variants:
        for seed in args.seeds:
            tag = f'{variant}_seed{seed}'
            if do_train:
                print(f'\n🚀 训练: {tag}')
                train_variant(variant, args.gpu, args.epochs, seed)
            if do_test:
                print(f'\n🧪 测试: {tag}')
                test_variant(variant, args.gpu, args.epochs, seed)

    if do_eval:
        print(f'\n📊 评估...')
        evaluate()

    print('\n✅ 完成！')


if __name__ == '__main__':
    main()
