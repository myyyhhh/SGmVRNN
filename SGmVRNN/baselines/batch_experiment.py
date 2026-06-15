#!/usr/bin/env python3
"""
batch_experiment.py — 批量运行基线 + 消融实验

对所有变体 × 3 个故障服务，依次执行训练 + 测试。

变体:
  lstm_ndt     LSTM-NDT 基线
  vae_no_rnn   VAE（无 RNN）消融
  cont_vrnn    连续 VRNN（无 Gumbel）消融
  t10          SGmVRNN 完整模型 + T=10 消融

用法:
    python baselines/batch_experiment.py --train    # 训练所有变体
    python baselines/batch_experiment.py --test     # 测试所有变体
    python baselines/batch_experiment.py --all      # 训练+测试
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


SERVICES = ['recommendationservice', 'checkoutservice', 'frontend']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')  # SGmVRNN root
DATA_ROOT = os.path.join(SCRIPT_DIR, '..', '..', 'OnlineBoutique_data')  # stm root

VARIANTS = {
    'lstm_ndt': {
        'script': os.path.join(SCRIPT_DIR, 'lstm_ndt.py'),
        'data_dir': os.path.join(DATA_ROOT, 'processed_seq'),
        'checkpoint_prefix': 'lstm_ndt',
        'checkpoint_arg': '--checkpoint_name',   # lstm_ndt uses --checkpoint_name not --checkpoint_prefix
    },
    'vae_no_rnn': {
        'script': os.path.join(SCRIPT_DIR, 'vae_no_rnn.py'),
        'data_dir': os.path.join(DATA_ROOT, 'processed_seq'),
        'checkpoint_prefix': 'vae_no_rnn',
    },
    'cont_vrnn': {
        'script': os.path.join(SCRIPT_DIR, 'continuous_vrnn.py'),
        'data_dir': os.path.join(DATA_ROOT, 'processed_seq'),
        'checkpoint_prefix': 'cont_vrnn',
    },
    't10': {
        'script': os.path.join(PROJECT_DIR, 'SGmVRNN', 'trainer.py'),
        'data_dir': os.path.join(DATA_ROOT, 'processed_seq_T10'),
        'checkpoint_prefix': 'catdim5_zdim10_cdim20_hdim20_winsize1_T10_l1',
        'test_script': os.path.join(PROJECT_DIR, 'SGmVRNN', 'tester.py'),
    },
}


def run_experiment(variant, service, mode, gpu_id=0, epochs=50):
    """Run train or test for one variant × service combination."""
    cfg = VARIANTS[variant]
    is_t10 = variant == 't10'
    script = cfg.get('test_script' if mode == 'test' else 'script', cfg['script'])

    data_path = os.path.join(cfg['data_dir'], service, mode)
    if not os.path.exists(data_path):
        print(f'  ⏭️  {variant}/{service} — 数据不存在: {data_path}')
        return False

    models_dir = os.path.join(PROJECT_DIR, 'baselines', 'models', variant, service)
    log_dir = os.path.join(PROJECT_DIR, 'baselines', 'log_tester', variant, service)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if is_t10:
        cmd = [
            sys.executable, script,
            '--dataset_path', data_path,
            '--gpu_id', str(gpu_id),
            '--n', '7',
            '--T', '10',
            '--learning_rate', '0.0002',
            '--batch_size', '128' if mode == 'train' else '1',
            '--checkpoints_path', models_dir,
            '--log_path', log_dir,
        ]

        # set log_file for tester
        if mode == 'test':
            cmd += ['--start_epoch', str(epochs)]
            # log_file is configured auto by tester.py
        else:
            cmd += ['--epochs', str(epochs)]
    else:
        cmd = [
            sys.executable, script,
            '--mode', mode,
            '--dataset_path', data_path,
            '--gpu_id', str(gpu_id),
            '--n', '7',
            '--T', '20',
            '--batch_size', '128' if mode == 'train' else '1',
            '--checkpoints_path', models_dir,
            '--log_path', log_dir,
        ]
        ckpt_arg = cfg.get('checkpoint_arg', '--checkpoint_prefix')
        if mode == 'train':
            cmd += ['--epochs', str(epochs), '--lr', '0.0002']
        else:
            cmd += ['--start_epoch', str(epochs)]
        cmd += [ckpt_arg, cfg['checkpoint_prefix']]

    print(f'  Running: {variant}/{service} [{mode}]')
    sys.stdout.flush()

    cwd = PROJECT_DIR
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=7200, cwd=cwd)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f'  ✅ {variant}/{service} [{mode}] — {elapsed:.0f}s')
        return True
    else:
        print(f'  ❌ {variant}/{service} [{mode}] — rc={result.returncode}')
        print(f'     stderr: {result.stderr[-300:]}')
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='跑训练')
    parser.add_argument('--test', action='store_true', help='跑测试')
    parser.add_argument('--all', action='store_true', help='训练+测试')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--variants', nargs='*',
                        default=['lstm_ndt', 'vae_no_rnn', 'cont_vrnn', 't10'])
    parser.add_argument('--services', nargs='*', default=SERVICES)
    args = parser.parse_args()

    do_train = args.train or args.all
    do_test = args.test or args.all

    for variant in args.variants:
        print(f'\n{"="*50}')
        print(f'变体: {variant}')
        print(f'{"="*50}')
        for svc in args.services:
            print(f'  --- {svc} ---')
            if do_train:
                run_experiment(variant, svc, 'train', args.gpu, args.epochs)
            if do_test:
                run_experiment(variant, svc, 'test', args.gpu, args.epochs)

    print('\n✅ 实验完成！')


if __name__ == '__main__':
    main()
