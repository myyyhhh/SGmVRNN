#!/usr/bin/env python3
"""
batch_test.py — 一键测试所有 Online Boutique 服务的 SGmVRNN 模型

对 model/ 下每个已训练的 service 依次执行 tester.py，
输出测试结果（每个样本的 log-likelihood + 标签）到 log_tester/{service}。

用法:
    python batch_test.py                          # 测试所有服务
    python batch_test.py --services checkoutservice frontend  # 只测试指定服务
    python batch_test.py --dry-run                # 只显示命令不执行
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


# ── 默认超参数（必须与训练时一致） ──
DEFAULT_HPARAMS = {
    'gpu_id': 0,
    'batch_size': 1,       # tester 默认 batch_size=1
    'T': 20,
    'win_size': 1,
    'l': 1,
    'n': 7,
    'categorical_dims': 5,
    'z_dims': 10,
    'conv_dims': 20,
    'hidden_dims': 20,
    'learning_rate': 0.0002,
    'temperature': 5.0,
    'min_temperature': 0.1,
    'anneal_rate': 0.1,
    'num_workers': 4,
    'start_epoch': 50,     # 用第 50 个 epoch 的 checkpoint
}


def find_services(processed_seq_dir):
    """扫描 processed_seq 下有测试数据的服务"""
    services = []
    for item in sorted(os.listdir(processed_seq_dir)):
        test_dir = os.path.join(processed_seq_dir, item, 'test')
        if os.path.isdir(test_dir) and any(f.endswith('.seq') for f in os.listdir(test_dir)):
            services.append(item)
    return services


def main():
    parser = argparse.ArgumentParser(description='SGmVRNN 批量测试脚本')
    parser.add_argument('--processed_seq', default='../OnlineBoutique_data/processed_seq',
                        help='processed_seq 根目录')
    parser.add_argument('--tester', default='SGmVRNN/tester.py',
                        help='tester.py 路径')
    parser.add_argument('--checkpoints_root', default='model',
                        help='checkpoint 根目录')
    parser.add_argument('--log_root', default='log_tester',
                        help='测试日志根目录')
    parser.add_argument('--services', nargs='*', default=None,
                        help='指定服务列表（默认全部）')
    parser.add_argument('--dry-run', action='store_true',
                        help='只显示命令不执行')
    for k, v in DEFAULT_HPARAMS.items():
        flag = '--' + k.replace('_', '-')
        parser.add_argument(flag, type=type(v), default=v,
                            help=f'默认: {v}')

    args = parser.parse_args()
    hparams = {k: getattr(args, k) for k in DEFAULT_HPARAMS}

    # 路径解析
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_seq_dir = os.path.join(script_dir, args.processed_seq)
    tester_path = os.path.join(script_dir, args.tester)

    if not os.path.exists(tester_path):
        alt_path = os.path.join(script_dir, 'SGmVRNN', 'tester.py')
        if os.path.exists(alt_path):
            tester_path = alt_path
        else:
            print(f'❌ 找不到 tester.py: {tester_path}')
            sys.exit(1)

    # 发现服务
    all_services = find_services(processed_seq_dir)
    if args.services:
        services = [s for s in args.services if s in all_services]
        missing = [s for s in args.services if s not in all_services]
        if missing:
            print(f'⚠️  以下服务不在 processed_seq 中: {missing}')
    else:
        services = all_services

    if not services:
        print('❌ 没有找到可测试的服务')
        sys.exit(1)

    # ── 日志文件 ──
    log_dir = os.path.join(script_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'batch_test_{timestamp}.log')

    def log(msg, also_print=True):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}\n')
        if also_print:
            print(msg)

    log(f'🚀 SGmVRNN 批量测试启动')
    log(f'📁 数据目录: {processed_seq_dir}')
    log(f'📝 测试脚本: {tester_path}')
    log(f'📋 服务列表 ({len(services)}): {", ".join(services)}')
    log(f'⚙️  超参数: {hparams}')
    log(f'')

    if args.dry_run:
        print(f'\n⚠️  DRY-RUN 模式 — 以下命令会依次执行:\n')

    # ── 依次测试每个服务 ──
    results = []
    total_start = time.time()

    for i, svc in enumerate(services, 1):
        test_path = os.path.join(processed_seq_dir, svc, 'test')
        model_path = os.path.join(script_dir, args.checkpoints_root, svc)
        log_path = os.path.join(script_dir, args.log_root, svc)

        if not os.path.exists(model_path):
            log(f'  ⏭️  {svc} — 无模型文件，跳过')
            results.append({'service': svc, 'status': 'skip'})
            continue

        os.makedirs(log_path, exist_ok=True)

        cmd = [
            sys.executable, tester_path,
            '--dataset_path', test_path,
            '--checkpoints_path', model_path,
            '--log_path', log_path,
            '--gpu_id', str(hparams['gpu_id']),
            '--batch_size', str(hparams['batch_size']),
            '--T', str(hparams['T']),
            '--win_size', str(hparams['win_size']),
            '--l', str(hparams['l']),
            '--n', str(hparams['n']),
            '--categorical_dims', str(hparams['categorical_dims']),
            '--z_dims', str(hparams['z_dims']),
            '--conv_dims', str(hparams['conv_dims']),
            '--hidden_dims', str(hparams['hidden_dims']),
            '--start_epoch', str(hparams['start_epoch']),
        ]

        log(f'[{i}/{len(services)}] 🔧 测试 {svc} ...')

        if args.dry_run:
            print(' '.join(cmd))
            print()
            results.append({'service': svc, 'status': 'DRY-RUN'})
            continue

        svc_start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout per service
                cwd=os.path.join(script_dir, 'SGmVRNN'),
            )
            elapsed = time.time() - svc_start

            with open(log_file, 'a', encoding='utf-8') as f:
                if proc.stdout:
                    f.write(f'─── {svc} stdout (last 2k) ───\n')
                    f.write(proc.stdout[-2000:] + '\n')
                if proc.stderr:
                    f.write(f'─── {svc} stderr ───\n')
                    f.write(proc.stderr[-1000:] + '\n')

            if proc.returncode == 0:
                log(f'  ✅ {svc} 完成 ({elapsed:.1f}s)')
                results.append({'service': svc, 'status': 'ok', 'time': elapsed})
            else:
                log(f'  ❌ {svc} 失败 (returncode={proc.returncode}, {elapsed:.1f}s)')
                results.append({'service': svc, 'status': 'fail', 'time': elapsed})

        except subprocess.TimeoutExpired:
            log(f'  ⏰ {svc} 超时 (>30min)')
            results.append({'service': svc, 'status': 'timeout'})
        except Exception as e:
            log(f'  💥 {svc} 异常: {e}')
            results.append({'service': svc, 'status': 'error', 'error': str(e)})

        log(f'')

    # ── 汇总 ──
    total_elapsed = time.time() - total_start
    ok_count = sum(1 for r in results if r['status'] == 'ok')

    summary_lines = [
        '═══════════════════════════════════════════',
        '📊 SGmVRNN 批量测试汇总',
        '═══════════════════════════════════════════',
        f'总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)',
        f'成功: {ok_count}/{len(results)}',
        '',
        f'{"Service":25s} {"Status":8s} {"Time":10s}',
        '-' * 45,
    ]
    for r in results:
        s = '✅' if r['status'] == 'ok' else ('⏭️ ' if r['status'] == 'skip' else '❌')
        t = f'{r.get("time", 0):.0f}s' if 'time' in r else '-'
        summary_lines.append(f'{r["service"]:25s} {s:8s} {t:10s}')

    summary_lines.append('')
    summary_lines.append(f'日志文件: {log_file}')

    summary_text = '\n'.join(summary_lines)
    log(summary_text)
    print(f'\n📄 详细日志: {log_file}')


if __name__ == '__main__':
    main()
