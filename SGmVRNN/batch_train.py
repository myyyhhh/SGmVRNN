#!/usr/bin/env python3
"""
batch_train.py — 一键训练所有 Online Boutique 服务的 SGmVRNN 模型

对 processed_seq 下每个 service 依次执行 trainer.py，
记录每个服务的训练日志到 logs/batch_train_{timestamp}.log。

用法:
    python batch_train.py                       # 训练所有服务
    python batch_train.py --services checkoutservice frontend  # 只训练指定服务
    python batch_train.py --dry-run             # 只显示要执行的命令，不训练

参数覆盖:
    --epochs 30 --batch_size 64 --learning_rate 0.0002
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

# ── 默认超参数（INFOCOM22 论文推荐值） ──
DEFAULT_HPARAMS = {
    'epochs': 50,
    'batch_size': 128,
    'learning_rate': 0.0002,
    'gpu_id': 0,
    'T': 20,
    'win_size': 1,
    'l': 1,
    'n': 7,  # 所有服务都是 7 个 KPI
    'categorical_dims': 5,
    'z_dims': 10,
    'conv_dims': 20,
    'hidden_dims': 20,
    'temperature': 5.0,
    'min_temperature': 0.1,
    'anneal_rate': 0.1,
    'num_workers': 4,
    'checkpoints_interval': 1,
}


def find_services(processed_seq_dir):
    """扫描 processed_seq 下的所有服务目录"""
    services = []
    for item in sorted(os.listdir(processed_seq_dir)):
        train_dir = os.path.join(processed_seq_dir, item, 'train')
        if os.path.isdir(train_dir) and any(f.endswith('.seq') for f in os.listdir(train_dir)):
            services.append(item)
    return services


def main():
    parser = argparse.ArgumentParser(description='SGmVRNN 批量训练脚本')
    parser.add_argument('--processed_seq', default='../OnlineBoutique_data/processed_seq',
                        help='processed_seq 根目录')
    parser.add_argument('--trainer', default='SGmVRNN/trainer.py',
                        help='trainer.py 路径')
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
    trainer_path = os.path.join(script_dir, args.trainer)

    if not os.path.exists(trainer_path):
        # 也试试从 SGmVRNN/SGmVRNN/ 下找
        alt_path = os.path.join(script_dir, 'SGmVRNN', 'trainer.py')
        if os.path.exists(alt_path):
            trainer_path = alt_path
        else:
            print(f'❌ 找不到 trainer.py: {trainer_path}')
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
        print('❌ 没有找到可训练的服务')
        sys.exit(1)

    # ── 日志文件 ──
    log_dir = os.path.join(script_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'batch_train_{timestamp}.log')
    summary_file = os.path.join(log_dir, f'batch_train_{timestamp}_summary.txt')

    def log(msg, also_print=True):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}\n')
        if also_print:
            print(msg)

    log(f'🚀 SGmVRNN 批量训练启动')
    log(f'📁 数据目录: {processed_seq_dir}')
    log(f'📝 训练脚本: {trainer_path}')
    log(f'📋 服务列表 ({len(services)}): {", ".join(services)}')
    log(f'⚙️  超参数: {hparams}')
    log(f'')

    if args.dry_run:
        print(f'\n⚠️  DRY-RUN 模式 — 以下命令会依次执行:\n')

    # ── 依次训练每个服务 ──
    results = []
    total_start = time.time()

    for i, svc in enumerate(services, 1):
        train_path = os.path.join(processed_seq_dir, svc, 'train')
        log_path = os.path.join(script_dir, 'log_trainer', svc)
        checkpoint_path = os.path.join(script_dir, 'model', svc)

        os.makedirs(log_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)

        cmd = [
            sys.executable, trainer_path,
            '--dataset_path', train_path,
            '--log_path', log_path,
            '--checkpoints_path', checkpoint_path,
            '--gpu_id', str(hparams['gpu_id']),
            '--epochs', str(hparams['epochs']),
            '--batch_size', str(hparams['batch_size']),
            '--learning_rate', str(hparams['learning_rate']),
            '--T', str(hparams['T']),
            '--win_size', str(hparams['win_size']),
            '--l', str(hparams['l']),
            '--n', str(hparams['n']),
            '--categorical_dims', str(hparams['categorical_dims']),
            '--z_dims', str(hparams['z_dims']),
            '--conv_dims', str(hparams['conv_dims']),
            '--hidden_dims', str(hparams['hidden_dims']),
            '--temperature', str(hparams['temperature']),
            '--min_temperature', str(hparams['min_temperature']),
            '--anneal_rate', str(hparams['anneal_rate']),
            '--num_workers', str(hparams['num_workers']),
            '--checkpoints_interval', str(hparams['checkpoints_interval']),
        ]

        log(f'[{i}/{len(services)}] 🔧 训练 {svc} ...')

        if args.dry_run:
            print(' '.join(cmd))
            print()
            results.append({'service': svc, 'status': 'DRY-RUN'})
            continue

        # 执行训练
        svc_start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 小时超时
                cwd=os.path.join(script_dir, 'SGmVRNN'),
            )
            elapsed = time.time() - svc_start

            # 写详细日志
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f'─── {svc} stdout ───\n')
                f.write(proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout)
                if proc.stderr:
                    f.write(f'─── {svc} stderr ───\n')
                    f.write(proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr)
                f.write('\n')

            if proc.returncode == 0:
                # 从输出中提取最终 loss
                final_loss = None
                for line in proc.stdout.split('\n'):
                    if 'Average Loss' in line:
                        try:
                            final_loss = float(line.split('Average Loss:')[1].split()[0])
                        except:
                            pass

                status = '✅'
                log(f'  ✅ {svc} 完成 ({elapsed:.1f}s)' +
                    (f', 最终 loss={final_loss:.4f}' if final_loss else ''))
                results.append({
                    'service': svc, 'status': 'ok', 'time': elapsed,
                    'loss': final_loss, 'returncode': 0
                })
            else:
                log(f'  ❌ {svc} 失败 (returncode={proc.returncode}, {elapsed:.1f}s)')
                log(f'     最后 stderr: {proc.stderr[-500:]}')
                results.append({
                    'service': svc, 'status': 'fail', 'time': elapsed,
                    'returncode': proc.returncode
                })

        except subprocess.TimeoutExpired:
            log(f'  ⏰ {svc} 超时 (>2h)')
            results.append({'service': svc, 'status': 'timeout'})
        except Exception as e:
            log(f'  💥 {svc} 异常: {e}')
            results.append({'service': svc, 'status': 'error', 'error': str(e)})

        log(f'')
        # 服务之间休息 3 秒，避免 GPU 显存释放不及时
        time.sleep(3)

    # ── 汇总 ──
    total_elapsed = time.time() - total_start
    ok_count = sum(1 for r in results if r['status'] == 'ok')
    fail_count = sum(1 for r in results if r['status'] != 'ok')

    summary = [
        '═══════════════════════════════════════════',
        '📊 SGmVRNN 批量训练汇总',
        '═══════════════════════════════════════════',
        f'总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)',
        f'成功: {ok_count}/{len(results)}',
        f'失败: {fail_count}/{len(results)}',
        '',
        f'{"Service":25s} {"Status":8s} {"Time":8s} {"Loss":12s}',
        '-' * 55,
    ]

    for r in results:
        status_str = '✅' if r['status'] == 'ok' else '❌'
        time_str = f'{r.get("time", 0):.0f}s' if 'time' in r else '-'
        loss_str = f'{r.get("loss", "-"):.4f}' if r.get('loss') else '-'
        summary.append(f'{r["service"]:25s} {status_str:8s} {time_str:8s} {loss_str:12s}')

    summary.append('')
    summary.append(f'日志文件: {log_file}')

    summary_text = '\n'.join(summary)
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_text + '\n')

    log(summary_text)

    print(f'\n📄 详细日志: {log_file}')
    print(f'📄 训练汇总: {summary_file}')


if __name__ == '__main__':
    main()
