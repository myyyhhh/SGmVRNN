#!/usr/bin/env python3
"""
compare_results.py — 对比所有实验变体的异常检测结果

扫描 baselines/log_tester/ 下所有变体 × 服务的测试结果，
运行 BF search 评估，输出汇总对比表和组合柱状图。

用法:
    python baselines/compare_results.py
    python baselines/compare_results.py --variants lstm_ndt vae_no_rnn
"""
import argparse
import glob as globmod
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SGmVRNN'))
from evaluate_pot import bf_search


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(BASE_DIR, 'log_tester')
OUTPUT_DIR = os.path.join(BASE_DIR, 'eval_results')

VARIANTS = ['lstm_ndt', 'vae_no_rnn', 'cont_vrnn', 't10']
SERVICES = ['recommendationservice', 'checkoutservice', 'frontend']
EPOCHS = 50


def load_tester_file(filepath):
    """读取 tester CSV，返回 (scores, labels)"""
    data = np.loadtxt(filepath, delimiter=',', dtype=str, unpack=False)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    scores = data[:, 1].astype(np.float64)
    labels = np.array([1 if l == 'Anomaly' else 0 for l in data[:, 2]])
    return scores, labels


def find_tester_file(variant, service):
    """在 log_tester/{variant}/{service}/ 下找 _loss.txt"""
    svc_dir = os.path.join(LOG_ROOT, variant, service)
    if not os.path.isdir(svc_dir):
        return None
    matches = globmod.glob(os.path.join(svc_dir, f'*_epochs{EPOCHS}_loss.txt'))
    if not matches:
        matches = globmod.glob(os.path.join(svc_dir, '*_loss.txt'))
    return matches[0] if matches else None


def evaluate_one(scores, labels):
    """对一组分数运行 BF search，返回指标字典"""
    if labels.sum() == 0:
        return {'f1': 0, 'precision': 0, 'recall': 0, 'threshold': 0}
    try:
        (best_f1, precision, recall, tp, tn, fp, fn, _), th = \
            bf_search(scores, labels, start=-200, end=50, step_num=500, verbose=False)
        return {'f1': best_f1, 'precision': precision, 'recall': recall,
                'threshold': th, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}
    except Exception:
        return {'f1': -1, 'precision': -1, 'recall': -1, 'threshold': -1}


def collect_all_results(variants=None):
    """收集所有变体×服务的结果"""
    if variants is None:
        variants = VARIANTS
    results = {}
    for variant in variants:
        results[variant] = {}
        for svc in SERVICES:
            fp = find_tester_file(variant, svc)
            if fp is None:
                results[variant][svc] = None
                continue
            scores, labels = load_tester_file(fp)
            metrics = evaluate_one(scores, labels)
            metrics['n_samples'] = len(scores)
            metrics['n_anomaly'] = int(labels.sum())
            results[variant][svc] = metrics
    return results


def print_table(results, variants=None):
    """打印对比汇总表"""
    if variants is None:
        variants = VARIANTS
    print()
    print('=' * 90)
    print(f'实验对比汇总 (epoch={EPOCHS})')
    print('=' * 90)
    header = f'{"Method":<16s}'
    for svc in SERVICES:
        header += f' {svc[:8]:>8s} F1 {"Prec":>5s} {"Rec":>5s}'
    print(header)
    print('-' * 90)

    for variant in variants:
        row = f'{variant:<16s}'
        for svc in SERVICES:
            r = results.get(variant, {}).get(svc)
            if r is None:
                row += f' {"N/A":>20s}'
            else:
                row += f' {r["n_anomaly"]:>4d}  {r["f1"]:.3f} {r["precision"]:.3f} {r["recall"]:.3f}'
        print(row)
    print('=' * 90)


def plot_comparison(results, output_path, variants):
    """生成组合柱状图"""
    if variants is None:
        variants = VARIANTS
    os.makedirs(output_path, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, svc in enumerate(SERVICES):
        ax = axes[idx]
        v_names = []
        f1s = []
        precs = []
        recs = []
        for v in variants:
            r = results.get(v, {}).get(svc)
            if r is not None:
                variants.append(v)
                f1s.append(r['f1'])
                precs.append(r['precision'])
                recs.append(r['recall'])

        x = np.arange(len(v_names))
        w = 0.25
        ax.bar(x - w, f1s, w, label='F1', color='steelblue')
        ax.bar(x, precs, w, label='Precision', color='coral')
        ax.bar(x + w, recs, w, label='Recall', color='seagreen')
        ax.set_title(svc)
        ax.set_xticks(x)
        ax.set_xticklabels(v_names, rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

        # 标注 F1 数值
        for i, f1 in enumerate(f1s):
            if f1 > 0:
                ax.text(x[i] - w, f1 + 0.02, f'{f1:.2f}', ha='center',
                        va='bottom', fontsize=7, fontweight='bold')

    plt.suptitle(f'SGmVRNN 基线 + 消融实验对比 (epoch={EPOCHS})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_path, 'comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'📊 对比图: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='*')
    args = parser.parse_args()

    variants = args.variants if args.variants else VARIANTS

    print(f'📋 收集 {len(variants)} 个变体 × {len(SERVICES)} 个服务的结果...')
    results = collect_all_results(variants)
    print_table(results, variants)
    plot_comparison(results, OUTPUT_DIR, variants)
    print(f'✅ 对比完成！结果目录: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
