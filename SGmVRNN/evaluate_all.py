#!/usr/bin/env python3
"""
evaluate_all.py — 批量评估 SGmVRNN 异常检测效果

对每个已测试的服务：
  1. 读取 tester.py 输出的 log-likelihood + 标签
  2. 运行 POT（Peak Over Threshold）自动计算阈值
  3. 运行暴力搜索（BF）寻找最佳 F1 阈值
  4. 输出 precision / recall / F1 等指标
  5. 生成异常分数可视化

用法:
    python evaluate_all.py                          # 评估所有服务
    python evaluate_all.py --services checkoutservice frontend  # 指定服务
    python evaluate_all.py --plot                   # 生成可视化图表
"""
import argparse
import os
import sys
import re
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，不弹窗
import matplotlib.pyplot as plt

# ── 加载原有的 evaluate_pot 和 spot ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SGmVRNN'))
from evaluate_pot import bf_search, pot_eval, calc_point2point, adjust_predicts


# ── 中文字体设置 ──
def setup_chinese_font():
    """尝试设置中文字体"""
    for font_name in ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']:
        try:
            matplotlib.font_manager.findfont(font_name, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            return
        except Exception:
            continue


setup_chinese_font()


def find_tester_outputs(log_root, epochs=50):
    """扫描 log_root 下的所有测试输出文件"""
    services = []
    for item in sorted(os.listdir(log_root)):
        svc_dir = os.path.join(log_root, item)
        if not os.path.isdir(svc_dir):
            continue
        # 查找 tester 输出的 loss txt 文件
        pattern = f'*_epochs{epochs}_loss.txt'
        import glob
        matches = glob.glob(os.path.join(svc_dir, pattern))
        if matches:
            services.append({
                'name': item,
                'tester_file': matches[0],
            })
        else:
            # 也试试找任意 _loss.txt
            matches = glob.glob(os.path.join(svc_dir, '*_loss.txt'))
            if matches:
                services.append({
                    'name': item,
                    'tester_file': matches[0],
                })
    return services


def load_tester_output(tester_file):
    """加载 tester 输出的 CSV 文件，返回 (timestamps, scores, labels_0_1)"""
    data = np.loadtxt(tester_file, delimiter=',', dtype=str, unpack=False)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    timestamps = data[:, 0].astype(np.int64)
    scores = data[:, 1].astype(np.float64)
    labels_str = data[:, 2]

    # "Anomaly" → 1, "Normaly" → 0
    labels = np.array([1 if l == 'Anomaly' else 0 for l in labels_str])

    return timestamps, scores, labels


def evaluate_service(svc_info, bf_min=-200, bf_max=50, bf_step=0.5, pot_level=0.003):
    """对单个服务做完整评估"""
    svc_name = svc_info['name']
    tester_file = svc_info['tester_file']

    timestamps, scores, labels = load_tester_output(tester_file)

    n_total = len(scores)
    n_anomaly = labels.sum()
    ground_truth_pct = 100 * n_anomaly / max(n_total, 1)

    result = {
        'service': svc_name,
        'n_samples': n_total,
        'n_anomaly': int(n_anomaly),
        'anomaly_pct': ground_truth_pct,
        'mean_score': float(scores.mean()),
        'std_score': float(scores.std()),
        'min_score': float(scores.min()),
        'max_score': float(scores.max()),
    }

    # ── POT 评估 ──
    try:
        pot_result = pot_eval(scores, scores, labels, q=1e-3, level=pot_level)
        result.update(pot_result)
    except Exception as e:
        print(f'  ⚠️  POT 失败: {e}')
        result.update({'pot-f1': -1, 'pot-precision': -1, 'pot-recall': -1,
                       'pot-threshold': -1, 'pot-latency': -1})

    # ── 暴力搜索最佳 F1 ──
    try:
        step_num = int(abs(bf_max - bf_min) / bf_step)
        (best_f1, precision, recall, tp, tn, fp, fn, latency), best_th = \
            bf_search(scores, labels, start=bf_min, end=bf_max, step_num=step_num, verbose=False)
        result.update({
            'bf-f1': best_f1,
            'bf-precision': precision,
            'bf-recall': recall,
            'bf-TP': int(tp),
            'bf-TN': int(tn),
            'bf-FP': int(fp),
            'bf-FN': int(fn),
            'bf-threshold': best_th,
            'bf-latency': latency,
        })
    except Exception as e:
        print(f'  ⚠️  BF 搜索失败: {e}')
        result.update({'bf-f1': -1, 'bf-precision': -1, 'bf-recall': -1,
                       'bf-threshold': -1})

    return result


def get_checkpoint_file(svc_name, model_root, epochs=50, params=None):
    """获取指定 epoch 的 checkpoint 文件路径，用于提取配置"""
    if params is None:
        params = 'catdim5_zdim10_cdim20_hdim20_winsize1_T20_l1'
    ckpt_path = os.path.join(model_root, svc_name, f'{params}_epochs{epochs}.pth')
    return ckpt_path


def plot_results(all_results, output_dir):
    """生成汇总图表"""
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. F1 柱状图（BF + POT 对比） ──
    services = [r['service'] for r in all_results]
    bf_f1 = [r.get('bf-f1', 0) for r in all_results]
    pot_f1 = [r.get('pot-f1', 0) for r in all_results]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(services))
    w = 0.35
    bars1 = ax.bar(x - w/2, bf_f1, w, label='BF Search (best F1)', color='steelblue')
    bars2 = ax.bar(x + w/2, pot_f1, w, label='POT (auto threshold)', color='coral')
    ax.set_xlabel('Service')
    ax.set_ylabel('F1 Score')
    ax.set_title('SGmVRNN 异常检测 F1 分数（每个服务独立模型）')
    ax.set_xticks(x)
    ax.set_xticklabels(services, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    # 在柱子上标注数值
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'f1_comparison.png'), dpi=150)
    plt.close()
    print(f'  📊 F1 对比图: {output_dir}/f1_comparison.png')

    # ── 2. Precision / Recall 散点图 ──
    fig, ax = plt.subplots(figsize=(10, 8))
    for r in all_results:
        bf_p = r.get('bf-precision', 0)
        bf_r = r.get('bf-recall', 0)
        name = r['service']
        if bf_p > 0 or bf_r > 0:
            ax.scatter(bf_p, bf_r, s=100, c='steelblue', edgecolors='white', zorder=5)
            ax.annotate(name, (bf_p, bf_r), textcoords='offset points',
                       xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_title('Precision vs Recall（BF Search 最佳阈值）')
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # 对角线（F1 等高线）
    f1_levels = [0.2, 0.4, 0.6, 0.8]
    for f1 in f1_levels:
        x_vals = np.linspace(0.01, 1, 100)
        y_vals = f1 * x_vals / (2 * x_vals - f1 + 1e-10)
        valid = (y_vals > 0) & (y_vals <= 1)
        ax.plot(x_vals[valid], y_vals[valid], 'gray', alpha=0.2, linestyle='--')
        ax.text(0.85, f1 * 0.85 / (2 * 0.85 - f1 + 0.01), f'F1={f1:.1f}',
               fontsize=7, color='gray')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'precision_recall.png'), dpi=150)
    plt.close()
    print(f'  📊 PR 散点图: {output_dir}/precision_recall.png')

    # ── 3. 有异常服务的异常分数曲线 ──
    for r in all_results:
        if r['n_anomaly'] == 0:
            continue
        svc_name = r['service']
        tester_file = None
        for svc in service_list:
            if svc['name'] == svc_name:
                tester_file = svc['tester_file']
                break
        if not tester_file or not os.path.exists(tester_file):
            continue

        timestamps, scores, labels = load_tester_output(tester_file)
        threshold = r.get('bf-threshold', r.get('pot-threshold', 0))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # 上：分数 + 阈值
        ax1.plot(timestamps, scores, 'b-', label='Log-Likelihood', linewidth=1)
        if threshold != -1:
            ax1.axhline(y=threshold, color='r', linestyle='--',
                       label=f'Threshold ({threshold:.2f})', linewidth=1.5)
        ax1.fill_between(timestamps, scores.min()-10, threshold,
                         where=(scores < threshold), color='red', alpha=0.15,
                         label='Detected Anomaly')
        # 真实异常区域
        anomaly_regions = get_anomaly_ranges(labels, timestamps)
        for start, end in anomaly_regions:
            ax1.axvspan(start, end, color='orange', alpha=0.2)
        ax1.set_ylabel('Log-Likelihood')
        ax1.set_title(f'{svc_name} — 异常分数曲线 (threshold={threshold:.2f})')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(alpha=0.3)

        # 下：预测 vs 真实
        pred = scores < threshold
        ax2.fill_between(timestamps, 0, labels, step='mid',
                         color='orange', alpha=0.5, label='Ground Truth')
        ax2.fill_between(timestamps, -0.05, pred.astype(float)-0.05, step='mid',
                         color='red', alpha=0.5, label='Predicted')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Label')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Normal', 'Anomaly'])
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'score_{svc_name}.png'), dpi=150)
        plt.close()
        print(f'  📊 异常分数曲线: {output_dir}/score_{svc_name}.png')

    # ── 4. 汇总表截图 ──
    fig, ax = plt.subplots(figsize=(16, max(4, len(all_results) * 0.4 + 2)))
    ax.axis('off')

    col_labels = ['Service', 'Samples', 'Anomaly%', 'BF-F1', 'BF-Prec', 'BF-Recall',
                  'POT-F1', 'POT-Prec', 'POT-Recall', 'Threshold']
    cell_text = []
    for r in all_results:
        cell_text.append([
            r['service'],
            str(r['n_samples']),
            f'{r["anomaly_pct"]:.1f}%',
            f'{r.get("bf-f1", -1):.4f}' if r.get('bf-f1', -1) >= 0 else '-',
            f'{r.get("bf-precision", -1):.4f}' if r.get('bf-precision', -1) >= 0 else '-',
            f'{r.get("bf-recall", -1):.4f}' if r.get("bf-recall", -1) >= 0 else '-',
            f'{r.get("pot-f1", -1):.4f}' if r.get('pot-f1', -1) >= 0 else '-',
            f'{r.get("pot-precision", -1):.4f}' if r.get('pot-precision', -1) >= 0 else '-',
            f'{r.get("pot-recall", -1):.4f}' if r.get('pot-recall', -1) >= 0 else '-',
            f'{r.get("bf-threshold", r.get("pot-threshold", 0)):.2f}',
        ])

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center',
                     cellLoc='center', colWidths=[0.12]*len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f5f5f5')
    ax.set_title(f'SGmVRNN Anomaly Detection — Summary (epoch=50)\n'
                 f'{datetime.now().strftime("%Y-%m-%d %H:%M")}',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'summary_table.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'  📊 汇总表: {output_dir}/summary_table.png')


def get_anomaly_ranges(labels, timestamps, buffer=3):
    """将点标签转换为连续异常区间"""
    ranges = []
    in_anomaly = False
    start = None
    for i in range(len(labels)):
        if labels[i] == 1 and not in_anomaly:
            start = timestamps[i]
            in_anomaly = True
        elif labels[i] == 0 and in_anomaly:
            ranges.append((start, timestamps[max(0, i-1)]))
            in_anomaly = False
    if in_anomaly:
        ranges.append((start, timestamps[-1]))
    return ranges


def print_summary(all_results):
    """打印汇总表格"""
    print()
    print('═' * 90)
    print('📊 SGmVRNN 异常检测评估汇总')
    print('═' * 90)
    print(f'{"Service":25s} {"#Samples":>9s} {"Anomaly":>8s} '
          f'{"BF-F1":>8s} {"BF-P":>7s} {"BF-R":>7s} '
          f'{"POT-F1":>8s} {"POT-P":>7s} {"POT-R":>7s} {"Thresh":>8s}')
    print('-' * 90)

    has_anomaly_bf_f1s = []
    for r in all_results:
        svc = r['service']
        ns = r['n_samples']
        ap = f'{r["anomaly_pct"]:.1f}%' if r['n_anomaly'] > 0 else '-'
        bf_f1 = f'{r.get("bf-f1", -1):.4f}' if r.get('bf-f1', -1) >= 0 else '  N/A'
        bf_p = f'{r.get("bf-precision", -1):.3f}' if r.get('bf-precision', -1) >= 0 else '  N/A'
        bf_r = f'{r.get("bf-recall", -1):.3f}' if r.get('bf-recall', -1) >= 0 else '  N/A'
        pot_f1 = f'{r.get("pot-f1", -1):.4f}' if r.get('pot-f1', -1) >= 0 else '  N/A'
        pot_p = f'{r.get("pot-precision", -1):.3f}' if r.get('pot-precision', -1) >= 0 else '  N/A'
        pot_r = f'{r.get("pot-recall", -1):.3f}' if r.get('pot-recall', -1) >= 0 else '  N/A'
        th = r.get('bf-threshold', r.get('pot-threshold', 0))
        print(f'{svc:25s} {ns:>9d} {ap:>8s} {bf_f1:>8s} {bf_p:>7s} {bf_r:>7s} '
              f'{pot_f1:>8s} {pot_p:>7s} {pot_r:>7s} {th:>8.2f}')

        if r['n_anomaly'] > 0 and r.get('bf-f1', -1) >= 0:
            has_anomaly_bf_f1s.append(r['bf-f1'])

    print('-' * 90)
    if has_anomaly_bf_f1s:
        print(f'有异常服务的平均 BF-F1: {np.mean(has_anomaly_bf_f1s):.4f}')
        print(f'有异常服务数量: {len(has_anomaly_bf_f1s)}/{len(all_results)}')
    print()


def main():
    parser = argparse.ArgumentParser(description='SGmVRNN 批量评估')
    parser.add_argument('--log_root', default='log_tester',
                        help='tester.py 输出根目录')
    parser.add_argument('--model_root', default='model',
                        help='checkpoint 根目录（用于配置信息）')
    parser.add_argument('--output_dir', default='eval_results',
                        help='评估结果输出目录')
    parser.add_argument('--services', nargs='*', default=None,
                        help='指定服务（默认全部）')
    parser.add_argument('--epochs', type=int, default=50,
                        help='使用第几个 epoch 的 checkpoint')
    parser.add_argument('--bf_min', type=float, default=-200,
                        help='BF 搜索下限')
    parser.add_argument('--bf_max', type=float, default=50,
                        help='BF 搜索上限')
    parser.add_argument('--bf_step', type=float, default=0.5,
                        help='BF 搜索步长')
    parser.add_argument('--pot_level', type=float, default=0.003,
                        help='POT 初始阈值概率')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='生成可视化图表')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_root = os.path.join(script_dir, args.log_root)
    model_root = os.path.join(script_dir, args.model_root)
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f'📂 测试输出目录: {log_root}')
    print(f'📁 模型目录: {model_root}')
    print(f'📁 评估输出: {output_dir}')

    # 扫描测试结果
    all_svc_list = find_tester_outputs(log_root, epochs=args.epochs)
    print(f'📋 发现 {len(all_svc_list)} 个服务的测试结果: '
          f'{", ".join(s["name"] for s in all_svc_list)}')

    # 筛选服务
    global service_list
    if args.services:
        service_list = [s for s in all_svc_list if s['name'] in args.services]
        missing = [s for s in args.services if s not in [x['name'] for x in all_svc_list]]
        if missing:
            print(f'⚠️  找不到以下服务的测试结果: {missing}')
    else:
        service_list = all_svc_list

    if not service_list:
        print('❌ 没有找到测试结果，请先运行 batch_test.py')
        sys.exit(1)

    # ── 逐个评估 ──
    all_results = []
    for svc in service_list:
        print(f'\n🔍 评估 {svc["name"]} ...')
        print(f'   文件: {os.path.basename(svc["tester_file"])}')
        try:
            result = evaluate_service(svc, bf_min=args.bf_min,
                                      bf_max=args.bf_max,
                                      bf_step=args.bf_step,
                                      pot_level=args.pot_level)
            all_results.append(result)

            if result['n_anomaly'] > 0:
                print(f'   样本: {result["n_samples"]} '
                      f'(异常: {result["n_anomaly"]}, '
                      f'{result["anomaly_pct"]:.1f}%)')
                print(f'   BF:  F1={result.get("bf-f1", -1):.4f}  '
                      f'P={result.get("bf-precision", -1):.4f}  '
                      f'R={result.get("bf-recall", -1):.4f}  '
                      f'阈值={result.get("bf-threshold", -1):.2f}')
                pot_f1 = result.get('pot-f1', -1)
                if pot_f1 >= 0:
                    print(f'   POT: F1={pot_f1:.4f}  '
                          f'P={result.get("pot-precision", -1):.4f}  '
                          f'R={result.get("pot-recall", -1):.4f}  '
                          f'阈值={result.get("pot-threshold", -1):.2f}')
            else:
                print(f'   样本: {result["n_samples"]} (无异常)  '
                      f'分数范围: {result["min_score"]:.2f} ~ {result["max_score"]:.2f}')
        except Exception as e:
            print(f'   ❌ 评估失败: {e}')
            import traceback
            traceback.print_exc()

    # ── 输出汇总 ──
    print_summary(all_results)

    # ── 保存 CSV ──
    csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        headers = ['service', 'n_samples', 'n_anomaly', 'anomaly_pct',
                   'bf-f1', 'bf-precision', 'bf-recall', 'bf-threshold',
                   'bf-latency', 'pot-f1', 'pot-precision', 'pot-recall',
                   'pot-threshold', 'pot-latency']
        f.write(','.join(headers) + '\n')
        for r in all_results:
            row = [r.get(h, '') for h in headers]
            f.write(','.join(str(v) for v in row) + '\n')
    print(f'📄 CSV 结果: {csv_path}')

    # ── 可视化 ──
    if args.plot and all_results:
        print('\n🎨 生成图表...')
        plot_results(all_results, output_dir)

    print(f'\n✅ 评估完成！结果目录: {output_dir}')


if __name__ == '__main__':
    main()
