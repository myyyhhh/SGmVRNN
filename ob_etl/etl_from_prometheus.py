#!/usr/bin/env python3
"""
etl_from_prometheus.py — 带 3 次重复实验支持的 Prometheus CSV → KPI 矩阵

分两步：
  1. 对每个 fault 场景的 r1/r2/r3，用对应时间窗口打标签
  2. 合并 baseline 做训练集，每个 run 独立做测试集

输出:
  OnlineBoutique_data/processed_v2/
    baseline_normal/{service}/train/   (全正常, 用于训练)
    cpu_stress_r1/{service}/test/
    cpu_stress_r2/{service}/test/
    cpu_stress_r3/{service}/test/
    pod_kill_r1/{service}/test/
    ...

用法:
    python ob_etl/etl_from_prometheus.py
"""
import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ── KPI 定义（同 build_kpi_matrix.py） ──────────────────────────────
KPI_DEFS = [
    'cpu_usage_rate', 'memory_working_set', 'disk_read_rate', 'disk_write_rate',
    'pod_restarts', 'pod_ready', 'pod_status',
]

# ── 故障时间窗口（2026-06-04 UTC） ──────────────────────────────────
# 实验时序:
#   pod_kill r1: 21:15-21:18 → cpu_stress r1: 21:38-21:44
#   pod_kill r2: 21:51-21:55 → cpu_stress r2: 22:14-22:20
#   pod_kill r3: 22:27-22:31 → cpu_stress r3: 22:51-22:57
FAULT_WINDOWS_2026_06_04 = {
    'pod_kill': {
        'r1': (datetime(2026,6,4,21,15,18,tzinfo=timezone.utc),
               datetime(2026,6,4,21,18,48,tzinfo=timezone.utc)),
        'r2': (datetime(2026,6,4,21,51,38,tzinfo=timezone.utc),
               datetime(2026,6,4,21,55, 8,tzinfo=timezone.utc)),
        'r3': (datetime(2026,6,4,22,27,58,tzinfo=timezone.utc),
               datetime(2026,6,4,22,31,28,tzinfo=timezone.utc)),
    },
    'cpu_stress': {
        'r1': (datetime(2026,6,4,21,38,31,tzinfo=timezone.utc),
               datetime(2026,6,4,21,44,31,tzinfo=timezone.utc)),
        'r2': (datetime(2026,6,4,22,14,51,tzinfo=timezone.utc),
               datetime(2026,6,4,22,20,51,tzinfo=timezone.utc)),
        'r3': (datetime(2026,6,4,22,51,11,tzinfo=timezone.utc),
               datetime(2026,6,4,22,57,11,tzinfo=timezone.utc)),
    },
}

# 目标服务映射
FAULT_TARGET = {
    'cpu_stress': 'recommendationservice',
    'pod_kill': 'frontend',
}

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'OnlineBoutique_data')


def unix_to_dt(ts):
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def load_metric(metric, scenario_dir, file_prefix):
    """加载单个 metric CSV，返回 (timestamp, pod, value) 透视表"""
    fpath = os.path.join(scenario_dir, f'{file_prefix}_{metric}.csv')
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    if df.empty:
        return None
    df['timestamp'] = df['timestamp'].apply(unix_to_dt)

    if metric == 'pod_status':
        mask = df.get('phase', '') == 'Running'
        pdf = df[mask].copy()
        if pdf.empty:
            return None
        pdf['value'] = pdf['value'].astype(float)
        return pdf.pivot_table(index='timestamp', columns='pod', values='value', aggfunc='mean')
    elif metric == 'pod_ready':
        mask = df.get('condition', '') == 'true'
        pdf = df[mask].copy()
        if pdf.empty:
            return None
        pdf['value'] = pdf['value'].astype(float)
        return pdf.pivot_table(index='timestamp', columns='pod', values='value', aggfunc='mean')
    else:
        df['value'] = df['value'].astype(float)
        return df.pivot_table(index='timestamp', columns='pod', values='value', aggfunc='mean')


def extract_pod_service(pod_name):
    """从 pod 名提取服务名: frontend-759775d795-xh68w → frontend"""
    parts = pod_name.rsplit('-', 2)
    if len(parts) >= 3 and len(parts[2]) == 5:
        return parts[0]
    return pod_name


def build_kpi_matrix_for_scenario(scenario_dir, target_service, fault_window=None):
    """
    对一个场景目录，构建目标服务的 KPI 矩阵 + 标签。

    返回: (DataFrame(index=datetime, columns=KPI列), Series(label))
    """
    # 探测文件前缀（目录名 'cpu_stress' → 文件前缀 'cpu_stress_r1'）
    items = [f for f in sorted(os.listdir(scenario_dir)) if f.endswith('_cpu_usage_rate.csv')]
    if not items:
        return None, None
    file_prefix = items[0].replace('_cpu_usage_rate.csv', '')

    # 加载所有 metric
    metrics = {}
    for m in KPI_DEFS:
        piv = load_metric(m, scenario_dir, file_prefix)
        if piv is not None:
            metrics[m] = piv

    if not metrics:
        return None, None

    # 找到目标服务的 pod
    all_pods = set()
    for piv in metrics.values():
        all_pods.update(piv.columns)
    target_pods = [p for p in all_pods if extract_pod_service(p) == target_service]

    if not target_pods:
        print(f'  ⚠️  未找到 {target_service} 的 pod（可选: {[extract_pod_service(p) for p in all_pods][:5]}）')
        return None, None

    # 对每个目标 pod，合并所有 metric
    result_dfs = []
    result_labels = []
    for pod in target_pods:
        dfs = []
        for metric, piv in metrics.items():
            if pod in piv.columns:
                col = piv[pod].copy()
                col.name = metric
                dfs.append(col)
        if not dfs:
            continue

        merged = pd.concat(dfs, axis=1)
        merged.index.name = 'timestamp'
        merged = merged.resample('15s').mean()
        if 'pod_restarts' in merged.columns:
            merged['pod_restarts'] = merged['pod_restarts'].ffill()
        merged = merged.interpolate(method='linear', limit=6).dropna()
        if merged.empty:
            continue
        result_dfs.append(merged)

        # 标签
        lbl = pd.Series(0, index=merged.index, name='label', dtype=int)
        if fault_window:
            anom_start, anom_end = fault_window
            lbl.loc[(merged.index >= anom_start) & (merged.index <= anom_end)] = 1
        result_labels.append(lbl)

    if not result_dfs:
        return None, None

    # 多 pod 副本取均值
    mat = pd.concat(result_dfs).groupby(level=0).mean()
    lab = pd.concat(result_labels).groupby(level=0).max().astype(int)
    return mat, lab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.join(DATA_ROOT, 'data'))
    parser.add_argument('--output_dir', default=os.path.join(DATA_ROOT, 'processed_v2'))
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    baseline_dir = os.path.join(data_dir, 'baseline')
    full_ts_dir = os.path.join(data_dir, 'full_timeseries')

    print(f'📂 数据目录: {data_dir}')
    print(f'📁 输出目录: {output_dir}')

    # ── 1. 基线数据（训练集） ──
    print('\n═══════════════════════════════════════════')
    print('处理基线数据 (训练集)...')
    # 用 full_timeseries （7.8h，包含所有数据，但取故障前的部分做训练）
    full_mat, _ = build_kpi_matrix_for_scenario(full_ts_dir, '__all__')
    if full_mat is None:
        print('⚠️ full_timeseries 加载失败，回退到 baseline')
        baseline_mat, _ = build_kpi_matrix_for_scenario(baseline_dir, '__all__')
        all_pods_df = baseline_mat if baseline_mat is not None else None
    else:
        all_pods_df = full_mat

    # 收集所有服务的名称
    from collections import defaultdict
    all_services = set()
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if not os.path.isdir(item_path):
            continue
        for f in sorted(os.listdir(item_path)):
            if f.endswith('_cpu_usage_rate.csv'):
                fp = os.path.join(item_path, f)
                df = pd.read_csv(fp)
                if 'pod' in df.columns:
                    for p in df['pod'].unique():
                        all_services.add(extract_pod_service(p))
                break

    services_of_interest = ['frontend', 'checkoutservice', 'recommendationservice']

    # 对每个目标服务，分别提取基线数据
    for svc in services_of_interest:
        print(f'\n🔧 提取训练数据: {svc}')
        mats, lbls = [], []
        for scenario_dir in [baseline_dir]:
            mat, lbl = build_kpi_matrix_for_scenario(scenario_dir, svc)
            if mat is not None:
                mats.append(mat)
                lbls.append(lbl)
        if not mats:
            print(f'  ⏭️  {svc}: 基线数据不可用')
            continue

        train_mat = pd.concat(mats).sort_index()
        train_mat = train_mat[~train_mat.index.duplicated(keep='first')]
        train_mat = train_mat.interpolate(limit=6).dropna()

        train_lbl = pd.concat(lbls).sort_index()
        train_lbl = train_lbl[~train_lbl.index.duplicated(keep='first')]

        # 保存
        out = os.path.join(output_dir, 'baseline_normal', svc, 'train')
        os.makedirs(out, exist_ok=True)
        train_mat.to_csv(os.path.join(out, 'kpi_matrix.csv'))
        train_lbl.to_frame('label').to_csv(os.path.join(out, 'labels.csv'))
        print(f'  ✅ {svc}: {train_mat.shape}')

    # ── 2. 按 run 重组输出 ──
    # 每个 fault run 数据点太少（15-25 个），T=20 窗口需要 20+ 点
    # 策略：用正常数据做基底，故障数据做异常段，拼接后产生混合测试集
    print('\n═══════════════════════════════════════════')
    print('按 run 重组输出（训练基底 + 故障测试）...')
    # 每个 run 一个独立目录，包含所有服务的训练数据 + 该 run 目标服务的测试数据
    # 后续 preprocess.py 可直接处理每个 run 目录
    print('\n═══════════════════════════════════════════')
    print('按 run 重组输出...')

    # 收集测试集
    test_sets = {}  # {run_label: {svc: (mat, lbl)}}
    for fault_type, runs in FAULT_WINDOWS_2026_06_04.items():
        target_svc = FAULT_TARGET[fault_type]
        for run_id, window in runs.items():
            scenario_dir = os.path.join(data_dir, f'fault_{fault_type}')
            mat, lbl = build_kpi_matrix_for_scenario(scenario_dir, target_svc, None)
            if mat is None:
                continue
            # 标签：整个 fault run 期间全标异常
            anom_start = mat.index.min()
            anom_end = mat.index.max()
            lbl = pd.Series(0, index=mat.index, name='label', dtype=int)
            lbl.loc[(mat.index >= anom_start) & (mat.index <= anom_end)] = 1
            test_sets[f'{fault_type}_{run_id}'] = {target_svc: {'mat': mat, 'lbl': lbl}}

    # 对每个 run 创建独立目录
    for run_label, svc_data in test_sets.items():
        run_output = os.path.join(output_dir, run_label)
        for svc in services_of_interest:
            train_csv = os.path.join(output_dir, 'baseline_normal', svc, 'train', 'kpi_matrix.csv')
            if not os.path.exists(train_csv):
                continue
            train_dir = os.path.join(run_output, svc, 'train')
            os.makedirs(train_dir, exist_ok=True)
            # 复制训练数据
            train_df = pd.read_csv(train_csv, index_col=0, parse_dates=True)
            train_df.to_csv(os.path.join(train_dir, 'kpi_matrix.csv'))
            pd.Series(0, index=train_df.index, name='label').to_frame('label').to_csv(
                os.path.join(train_dir, 'labels.csv'))

            # 测试数据（如果有）
            if svc in svc_data:
                test_dir = os.path.join(run_output, svc, 'test')
                os.makedirs(test_dir, exist_ok=True)
                # 拼接正常数据做基底（使测试集有混合的 normal+anomaly 样本）
                n_fault = len(svc_data[svc]['mat'])
                n_padding = max(80, n_fault * 2)
                normal_before = train_df[train_df.index < svc_data[svc]['mat'].index.min()].iloc[-n_padding:]
                normal_after = train_df[train_df.index > svc_data[svc]['mat'].index.max()].iloc[:n_padding]
                test_mat = pd.concat([normal_before, svc_data[svc]['mat'], normal_after])
                test_mat = test_mat.sort_index()
                test_mat = test_mat[~test_mat.index.duplicated(keep='first')]

                anom_start = svc_data[svc]['mat'].index.min()
                anom_end = svc_data[svc]['mat'].index.max()
                test_lbl = pd.Series(0, index=test_mat.index, name='label', dtype=int)
                test_lbl.loc[(test_mat.index >= anom_start) & (test_mat.index <= anom_end)] = 1

                test_mat.to_csv(os.path.join(test_dir, 'kpi_matrix.csv'))
                test_lbl.to_frame('label').to_csv(os.path.join(test_dir, 'labels.csv'))

        # 输出结构信息
        print(f'\n  📁 {run_label}/')
        for svc in services_of_interest:
            r = os.path.join(run_output, svc, 'train')
            t = os.path.join(run_output, svc, 'test')
            train_ok = os.path.exists(os.path.join(r, 'kpi_matrix.csv'))
            test_ok = os.path.exists(os.path.join(t, 'kpi_matrix.csv'))
            if test_ok:
                ldf = pd.read_csv(os.path.join(t, 'labels.csv'), index_col=0)
                n_anom = ldf.iloc[:, 0].sum()
                print(f'    {svc:25s} 🚆{train_ok} 🧪{test_ok} (异常={int(n_anom)})')
            else:
                print(f'    {svc:25s} 🚆{train_ok} 🧪{test_ok}')

    # 清理临时 baseline_normal
    import shutil
    shutil.rmtree(os.path.join(output_dir, 'baseline_normal'), ignore_errors=True)

    print('\n═══════════════════════════════════════════')
    print('✅ 完成！输出结构:')
    print('═' * 50)
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = '  ' * level
        if level <= 2:
            print(f'{indent}{os.path.basename(root)}/')
            if level == 2:
                csvs = [f for f in files if f.endswith('.csv')]
                if csvs:
                    print(f'{indent}  ├─ {len(csvs)} CSV files')


if __name__ == '__main__':
    main()
