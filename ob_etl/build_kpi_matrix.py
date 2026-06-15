#!/usr/bin/env python3
"""
build_kpi_matrix.py — Online Boutique Prometheus CSV → 每服务 KPI 矩阵

从 Online Boutique 的 Prometheus 导出 CSV 构建每服务（per-pod）多变量 KPI 时间序
列矩阵，同时生成异常标签文件。输出格式可直接喂给 SGmVRNN 预处理管道。

流程：
  1. 加载每个场景 9 个 Prometheus CSV
  2. 对每个 metric 做 pivot（行=时间戳，列=pod）
  3. 合并所有 metric → 每个 pod 一张 KPI 宽表
  4. 对齐时间戳到 15s 均匀间隔
  5. 按已知故障时间窗口打异常标签
  6. 输出 KPI CSV + 标签 CSV（兼容 SGmVRNN 预处理）

用法:
    python build_kpi_matrix.py \\
        --data_dir ../OnlineBoutique_data/data \\
        --output_dir ../OnlineBoutique_data/processed \\
        [--resample 15s]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ── 故障注入时间窗口（UTC，实验时间为 UTC+8，已转为 UTC）──────────────
# 实验日志记录的是北京时间 (UTC+8)，Prometheus 返回 Unix UTC
# 以下已从北京时间转为 UTC：
#   F1 pod-kill:       16:10:22~16:10:50 CST = 08:10:22~08:10:50 UTC
#   F2 net-delay:      16:14:34~16:20:56 CST = 08:14:34~08:20:56 UTC
#   F3 cpu-stress:     16:25:36~16:30:43 CST = 08:25:36~08:30:43 UTC
FAULT_WINDOWS = {
    'fault_pod_kill': {
        'services': ['frontend'],
        'anomaly': (datetime(2026, 6, 2, 8, 10, 22, tzinfo=timezone.utc),
                    datetime(2026, 6, 2, 8, 10, 50, tzinfo=timezone.utc)),
    },
    'fault_net_delay': {
        'services': ['checkoutservice'],
        'anomaly': (datetime(2026, 6, 2, 8, 14, 34, tzinfo=timezone.utc),
                    datetime(2026, 6, 2, 8, 20, 56, tzinfo=timezone.utc)),
    },
    'fault_cpu_stress': {
        'services': ['recommendationservice'],
        'anomaly': (datetime(2026, 6, 2, 8, 25, 36, tzinfo=timezone.utc),
                    datetime(2026, 6, 2, 8, 30, 43, tzinfo=timezone.utc)),
    },
}

# ── 每个服务提取的 KPI 列表 ──────────────────────────────────────────
# 每个 tuple: (metric_short_name, csv_column_hint, transform_fn_name)
# cpu/restarts/ready/status 都是 per-pod 粒度的；network 是节点级，跳过
KPI_DEFS = [
    'cpu_usage_rate',
    'memory_working_set',
    'disk_read_rate',
    'disk_write_rate',
    'pod_restarts',
    'pod_ready',      # 取 condition='true' 的 value
    'pod_status',     # 取 phase='Running' 的 value → is_running
]

# ── 时间戳转换 ──────────────────────────────────────────────────────
# CSV 中的 timestamp 是 Unix 秒（浮点），转 datetime 用
UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def unix_to_dt(ts_str):
    """Prometheus CSV 的 Unix 时间戳 → 带时区 datetime"""
    return datetime.fromtimestamp(float(ts_str), tz=timezone.utc)


# ═══════════════════════════════════════════════════════════════════
#  1.  读取单个 Prometheus CSV 并透视成宽表（行=时间，列=pod）
# ═══════════════════════════════════════════════════════════════════

def load_metric_csv(scenario_dir, metric_name, file_prefix):
    """
    读一个 metric 的 CSV，返回宽表 DataFrame(index=datetime, columns=pod)。
    特殊处理 pod_status（多 phase 行）和 pod_ready（多 condition 行）。

    参数:
        scenario_dir: 场景目录路径
        metric_name : KPI 简称（如 cpu_usage_rate）
        file_prefix : CSV 文件名前缀（从 get_file_prefix 获取）
    """
    fname = f'{file_prefix}_{metric_name}.csv'
    fpath = os.path.join(scenario_dir, fname)
    if not os.path.exists(fpath):
        return None

    df = pd.read_csv(fpath)

    if df.empty:
        return None

    # timestamp → datetime index
    df['timestamp'] = df['timestamp'].apply(unix_to_dt)

    # ── 根据 metric 类型选择 pivot 方式 ──
    if metric_name == 'pod_status':
        # 多行（每 phase 一行），只取 Running phase
        mask = df.get('phase', '') == 'Running'
        pdf = df[mask].copy()
        if pdf.empty:
            return None
        pdf['value'] = pdf['value'].astype(float)
        # 可能有重复的时间戳（不同 uid），取 mean
        pivot = pdf.pivot_table(index='timestamp', columns='pod',
                                values='value', aggfunc='mean')

    elif metric_name == 'pod_ready':
        # 多行（condition=true/false/unknown），只取 condition='true'
        mask = df.get('condition', '') == 'true'
        pdf = df[mask].copy()
        if pdf.empty:
            return None
        pdf['value'] = pdf['value'].astype(float)
        pivot = pdf.pivot_table(index='timestamp', columns='pod',
                                values='value', aggfunc='mean')

    else:
        # 简单 metric：直接 pivot
        df['value'] = df['value'].astype(float)
        pivot = df.pivot_table(index='timestamp', columns='pod',
                               values='value', aggfunc='mean')

    # 去掉空列（全 NaN）
    pivot = pivot.dropna(axis=1, how='all')
    return pivot


def get_file_prefix(scenario_dir, metric_name='cpu_usage_rate'):
    """探测场景目录中 CSV 文件的公共前缀（export_prometheus.py --label 参数决定的）"""
    if not os.path.isdir(scenario_dir):
        return None
    for fname in sorted(os.listdir(scenario_dir)):
        if fname.endswith(f'{metric_name}.csv'):
            # e.g. "pod_kill_cpu_usage_rate.csv" → "pod_kill"
            return fname.replace(f'_{metric_name}.csv', '')
    return None


def discover_pods(data_dir, scenario_labels):
    """遍历所有场景，收集全部的 pod 列表，返回 {pod_name: service_name} 映射"""
    pods = set()
    for label in scenario_labels:
        sdir = os.path.join(data_dir, label)
        if not os.path.isdir(sdir):
            continue
        prefix = get_file_prefix(sdir)
        if prefix is None:
            continue
        p = load_metric_csv(sdir, 'cpu_usage_rate', prefix)
        if p is not None:
            pods.update(p.columns.tolist())

    # 去掉 pod 后缀中的随机 ID，提取服务名
    # K8s pod 名格式: <service>-<deployment_hash>-<pod_hash>
    # 其中 pod_hash 固定 5 字符，deployment_hash 长度可变（8-10）
    pod_to_service = {}
    for p in sorted(pods):
        # 从末尾去掉两个 "-xxx" 段，前面就是服务名
        idx = p.rfind('-')
        if idx > 0:
            idx2 = p[:idx].rfind('-')
            if idx2 > 0:
                svc = p[:idx2]
            else:
                svc = p
        else:
            svc = p
        pod_to_service[p] = svc
    return pod_to_service


# ═══════════════════════════════════════════════════════════════════
#  2.  合并所有 metric → 每 pod 一张宽表
# ═══════════════════════════════════════════════════════════════════

def build_per_pod_matrix(data_dir, scenario_label, pod_to_service,
                         resample_rule='15s'):
    """
    对一个场景目录，构建每 pod 的 KPI 矩阵 + 标签。

    返回: {service_name: DataFrame(index=datetime, columns=[kpi1,kpi2,...])}
           labels: {service_name: Series(label, index=datetime)}
    """
    scenario_dir = os.path.join(data_dir, scenario_label)
    if not os.path.isdir(scenario_dir):
        return {}, {}

    # 探测文件前缀（目录名 'fault_pod_kill' → 文件前缀 'pod_kill'）
    file_prefix = get_file_prefix(scenario_dir)
    if file_prefix is None:
        print(f'  ⚠️  无法探测文件前缀: {scenario_label}')
        return {}, {}

    print(f'  加载 {scenario_label} (prefix={file_prefix}) ...')

    # 1) 加载所有 metric
    metric_pivots = {}
    for metric in KPI_DEFS:
        piv = load_metric_csv(scenario_dir, metric, file_prefix)
        if piv is not None:
            metric_pivots[metric] = piv

    if not metric_pivots:
        print(f'  ⚠️  没有成功加载任何 metric')
        return {}, {}

    # 2) 对每个 pod，合并所有 metric
    all_pods = set()
    for piv in metric_pivots.values():
        all_pods.update(piv.columns.tolist())

    # 按 pod 分组
    pod_matrices = {}
    for pod in all_pods:
        svc = pod_to_service.get(pod, pod)
        dfs = []
        common_cols = set()
        for metric, piv in metric_pivots.items():
            if pod in piv.columns:
                col = piv[pod].copy()
                # Ensure it's a DataFrame
                col.name = metric
                dfs.append(col)
                common_cols.add(metric)

        if not dfs:
            continue

        # 3) 合并并重采样到均匀间隔
        merged = pd.concat(dfs, axis=1)
        merged.index.name = 'timestamp'
        merged = merged.resample(resample_rule).mean()

        # 如果是 restarts（累计计数），填充用 ffill（不会突变）
        if 'pod_restarts' in merged.columns:
            merged['pod_restarts'] = merged['pod_restarts'].ffill()

        # 线性插值其余 NaN（小间隙）
        merged = merged.interpolate(method='linear', limit=6)

        # 丢弃首尾仍有 NaN 的行
        merged = merged.dropna()

        if merged.empty:
            continue

        # 重命名列
        merged.columns = [f'{svc}_{c}' for c in merged.columns]

        # 注意：一个 pod 一个 service，但 loadgenerator 和 redis-cart 也是独立 entity
        pod_matrices[pod] = merged

    # 4) 生成标签（按 service 分组）
    # 对于非故障场景，标签全 0
    # 对于故障场景，在 anomaly 窗口内标 1
    labels = {}
    if scenario_label in FAULT_WINDOWS:
        fw = FAULT_WINDOWS[scenario_label]
        affected_svcs = set(fw['services'])
        anom_start, anom_end = fw['anomaly']
        for pod, mat in pod_matrices.items():
            svc = pod_to_service.get(pod, pod)
            if svc in affected_svcs:
                lbl = pd.Series(0, index=mat.index, name='label', dtype=int)
                lbl[(mat.index >= anom_start) & (mat.index <= anom_end)] = 1
            else:
                lbl = pd.Series(0, index=mat.index, name='label', dtype=int)
            labels[pod] = lbl
    else:
        # baseline 场景 → 全 0
        for pod, mat in pod_matrices.items():
            labels[pod] = pd.Series(0, index=mat.index, name='label', dtype=int)

    # 再按 service 名分组（多个 pod 副本 → 选一个，或取平均）
    # 简化：直接返回 per-pod 结果，由 downstream 决定
    return pod_matrices, labels


# ═══════════════════════════════════════════════════════════════════
#  3.  输出
# ═══════════════════════════════════════════════════════════════════

def save_per_service(output_dir, per_pod_matrices, per_pod_labels):
    """按 service 分组保存，每个 service 一个目录"""
    # 先把 pod 映射到 service
    pod_to_svc = {}
    for pod in per_pod_matrices:
        tokens = pod.rsplit('-', 2)
        if len(tokens) >= 3 and len(tokens[-1]) == 5 and len(tokens[-2]) == 9:
            pod_to_svc[pod] = tokens[0]
        else:
            pod_to_svc[pod] = pod

    # 按 service 分组
    svc_groups = defaultdict(list)
    for pod in per_pod_matrices:
        svc_groups[pod_to_svc[pod]].append(pod)

    for svc, pods in svc_groups.items():
        svc_dir = os.path.join(output_dir, svc)
        os.makedirs(svc_dir, exist_ok=True)
        print(f'  📁 {svc}/ ({len(pods)} pod(s))')

        # 取第一个 pod（简化处理），或者如果有多个副本取平均
        if len(pods) == 1:
            mat = per_pod_matrices[pods[0]]
            lbl = per_pod_labels.get(pods[0], None)
        else:
            # 多个副本取均值（它们应该是同一个服务的不同实例）
            mats = [per_pod_matrices[p] for p in pods]
            # align by index
            combined = pd.concat(mats, axis=1).groupby(level=0, axis=1).mean()
            # 列名简化
            combined.columns = [f'{svc}_{c.split("_",1)[1] if "_" in c else c}'
                                for c in combined.columns]
            mat = combined
            # 标签：如果任何副本异常则标记异常
            lbls = [per_pod_labels.get(p, None) for p in pods]
            lbls = [l for l in lbls if l is not None]
            if lbls:
                lbl = pd.concat(lbls, axis=1).max(axis=1).astype(int)
                lbl.name = 'label'
            else:
                lbl = None

        # 保持列名简洁：去掉服务名前缀（因为已经在目录名里了）
        mat.columns = [c.split('_', 1)[1] if '_' in c else c
                       for c in mat.columns]

        # 保存 KPI matrix
        kpi_path = os.path.join(svc_dir, 'kpi_matrix.csv')
        mat.to_csv(kpi_path)
        print(f'    ✅ KPI matrix: {mat.shape} → {kpi_path}')

        if lbl is not None:
            lbl_path = os.path.join(svc_dir, 'labels.csv')
            lbl.to_csv(lbl_path, header=['label'])
            anom_count = lbl.sum()
            print(f'    ✅ Labels: {len(lbl)} rows, {anom_count} anomalies → {lbl_path}')


# ═══════════════════════════════════════════════════════════════════
#  4.  场景合并：baseline 做训练，fault 场景做测试
# ═══════════════════════════════════════════════════════════════════

def merge_train_test(output_dir):
    """
    合并所有场景到 per-service 的 train/test 目录。

    约定：
      - baseline → train （标签全 0）
      - fault_*  → test （按窗口标 1）
      - 每个 service 一个目录：
          {svc}/train/kpi_matrix.csv, labels.csv
          {svc}/test/kpi_matrix.csv, labels.csv
    """
    print('\n═══════════════════════════════════════════')
    print('合并 train / test 数据...')
    print('═══════════════════════════════════════════')

    services = set()
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and not item.startswith('_'):
            services.add(item)

    for svc in sorted(services):
        svc_dir = os.path.join(output_dir, svc)
        print(f'\n  🔄 {svc}')

        # 读取该 service 下所有场景的 CSV
        all_kpis = []
        all_labels = []
        train_kpis = []
        train_labels = []
        test_kpis = []
        test_labels = []

        for fname in sorted(os.listdir(svc_dir)):
            if not fname.endswith('.csv'):
                continue
            fpath = os.path.join(svc_dir, fname)
            # 跳过已处理好的文件
            if fname in ('kpi_matrix.csv', 'labels.csv'):
                continue

            # 文件名格式: {scenario_label}__kpi.csv / {label}__labels.csv
            if '__kpi.csv' in fname:
                scenario = fname.replace('__kpi.csv', '')
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                all_kpis.append((scenario, df))
            elif '__labels.csv' in fname:
                scenario = fname.replace('__labels.csv', '')
                lbl = pd.read_csv(fpath, index_col=0, parse_dates=True)
                all_labels.append((scenario, lbl))

        # 训练 = baseline, 测试 = fault_*
        for scenario, df in all_kpis:
            out_path = os.path.join(svc_dir, f'{scenario}__kpi.csv')

        train_data = None
        train_lbl = None
        test_data = {}
        test_lbl = {}

        for scenario, df in all_kpis:
            if scenario == 'baseline':
                train_data = df
                # 找对应的标签
                for s, l in all_labels:
                    if s == scenario:
                        train_lbl = l
                        break
                if train_lbl is None:
                    train_lbl = pd.Series(0, index=df.index, name='label')
            elif scenario.startswith('fault_'):
                test_data[scenario] = df
                for s, l in all_labels:
                    if s == scenario:
                        test_lbl[scenario] = l
                        break

        # 保存 train
        train_dir = os.path.join(svc_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)
        if train_data is not None:
            train_data.to_csv(os.path.join(train_dir, 'kpi_matrix.csv'))
            print(f'    🚆 Train: {train_data.shape}')
        if train_lbl is not None:
            train_lbl.to_csv(os.path.join(train_dir, 'labels.csv'), header=['label'])

        # 保存 test（合并所有 fault 场景）
        test_dir = os.path.join(svc_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)
        if test_data:
            combined_test = []
            combined_lbl = []
            for scenario in sorted(test_data.keys()):
                df = test_data[scenario]
                df.index.name = 'timestamp'
                combined_test.append(df)
                lbl = test_lbl.get(scenario)
                if lbl is not None:
                    lbl.index.name = 'timestamp'
                    combined_lbl.append(lbl)
                else:
                    combined_lbl.append(
                        pd.Series(0, index=df.index, name='label'))

            test_concat = pd.concat(combined_test)
            test_concat = test_concat[~test_concat.index.duplicated(keep='first')]
            test_concat = test_concat.sort_index()
            # 填充 NaN（不同场景的 KPI 列可能不完全对齐）
            test_concat = test_concat.interpolate(method='linear', limit=6)
            test_concat = test_concat.ffill().bfill()
            test_concat = test_concat.fillna(0.0)  # 最后保底
            test_concat.to_csv(os.path.join(test_dir, 'kpi_matrix.csv'))
            print(f'    🧪 Test:  {test_concat.shape}')

            lbl_concat = pd.concat(combined_lbl, axis=0)
            lbl_concat = lbl_concat[~lbl_concat.index.duplicated(keep='first')]
            lbl_concat = lbl_concat.sort_index()
            # Ensure single column named 'label'
            lbl_concat = lbl_concat.squeeze('columns').to_frame('label') \
                if isinstance(lbl_concat, pd.Series) else lbl_concat
            lbl_concat.to_csv(os.path.join(test_dir, 'labels.csv'))
            total_anom = int(lbl_concat.iloc[:, 0].sum())
            print(f'    🏷️  Labels: {total_anom} anomalies from '
                  f'{", ".join(sorted(test_data.keys()))}')


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Online Boutique Prometheus CSV → 每服务 KPI 矩阵')
    parser.add_argument('--data_dir', default='../OnlineBoutique_data/data',
                        help='Online Boutique 数据根目录 (含 baseline/, fault_*/)')
    parser.add_argument('--output_dir', default='../OnlineBoutique_data/processed',
                        help='输出目录')
    parser.add_argument('--resample', default='15s',
                        help='重采样间隔（默认 15s）')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 发现所有场景
    scenario_labels = []
    for item in sorted(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, item)):
            scenario_labels.append(item)

    print(f'发现场景: {scenario_labels}')
    print(f'输出目录: {output_dir}')

    # 1) 发现所有 pod
    pod_to_service = discover_pods(data_dir, scenario_labels)
    print(f'\n发现 {len(pod_to_service)} 个 pod:')
    for pod, svc in sorted(pod_to_service.items()):
        print(f'   {pod:45s} → {svc}')

    # 2) 对每个场景构建 KPI 矩阵
    all_matrices = {}   # {scenario: {pod: DataFrame}}
    all_labels = {}     # {scenario: {pod: Series}}

    for label in scenario_labels:
        mats, lbls = build_per_pod_matrix(
            data_dir, label, pod_to_service,
            resample_rule=args.resample)
        all_matrices[label] = mats
        all_labels[label] = lbls

    # 3) 按场景保存（合并同服务的多个 pod 副本）
    print('\n═══════════════════════════════════════════')
    print('保存每服务 KPI 矩阵...')
    print('═══════════════════════════════════════════')

    for label in scenario_labels:
        print(f'\n  📁 {label}/')
        mats = all_matrices.get(label, {})
        lbls = all_labels.get(label, {})

        # 按 service 分组（一个 service 可能有多个 pod 副本）
        svc_pods = defaultdict(list)
        for pod in mats:
            svc = pod_to_service.get(pod, pod)
            svc_pods[svc].append(pod)

        for svc, pods in svc_pods.items():
            svc_dir = os.path.join(output_dir, svc)
            os.makedirs(svc_dir, exist_ok=True)

            if len(pods) == 1:
                # 单 pod：直接保存
                mat = mats[pods[0]]
                mat = mat.loc[:, ~mat.columns.duplicated()]
                lbl = lbls.get(pods[0])
            else:
                # 多 pod：取 KPI 均值（列并集，NaN 由下游处理）
                print(f'    ⚡ 合并 {len(pods)} 个 {svc} 副本...')
                # 取所有列名的并集
                all_cols = pd.Index([])
                for p in pods:
                    m = mats[p].loc[:, ~mats[p].columns.duplicated()]
                    all_cols = all_cols.union(m.columns)
                # 对齐到统一列集
                stack = [mats[p].loc[:, ~mats[p].columns.duplicated()].reindex(columns=all_cols)
                         for p in pods]
                mat = pd.concat(stack).groupby(level=0).mean()
                mat = mat.sort_index()

                # 标签合并
                all_lbls = [lbls.get(p) for p in pods if lbls.get(p) is not None]
                if all_lbls:
                    lbl = pd.concat(all_lbls, axis=1).max(axis=1).astype(int)
                    lbl.name = 'label'
                else:
                    lbl = None

            # 保存场景 KPI 矩阵
            kpi_path = os.path.join(svc_dir, f'{label}__kpi.csv')
            mat.to_csv(kpi_path)

            # 保存标签
            if lbl is not None:
                lbl_path = os.path.join(svc_dir, f'{label}__labels.csv')
                # lbl is pd.Series, convert to DataFrame for header
                lbl.to_frame('label').to_csv(lbl_path)

            lbl_str = f'  label({int(lbl.sum())} anomalies)' if lbl is not None else ''
            print(f'    {svc}: KPI {mat.shape}{lbl_str}')

    # 4) 合并 train/test
    merge_train_test(output_dir)

    # 5) 打印汇总
    print('\n═══════════════════════════════════════════')
    print('✅ 完成！数据已保存到:', output_dir)
    print('═══════════════════════════════════════════')
    print('\n每个 service 目录结构:')
    print(f'  {output_dir}/<service>/')
    print(f'    train/kpi_matrix.csv     ← 训练数据（baseline, 全正常）')
    print(f'    train/labels.csv         ← 训练标签（全 0）')
    print(f'    test/kpi_matrix.csv      ← 测试数据（所有 fault 场景合并）')
    print(f'    test/labels.csv          ← 测试标签（按故障窗口标 1）')

    # 统计 KPI 数量
    print('\n各服务 KPI 维度:')
    for svc in sorted(os.listdir(output_dir)):
        svc_dir = os.path.join(output_dir, svc)
        train_csv = os.path.join(svc_dir, 'train', 'kpi_matrix.csv')
        if os.path.exists(train_csv):
            df = pd.read_csv(train_csv, index_col=0)
            print(f'  {svc:30s}: {df.shape[1]} KPIs, {df.shape[0]} 训练样本')


if __name__ == '__main__':
    main()
