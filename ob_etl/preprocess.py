#!/usr/bin/env python3
"""
preprocess.py — KPI 矩阵 CSV → SGmVRNN .seq 滑动窗口文件

将 build_kpi_matrix.py 输出的 KPI 矩阵转换为 SGmVRNN 训练/测试可直接加载的
.seq 文件。兼容 data_preprocess_2nd.py 的输出格式。

流程：
  1. 读 KPI matrix CSV (index=timestamp, columns=KPI_1..KPI_n)
  2. 用 T 长度的滑动窗口切分，步长 l
  3. 每个窗口保存为 .seq 文件 (torch.save)
  4. 生成 SGmVRNN 的 KpiReader 可读的数据目录

用法:
    python preprocess.py \\
        --data_dir ../OnlineBoutique_data/processed \\
        --output_dir ../OnlineBoutique_data/processed_seq \\
        --T 20 --l 1

每个 service 的输出结构:
  {output_dir}/{svc}/
    train/
      1.seq, 2.seq, ...
    test/
      1.seq, 2.seq, ...

SGmVRNN 训练:
  python trainer.py --dataset_path {output_dir}/{svc}/train --n {n_KPIs}

SGmVRNN 测试:
  python tester.py  --dataset_path {output_dir}/{svc}/test  --n {n_KPIs}
"""
import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def normalize_kpi(kpi_csv_path, output_norm_path, scaler=None):
    """
    归一化 KPI 数据到 [-1, 1] 区间（SGmVRNN 模型要求）。
    如果 scaler 为 None，从数据拟合（用于训练集）；
    否则用已有的 scaler 变换（用于测试集/验证集）。

    返回: (normalized_df, scaler)
    """
    df = pd.read_csv(kpi_csv_path, index_col=0, parse_dates=True)
    if df.empty:
        return df, scaler

    # 填充 NaN（列对齐问题）
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        df = df.interpolate(method='linear', limit=6) \
               .ffill().bfill().fillna(0.0)
        print(f'      🔧 填充 {nan_count} 个 NaN')

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = scaler.fit_transform(df.values)
    else:
        scaled_values = scaler.transform(df.values)

    normalized = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
    if output_norm_path:
        normalized.to_csv(output_norm_path)

    return normalized, scaler


def kpi_csv_to_samples(kpi_csv_path, label_csv_path, output_dir,
                        T=20, l=1, sample_id_start=1):
    """
    将 KPI matrix CSV 转换为 SGmVRNN 的 .seq 文件。

    参数:
        kpi_csv_path  : KPI matrix CSV (index=datetime, columns=KPI 名)
        label_csv_path: 标签 CSV (index=datetime, column='label', 0/1)
        output_dir    : 输出目录（放 .seq 文件）
        T             : 滑动窗口长度（SGmVRNN 的 T，默认 20）
        l             : 窗口步长（默认 1）
        sample_id_start: 起始编号

    返回:
        写入的 .seq 文件数量
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读 KPI 数据
    kpi_df = pd.read_csv(kpi_csv_path, index_col=0, parse_dates=True)
    if kpi_df.empty:
        print(f'    ⚠️  空 KPI 文件: {kpi_csv_path}')
        return 0

    # 填充 NaN（列不对齐或归一化边界问题）
    nan_count = kpi_df.isna().sum().sum()
    if nan_count > 0:
        kpi_df = kpi_df.interpolate(method='linear', limit=6) \
                         .ffill().bfill().fillna(0.0)
        print(f'    🔧 填充了 {nan_count} 个 NaN')

    # 读标签
    label_df = pd.read_csv(label_csv_path, index_col=0, parse_dates=True)
    label_series = label_df.squeeze('columns')

    # 对齐时间轴（取交集）
    common_idx = kpi_df.index.intersection(label_series.index)
    kpi_df = kpi_df.loc[common_idx]
    label_series = label_series.loc[common_idx]

    # 确保按时间排序
    kpi_df = kpi_df.sort_index()
    label_series = label_series.sort_index()

    n_kpis = kpi_df.shape[1]
    n_timestamps = len(kpi_df)

    print(f'    📊 {n_kpis} KPIs × {n_timestamps} 时间步')

    # 转置为 SGmVRNN 格式: 行=KPI, 列=时间
    # scaled_data: (n_kpis, n_timestamps)
    scaled_data = kpi_df.values.T.astype(np.float64)

    # 时间戳转 SGmVRNN 格式 (YYYYMMDDHHMMSS 字符串)
    ts_strings = [dt.strftime('%Y%m%d%H%M%S') for dt in common_idx]
    raw_ts = np.array([ts_strings])  # (1, n_timestamps)
    raw_label = label_series.values.reshape(1, -1)  # (1, n_timestamps)

    # ── 滑动窗口采样 ──
    # SGmVRNN 的 data_preprocess_2nd 用 l 个偏移量做多组采样
    rectangle_samples = []
    rectangle_labels = []
    rectangle_tss = []

    for j in range(l):
        rect_sample = []
        rect_label = []
        rect_ts = []
        # 窗口滑过时间轴（列），窗口宽度 = n_kpis（对于 w=1 的情况取全部 KPI）
        # 注意: data_preprocess_2nd 中 win_size=36 是指滑动窗口宽度（KPI 维度？）
        # 不对，win_size 是时间维度上的窗口，而 n 是 KPI 维度。
        # 对于我们的情况，w=1 所以每个时间步取所有 KPI，即窗口宽度=1 时间步，包含 n_kpis 维
        #
        # 但 data_preprocess_2nd 的编码有 win_size 作为时间方向的矩形宽度。
        # 因为 w=1, 所以我们跳过矩形聚合，直接做时间切片。
        for i in range(0, n_timestamps - 1, l):
            if i + j <= n_timestamps - 1:
                # 取单列（w=1）：scaled_data[:, i+j] → (n_kpis,)
                rect_sample.append(scaled_data[:, i+j].tolist())
                rect_label.append(raw_label[:, i+j].tolist())
                rect_ts.append(raw_ts[:, i+j].tolist())

        rectangle_samples.append(np.array(rect_sample))
        rectangle_labels.append(np.array(rect_label))
        rectangle_tss.append(np.array(rect_ts))

    # ── 组装 T 长度的序列样本 ──
    sample_id = sample_id_start
    count = 0
    for i in range(len(rectangle_samples)):
        for data_id in range(T, len(rectangle_samples[i])):
            # 取 T 个连续时间步
            kpi_data = rectangle_samples[i][data_id - T:data_id]  # (T, n_kpis)
            kpi_label = rectangle_labels[i][data_id - T:data_id]  # (T, 1)
            kpi_ts = rectangle_tss[i][data_id - T:data_id]  # (T, 1)

            # 添加 w=1 维度 → (T, 1, n_kpis)
            kpi_data = torch.tensor(kpi_data, dtype=torch.float32).unsqueeze(1)

            data = {
                'ts': kpi_ts,
                'label': kpi_label,
                'value': kpi_data,  # (T, 1, n_kpis, 1) if w=1?
            }

            # 实际 SGmVRNN 期望的 value shape: [T, 1, n, w]
            # 我们的 kpi_data 是 (T, 1, n_kpis), 需要加 w 维度
            data['value'] = data['value'].unsqueeze(-1)  # (T, 1, n_kpis, 1)

            out_path = os.path.join(output_dir, f'{sample_id}.seq')
            torch.save(data, out_path)
            sample_id += 1
            count += 1

    print(f'    ✅ 生成 {count} 个 .seq 样本 (T={T}, id={sample_id_start}~{sample_id - 1})')
    return count


def process_service(svc_dir, output_base_dir, T=20, l=1):
    """处理一个 service：归一化 → 滑动窗口 → .seq"""
    svc_name = os.path.basename(svc_dir)
    train_csv = os.path.join(svc_dir, 'train', 'kpi_matrix.csv')
    train_label = os.path.join(svc_dir, 'train', 'labels.csv')
    test_csv = os.path.join(svc_dir, 'test', 'kpi_matrix.csv')
    test_label = os.path.join(svc_dir, 'test', 'labels.csv')

    # 检查数据存在
    train_ok = os.path.exists(train_csv) and os.path.exists(train_label)
    test_ok = os.path.exists(test_csv) and os.path.exists(test_label)

    if not train_ok and not test_ok:
        print(f'  ⏭️  无训练/测试数据: {svc_name}')
        return None

    # 读 KPI 矩阵确定 n
    if train_ok:
        sample_df = pd.read_csv(train_csv, index_col=0)
    else:
        sample_df = pd.read_csv(test_csv, index_col=0)
    n_kpis = sample_df.shape[1]

    # ── 归一化到 [-1, 1] ──
    # 用训练集拟合 scaler 并变换训练集；用同一 scaler 变换测试集
    scaler = None
    train_norm_csv = train_csv.replace('kpi_matrix.csv', 'kpi_norm.csv')
    test_norm_csv = test_csv.replace('kpi_matrix.csv', 'kpi_norm.csv')

    if train_ok:
        _, scaler = normalize_kpi(train_csv, train_norm_csv, scaler=None)
        print(f'    📊 归一化: fit on train')
        if test_ok:
            normalize_kpi(test_csv, test_norm_csv, scaler=scaler)
            print(f'    📊 归一化: transform test')
    elif test_ok:
        _, scaler = normalize_kpi(test_csv, test_norm_csv, scaler=None)
        print(f'    ⚠️  归一化: no train data, fit on test')

    # ── 生成 .seq 文件 ──
    train_count = 0
    test_count = 0

    if train_ok:
        train_out = os.path.join(output_base_dir, svc_name, 'train')
        train_count = kpi_csv_to_samples(train_norm_csv, train_label, train_out, T=T, l=l)
    if test_ok:
        test_out = os.path.join(output_base_dir, svc_name, 'test')
        test_count = kpi_csv_to_samples(test_norm_csv, test_label, test_out, T=T, l=l)

    return {
        'service': svc_name,
        'n_kpis': n_kpis,
        'train_samples': train_count,
        'test_samples': test_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Online Boutique KPI 矩阵 → SGmVRNN .seq 文件')
    parser.add_argument('--data_dir', default='../OnlineBoutique_data/processed',
                        help='build_kpi_matrix.py 的输出目录')
    parser.add_argument('--output_dir', default='../OnlineBoutique_data/processed_seq',
                        help='.seq 文件输出目录')
    parser.add_argument('--T', type=int, default=20,
                        help='滑动窗口长度（默认 20）')
    parser.add_argument('--l', type=int, default=1,
                        help='窗口滑步步长（默认 1）')
    parser.add_argument('--services', nargs='*', default=None,
                        help='只处理指定服务（默认处理所有）')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)

    print(f'📂 数据目录: {data_dir}')
    print(f'📁 输出目录: {output_dir}')
    print(f'⚙️  T={args.T}, l={args.l}')
    print()

    # 发现所有 service
    services = []
    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and not item.startswith('_'):
            if args.services is None or item in args.services:
                services.append(item)

    print(f'发现 {len(services)} 个服务: {", ".join(services)}')
    print()

    results = []
    for svc in services:
        svc_path = os.path.join(data_dir, svc)
        print(f'🔧 处理 {svc} ...')
        try:
            result = process_service(svc_path, output_dir, T=args.T, l=args.l)
            if result:
                results.append(result)
                print(f'  ✅ {result["service"]}: n={result["n_kpis"]}, '
                      f'train={result["train_samples"]}, test={result["test_samples"]}')
            else:
                print(f'  ⏭️  跳过')
        except Exception as e:
            print(f'  ❌ 错误: {e}')
        print()

    # 汇总
    print('═══════════════════════════════════════════')
    print('✅ 预处理完成！')
    print('═══════════════════════════════════════════')
    print()
    print(f'{"Service":30s} {"n_KPIs":>8s} {"Train":>8s} {"Test":>8s}')
    print('-' * 56)
    for r in results:
        print(f'{r["service"]:30s} {r["n_kpis"]:>8d} {r["train_samples"]:>8d} {r["test_samples"]:>8d}')
    print()

    print('训练命令示例:')
    for r in results:
        n = r['n_kpis']
        train_path = os.path.join(output_dir, r['service'], 'train')
        print(f'  python trainer.py --dataset_path {train_path} --n {n} '
              f'--log_path log_trainer/{r["service"]} --checkpoints_path model/{r["service"]}')

    print()
    print('测试命令示例:')
    for r in results:
        n = r['n_kpis']
        test_path = os.path.join(output_dir, r['service'], 'test')
        print(f'  python tester.py --dataset_path {test_path} --n {n} '
              f'--checkpoints_path model/{r["service"]} --start_epoch 20 '
              f'--log_path log_tester/{r["service"]}')


if __name__ == '__main__':
    main()
