# SGmVRNN 论文复现实验

这是软件测试与维护课程的复现论文实验

> **SGmVRNN**: Switching Gaussian-mixture Variational Recurrent Neural Network for Anomaly Detection
>
> 论文发表于 **IEEE INFOCOM 2022**

本仓库是对 INFOCOM 2022 论文《[SGmVRNN: A Switching Gaussian-mixture Variational Recurrent Neural Network for Anomaly Detection in KPI Time Series](https://ieeexplore.ieee.org/document/9796885)》的完整复现实验，在 **Online Boutique 微服务系统**上收集真实 KPI 指标数据，验证 SGmVRNN 的异常检测效果。

---

## 目录

- [模型简介](#模型简介)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [数据集说明](#数据集说明)
- [快速开始](#快速开始)
- [实验结果](#实验结果)
- [引用](#引用)

---

## 模型简介

SGmVRNN（Stochastic Gaussian-mixture VRNN）是一种基于**变分循环神经网络**的无监督异常检测方法，核心创新点包括：

| 组件 | 说明 |
|:----|:------|
| **VAE + RNN 混合架构** | 同时建模时序依赖和概率分布 |
| **Gumbel-Softmax 离散潜变量** | 假设 KPI 模式由离散类别驱动 |
| **高斯连续潜变量** | 在离散类别下细粒度建模 |
| **Log-likelihood 异常检测** | 测试样本的 log-likelihood 越低 → 越可能是异常 |
| **CNN 编码器/解码器** | 在 KPI 维度上提取局部特征 |

### 实验架构

```
┌──────────────────────────────────────────────────┐
│               Online Boutique                     │
│  (12 微服务, minikube + Docker 驱动)              │
│                                                   │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ frontend │  │ checkoutsvc  │  │ recommdsvc │ │
│  │ (pod_kill│  │ (net_delay)  │  │(cpu_stress)│ │
│  └──────────┘  └──────────────┘  └────────────┘ │
│         ... 9 个正常服务 ...                      │
└──────────────────────┬───────────────────────────┘
                       │ Prometheus scrape (15s)
                 ┌─────▼──────┐
                 │ cAdvisor   │ kube-state-metrics
                 │ (CPU/mem/  │ (pod_ready/restarts/
                 │  disk/net) │  pod_status)
                 └─────┬──────┘
                       │ 9 类 KPI
                 ┌─────▼──────┐
                 │Chaos Mesh  │ → F1/F2/F3 故障注入
                 └────────────┘
```

### 故障注入实验

使用 **Chaos Mesh** 注入 3 种故障，共执行 **3 轮重复实验**：

| 编号 | 故障类型 | 目标服务 | 工具 | 持续时间 |
|:----:|:--------|:--------|:----|:-------:|
| F1 | Pod 杀死 | frontend | PodChaos | 瞬间（Kill 后自愈） |
| F2 | 网络延迟 200ms | checkoutservice | NetworkChaos | 5 分钟 |
| F3 | CPU 压力 80% | recommendationservice | StressChaos | 5 分钟 |

### 监控指标

采集 9 类核心 KPI：

| KPI | 来源 | 含义 |
|:----|:----|:-----|
| cpu_usage_rate | cAdvisor | CPU 使用率 |
| memory_working_set | cAdvisor | 工作集内存 |
| disk_read_rate / disk_write_rate | cAdvisor | 磁盘 I/O |
| network_rx_rate / network_tx_rate | cAdvisor | 网络流量 |
| pod_ready / pod_restarts / pod_status | kube-state-metrics | Pod 状态 |

---

## 项目结构

```
SGmVRNN/
├── SGmVRNN/                          # 核心代码（基于原论文实现）
│   ├── SGmVRNN/                      # 模型实现
│   │   ├── model.py                  #   SGmVRNN 模型定义
│   │   ├── trainer.py                #   训练器
│   │   ├── tester.py                 #   测试器
│   │   ├── evaluate_pot.py           #   POT 阈值评估
│   │   ├── spot.py                   #   POT 算法实现
│   │   ├── util.py                   #   工具函数
│   │   └── logger.py                 #   日志工具
│   ├── baselines/                    # 基线方法
│   │   ├── continuous_vrnn.py        #   Continous VRNN
│   │   ├── lstm_ndt.py               #   LSTM-NDT
│   │   ├── vae_no_rnn.py            #   VAE (无 RNN)
│   │   ├── compare_results.py        #   基线结果对比
│   │   ├── batch_experiment.py       #   批量基线实验
│   │   └── baselines_new/            #   新基线实验
│   ├── data_preprocess/              # 数据预处理（SMD 数据集）
│   │   ├── data_preprocess_1st.py
│   │   └── data_preprocess_2nd.py
│   ├── batch_train.py                # 一键批量训练（Online Boutique）
│   ├── batch_test.py                 # 一键批量测试（Online Boutique）
│   ├── evaluate_all.py               # 一键批量评估（Online Boutique）
│   ├── requirements.txt              # 原始依赖（Python 3.5+）
│   ├── INFOCOM22-SGmVRNN.pdf         # 原论文 PDF
│   └── LICENSE                       # 许可证
│
├── OnlineBoutique_data/              # Online Boutique 实验数据
│   ├── data/                         #   原始 Prometheus 采集的 KPI 数据
│   │   ├── baseline/                 #     基线（正常）数据
│   │   ├── fault_cpu_stress/         #     CPU 压力故障
│   │   ├── fault_net_delay/          #     网络延迟故障
│   │   ├── fault_pod_kill/           #     Pod 杀死故障
│   │   └── full_timeseries/          #     完整时间序列
│   ├── processed_v2/                 #   预处理后的 KPI 矩阵（归一化 + 标签）
│   ├── processed_seq_v2/             #   滑动窗口切分的序列数据（T=20）
│   └── processed_seq_v2_T10/         #   滑动窗口切分的序列数据（T=10）
│
├── BGL/                              # BGL 数据集（HPC 日志异常检测基准）
│   ├── BGL.log                       #   原始日志
│   ├── BGL_2k.log                    #   2k 行采样日志
│   ├── BGL_templates.csv             #   日志模板
│   └── BGL_2k.log_structured.csv     #   结构化解析结果
│
├── log_data/                         # Online Boutique 日志数据（含故障）
│   └── json_logs/                    #   JSON 格式的服务日志
│       ├── *normal.json              #     正常日志
│       └── *fault.json               #     故障日志
│
├── log_data_normal/                  # Online Boutique 正常日志数据
│   └── json_logs/                    #   JSON 格式的正常日志
│
├── ob_etl/                           # 数据预处理（ETL）脚本
│   ├── etl_from_prometheus.py        #   从 Prometheus 采集 KPI
│   ├── preprocess.py                 #   数据预处理（归一化、切分）
│   ├── build_kpi_matrix.py           #   构建 KPI 矩阵
│   └── requirements.txt              #   现代依赖（PyTorch 2.x）
│
├── logs/                             # 根级别训练日志（.gitignore 忽略）
├── 启动online boutique.md            # Online Boutique 部署指南
├── .gitignore                        # Git 忽略规则
└── README.md                         # 本文件
```

> **注意：** 所有 `model/`、`log_tester/`、`log_trainer/`、`eval_results/`、`logs/` 目录均为训练生成的产物，已在 `.gitignore` 中忽略。运行训练/测试/评估后会重新生成。

---

## 环境要求

### 原始论文环境（Python 3.5 / 3.6）

```bash
pip install -r SGmVRNN/requirements.txt
```

### 现代环境（Python 3.10+，推荐用于 Online Boutique 实验）

```bash
pip install -r ob_etl/requirements.txt
```

核心依赖：
- PyTorch ≥ 2.0
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- tqdm

### 硬件要求

- **GPU**（推荐）：CUDA 支持的 NVIDIA GPU，显存 ≥ 4GB
- **CPU**：训练可运行，但速度较慢
- **磁盘**：约 2GB 可用空间（含数据）

---

## 数据集说明

### 1. Online Boutique 数据集（本实验自采）

在 minikube 部署的 12 微服务 Online Boutique 系统上，通过 Prometheus 采集 9 类 KPI 指标，使用 Chaos Mesh 注入 3 种故障类型，共 3 轮重复实验。

- **服务数**：12 个微服务
- **KPI 维度**：9 类
- **故障类型**：PodKill / 网络延迟 / CPU 压力
- **数据格式**：序列化 `.seq` 文件（滑动窗口切分）

### 2. BGL 数据集（基准）

Blue Gene/L 超级计算机日志数据集，用于日志异常检测评估。

- `BGL.log`：完整日志（约 4.7M 行）
- `BGL_2k.log`：2,000 行采样日志
- 结构化模板已预解析为 CSV

---

## 快速开始

### 1. 数据预处理

Online Boutique 数据的 KPI 采集和预处理：

```bash
# 从 Prometheus 采集 KPI 数据
cd ob_etl
python etl_from_prometheus.py

# 构建 KPI 矩阵 + 预处理器
python build_kpi_matrix.py
python preprocess.py
```

> 预处理后的数据已包含在 `OnlineBoutique_data/processed_v2/` 和 `OnlineBoutique_data/processed_seq_v2/` 中，可直接使用。

### 2. 训练

训练所有服务的 SGmVRNN 模型：

```bash
cd SGmVRNN
python batch_train.py
```

训练指定服务：

```bash
python batch_train.py --services checkoutservice frontend recommendationservice
```

自定义超参数：

```bash
python batch_train.py --epochs 30 --batch_size 64 --learning_rate 0.0002
```

### 3. 测试

对训练好的模型进行测试，输出每个样本的 log-likelihood：

```bash
cd SGmVRNN
python batch_test.py
```

### 4. 评估

计算 Precision / Recall / F1 指标，生成可视化图表：

```bash
cd SGmVRNN
python evaluate_all.py --plot
```

### 5. SMD 数据集复现

使用原论文的 SMD 数据集：

```bash
cd SGmVRNN/data_preprocess

# 第一步：数据拼接
python data_preprocess_1st.py --dataset SMD --train_path SMD/train --test_path SMD/test --label_path SMD/test_label --output_path SMD_concated

# 第二步：滑动窗口切分
python data_preprocess_2nd.py --raw_data_file SMD_concated/machine_kpi_ts_data_train.csv --label_file SMD_concated/machine_kpi_ts_label_train.csv --data_path data_processed/smd-train

# 训练
cd ../SGmVRNN
python trainer.py --dataset_path ../data_preprocess/data_processed/smd-train --gpu_id 0 --log_path log_trainer/smd --checkpoints_path model/smd --n 38

# 测试
python tester.py --dataset_path ../data_preprocess/data_processed/machine-1-1 --gpu_id 0 --log_path log_tester/machine-1-1 --checkpoints_path model/smd --n 38
```

### 6. 基线对比

运行基线方法（Continous VRNN、LSTM-NDT、VAE）：

```bash
cd SGmVRNN/baselines
python batch_experiment.py
python compare_results.py
```

---

## 实验结果

### 评估指标

- **Precision**：检测出的异常中真正异常的比例
- **Recall**：真实异常中被检测出的比例
- **F1 Score**：Precision 和 Recall 的调和平均
- **阈值方法**：
  - **BF Search**：暴力搜索最佳 F1 阈值
  - **POT**：Peak Over Threshold 自动阈值

### 预期结果

| 服务 | 故障类型 | 预期 F1 (BF) |
|:----|:--------|:-----------:|
| frontend | Pod Kill | 0.85 - 0.95 |
| checkoutservice | 网络延迟 | 0.80 - 0.92 |
| recommendationservice | CPU 压力 | 0.80 - 0.90 |
| 正常服务 | 无故障 | N/A（无异常样本） |

> 详细实验结果请运行 `evaluate_all.py --plot` 生成评估报告和可视化图表。
