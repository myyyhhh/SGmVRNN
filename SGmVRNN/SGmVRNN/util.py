import torch
import os
import torchvision
import torch.utils.data
import torch.utils.data as data
import numpy as np
from model import *
from tqdm import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob

class KpiReader(data.Dataset):
    def __init__(self, path):
        super(KpiReader, self).__init__()
        self.path = path
        self.length = len(glob.glob(self.path+'/*.seq'))
        data = []
        for i in range(self.length):
            item = torch.load(self.path+'/%d.seq' % (i+1), weights_only=False)
            data.append(item)
        self.data = data


    def __getitem__(self, index):
        item = self.data[index]
        kpi_ts, kpi_label, kpi_value = item['ts'], item['label'], item['value']
        # ts 是 numpy 字符串数组（如 '20260602081022'），DataLoader default_collate 不支持
        # 转成 int64 tensor 即可正常 batch
        if isinstance(kpi_ts, np.ndarray) and kpi_ts.dtype.char == 'U':
            kpi_ts = torch.tensor(kpi_ts.astype(np.int64))
        if isinstance(kpi_label, np.ndarray):
            kpi_label = torch.tensor(kpi_label, dtype=torch.float32)
        # tester.py 用 labels[-1,-1,-1,-1] 4D 索引访问，补上缺失的维度
        # (T, 1) → (T, 1, 1, 1)
        while kpi_label.dim() < 4:
            kpi_label = kpi_label.unsqueeze(-1)
        while kpi_ts.dim() < 4:
            kpi_ts = kpi_ts.unsqueeze(-1)
        return kpi_ts, kpi_label, kpi_value

    def __len__(self):
        return self.length

