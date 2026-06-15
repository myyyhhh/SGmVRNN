"""
LSTM-NDT 基线模型 — 时序重建式异常检测

2 层 LSTM 编码 T 窗口，全连接重建，MSE 做异常分。
输出格式对齐 SGmVRNN tester.py: {timestamp},{score},{Anomaly/Normaly}

用法:
    python baselines/lstm_ndt.py --mode train --dataset_path <train_path> ...
    python baselines/lstm_ndt.py --mode test  --dataset_path <test_path> ...
"""
import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SGmVRNN'))
from util import KpiReader


class LSTMNDT(nn.Module):
    """LSTM 重建模型: 输入 T 窗口，重建整个窗口，MSE 做异常分"""
    def __init__(self, n=7, T=20, hidden_dim=20, num_layers=2):
        super().__init__()
        self.n = n
        self.T = T
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(n, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n)

    def forward(self, x):
        # x: (B, T, 1, n, 1) → (B, T, n)
        x = x.squeeze(-1).squeeze(-2)            # remove w=1 and channel=1 dims
        h, _ = self.lstm(x)                      # (B, T, hidden_dim)
        out = self.fc(h)                          # (B, T, n)
        return out


def train_model(model, train_loader, epochs=50, lr=0.001, device='cuda:0',
                checkpoints_path='model', checkpoint_name='lstm_ndt', seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    os.makedirs(checkpoints_path, exist_ok=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for dataitem in train_loader:
            _, _, data = dataitem
            data = data.to(device)
            optimizer.zero_grad()
            recon = model(data)                    # (B, T, n)
            # 输入取 n 维度
            inp = data.squeeze(-1).squeeze(-2)  # (B, T, n)
            loss = criterion(recon, inp)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.6f}')

        # save checkpoint
        ckpt_path = os.path.join(checkpoints_path,
                                 f'{checkpoint_name}_epochs{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)
    print('Training complete!')


def test_model(model, test_loader, device='cuda:0',
               checkpoints_path='model', checkpoint_name='lstm_ndt', start_epoch=50,
               log_path='log_tester', log_file='lstm_ndt_loss.txt'):
    os.makedirs(log_path, exist_ok=True)

    # load checkpoint
    ckpt = torch.load(os.path.join(checkpoints_path,
                                   f'{checkpoint_name}_epochs{start_epoch}.pth'),
                      weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    print(f'Loaded checkpoint epoch {ckpt["epoch"]}')

    model.to(device)
    model.eval()

    log_path_full = os.path.join(log_path, log_file)
    criterion = nn.MSELoss(reduction='none')

    with torch.no_grad(), open(log_path_full, 'w') as f:
        for dataitem in test_loader:
            timestamps, labels, data = dataitem
            data = data.to(device)
            recon = model(data)                    # (B, T, n)
            inp = data.squeeze(-1).squeeze(-2)  # (B, T, n)

            # Per-sample MSE for last timestep
            mse_per_step = ((inp - recon) ** 2)     # (B, T, n)
            mse_last = mse_per_step[:, -1, :].mean(dim=1)  # (B,)
            score = -mse_last.cpu()                 # negative = anomalous

            for i in range(len(score)):
                ts = timestamps[i, -1, -1, -1].item() if timestamps.dim() >= 4 else 0
                lbl = labels[i, -1, -1, -1].item() if labels.dim() >= 4 else 0
                is_anomaly = 'Anomaly' if lbl >= 0.5 else 'Normaly'
                f.write(f'{ts},{score[i].item():.6f},{is_anomaly}\n')

    print(f'Testing complete! Output: {log_path_full}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n', type=int, default=7)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--checkpoints_path', type=str, default='model')
    parser.add_argument('--checkpoint_name', type=str, default='lstm_ndt')
    parser.add_argument('--log_path', type=str, default='log_tester')
    parser.add_argument('--log_file', type=str, default='')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    dataset = KpiReader(args.dataset_path)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=(args.mode == 'train'), num_workers=4)

    model = LSTMNDT(n=args.n, T=args.T, hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers)

    if args.mode == 'train':
        train_model(model, loader, epochs=args.epochs, lr=args.lr,
                    device=device, checkpoints_path=args.checkpoints_path,
                    checkpoint_name=args.checkpoint_name, seed=args.seed)
    else:
        log_file = args.log_file or f'{args.checkpoint_name}_epochs{args.start_epoch}_loss.txt'
        test_model(model, loader, device=device,
                   checkpoints_path=args.checkpoints_path,
                   checkpoint_name=args.checkpoint_name,
                   start_epoch=args.start_epoch,
                   log_path=args.log_path, log_file=log_file)


if __name__ == '__main__':
    main()
