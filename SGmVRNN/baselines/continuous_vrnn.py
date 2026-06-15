"""
连续 VRNN（无 Gumbel）消融变体 — 把 Gumbel-Softmax 换成连续高斯

基于 model.py，核心改动：
  - xh_c_layer 输出 mean+logvar（2×cate_dim），替代 logits（cate_dim）
  - Gumbel-Softmax 采样 → Gaussian 重参数化采样
  - cate 的 KL 从分类熵改为高斯 KL: KL(N(μ,σ²) || N(0,1))
  - LSTM 结构保持不变

用法:
    python baselines/continuous_vrnn.py --mode train ...
    python baselines/continuous_vrnn.py --mode test  ...
"""
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SGmVRNN'))
from util import KpiReader


# ── 基础组件（同 model.py） ──

class ConvUnit1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0,
                 nonlinearity=nn.LeakyReLU(0.2)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride, padding), nonlinearity)
    def forward(self, x):
        return self.model(x)

class ConvUnitTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0,
                 out_padding=0, nonlinearity=nn.LeakyReLU(0.2)):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel, stride, padding), nonlinearity)
    def forward(self, x):
        return self.model(x)

class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=nn.LeakyReLU(0.2)):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_features, out_features), nonlinearity)
    def forward(self, x):
        return self.model(x)


class EncX(nn.Module):
    """CNN encoder — 同 model.py"""
    def __init__(self, enc_dim, enc='CNN', n=38, w=1, T=20):
        super().__init__()
        self.n = n
        self.T = T
        self.conv_dim = enc_dim
        k, s, p = 3, 2, 1
        s_d = (n + 2*p - k) // s + 1
        s_d = (s_d + 2*p - k) // s + 1
        s_d = (s_d + 2*p - k) // s + 1
        self.conv = nn.Sequential(
            ConvUnit1d(1, 8, k, s, p),
            ConvUnit1d(8, 16, k, s, p),
            ConvUnit1d(16, 32, k, s, p),
        )
        self.conv_fc = nn.Sequential(
            LinearUnit(32 * s_d, self.conv_dim * 2),
            LinearUnit(self.conv_dim * 2, self.conv_dim))
    def enc_x(self, x):
        BT = x.size(0) * x.size(1)
        x = x.squeeze(-1).contiguous().view(BT, 1, self.n)
        x = self.conv(x).view(BT, -1)
        x = self.conv_fc(x).view(-1, self.T, self.conv_dim)
        return x
    def forward(self, x):
        return self.enc_x(x)


class DecX(nn.Module):
    """CNN decoder — 同 model.py"""
    def __init__(self, enc_dim, dec_init_dim, dec='CNN', n=38, w=1, T=20):
        super().__init__()
        self.n = n
        self.T = T
        self.conv_dim = enc_dim
        k, s, p = 3, 2, 1
        s_d = (n + 2*p - k) // s + 1
        s_d = (s_d + 2*p - k) // s + 1
        s_d = (s_d + 2*p - k) // s + 1
        self.decoder_out = 8 * s_d - 7
        self.cd = [32, s_d]

        def make_seq(nl):
            return nn.Sequential(
                ConvUnitTranspose1d(32, 16, k, s, p),
                ConvUnitTranspose1d(16, 8, k, s, p),
                ConvUnitTranspose1d(8, 1, k, s, p, nonlinearity=nl))

        self.deconv_fc_mu = nn.Sequential(
            LinearUnit(dec_init_dim, self.conv_dim * 2),
            LinearUnit(self.conv_dim * 2, self.cd[0] * self.cd[1]))
        self.deconv_mu = make_seq(nn.Tanh())
        self.deconv_fc_logsigma = nn.Sequential(
            LinearUnit(dec_init_dim, self.conv_dim * 2),
            LinearUnit(self.conv_dim * 2, self.cd[0] * self.cd[1]))
        self.deconv_logsigma = make_seq(nn.Tanh())

    def _decode(self, x, deconv_fc, deconv):
        x = deconv_fc(x).view(-1, self.cd[0], self.cd[1])
        x = deconv(x)
        if self.decoder_out != self.n:
            x = F.interpolate(x, size=self.n, mode='linear', align_corners=False)
        return x.view(-1, 1, 1, self.n, 1)

    def dec_x_mu(self, x):
        return self._decode(x, self.deconv_fc_mu, self.deconv_mu)
    def dec_x_logsigma(self, x):
        return self._decode(x, self.deconv_fc_logsigma, self.deconv_logsigma)

    def forward(self, x):
        return self.dec_x_mu(x), self.dec_x_logsigma(x)


class ReparameterizeTrick:
    @staticmethod
    def reparameterize_gaussian(mean, logvar, random_sampling=True):
        if random_sampling:
            return mean + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        return mean


# ── 连续 VRNN 模型 ──

class ContVRNN_InferenceNet(nn.Module):
    """
    连续 VRNN InferenceNet。
    和原版唯一区别：cate 从 Gumbel-Softmax 变成连续高斯。
    """
    def __init__(self, cate_dim, z_dim, hidden_dim, enc_dim, enc='CNN',
                 T=20, w=1, n=38, device='cuda:0'):
        super().__init__()
        self.cate_dim = cate_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.T = T
        self.device = device
        self.enc_x = EncX(enc_dim, enc=enc, n=n, w=w, T=T)

        # 连续版: 输出 2*cate_dim (mean + logvar)，替代 Gumbel logits
        self.xh_c_layer = nn.Sequential(
            LinearUnit(enc_dim + hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 2 * cate_dim))

        self.phi_xz = LinearUnit(enc_dim + z_dim, 2 * hidden_dim)
        self.rnn_enc = nn.LSTMCell(2 * hidden_dim, hidden_dim)

        self.Pz_xhc_mean = nn.Sequential(
            LinearUnit(enc_dim + hidden_dim + cate_dim, hidden_dim),
            nn.Linear(hidden_dim, z_dim))
        self.Pz_xhc_logvar = nn.Sequential(
            LinearUnit(enc_dim + hidden_dim + cate_dim, hidden_dim),
            nn.Linear(hidden_dim, z_dim))

    def forward(self, x, temperature):
        B = x.size(0)
        x = x.float()
        x_enc = self.enc_x(x)  # (B, T, enc_dim)

        cate_list, z_list, z_mean_list, z_logvar_list = [], [], [], []
        h_out_list = []
        c_mean_list, c_logvar_list = [], []

        z_t = torch.zeros(B, self.z_dim, device=self.device)
        h_t = torch.zeros(B, self.hidden_dim, device=self.device)
        c_t = torch.zeros(B, self.hidden_dim, device=self.device)

        for t in range(self.T):
            x_h = torch.cat([x_enc[:, t], h_t], dim=1)

            # 连续高斯采样（替代 Gumbel-Softmax）
            c_params = self.xh_c_layer(x_h)  # (B, 2*cate_dim)
            c_mean, c_logvar = c_params.chunk(2, dim=-1)
            cate_t = ReparameterizeTrick.reparameterize_gaussian(c_mean, c_logvar)
            # 用 tanh 约束范围（避免极端值）
            cate_t = torch.tanh(cate_t)

            # z 后验
            x_h_c = torch.cat([x_h, cate_t], dim=1)
            z_mean_t = self.Pz_xhc_mean(x_h_c)
            z_logvar_t = self.Pz_xhc_logvar(x_h_c)
            z_t = ReparameterizeTrick.reparameterize_gaussian(z_mean_t, z_logvar_t, self.training)

            # LSTM 步
            x_z = torch.cat([x_enc[:, t], z_t], dim=1)
            phi = self.phi_xz(x_z)
            h_t, c_t = self.rnn_enc(phi, (h_t, c_t))

            cate_list.append(cate_t.unsqueeze(1))
            z_list.append(z_t.unsqueeze(1))
            z_mean_list.append(z_mean_t.unsqueeze(1))
            z_logvar_list.append(z_logvar_t.unsqueeze(1))
            h_out_list.append(h_t.unsqueeze(1))
            c_mean_list.append(c_mean.unsqueeze(1))
            c_logvar_list.append(c_logvar.unsqueeze(1))

        return (torch.cat(z_list, dim=1), torch.cat(z_mean_list, dim=1),
                torch.cat(z_logvar_list, dim=1), torch.cat(h_out_list, dim=1),
                torch.cat(cate_list, dim=1),
                torch.cat(c_mean_list, dim=1), torch.cat(c_logvar_list, dim=1))


class ContVRNN_GenerationNet(nn.Module):
    """
    生成网络 — 完全同原版 model.py 的 GenerationNet。
    Pz_prior(h, cate) 和 gen_px_hz(h, z_posterior) 接口不变。
    """
    def __init__(self, hidden_dim, cate_dim, z_dim, enc_dim, dec_init_dim,
                 dec='CNN', T=20, w=1, n=38, device='cuda:0'):
        super().__init__()
        self.cate_dim = cate_dim
        self.z_dim = z_dim
        self.T = T
        self.device = device
        self.Gen_net = DecX(enc_dim, dec_init_dim, dec=dec, n=n, w=w, T=T)
        self.Pz_hc_mean = nn.Sequential(
            LinearUnit(hidden_dim + cate_dim, hidden_dim),
            nn.Linear(hidden_dim, z_dim))
        self.Pz_hc_logvar = nn.Sequential(
            LinearUnit(hidden_dim + cate_dim, hidden_dim),
            nn.Linear(hidden_dim, z_dim))

    def Pz_prior(self, h, cate_posterior):
        z_mean, z_logvar = [], []
        for t in range(self.T):
            hc = torch.cat((h[:, t], cate_posterior[:, t]), dim=1)
            z_mean.append(self.Pz_hc_mean(hc).unsqueeze(1))
            z_logvar.append(self.Pz_hc_logvar(hc).unsqueeze(1))
        return torch.cat(z_mean, dim=1), torch.cat(z_logvar, dim=1)

    def gen_px_hz(self, h, z_posterior):
        x_mu, x_logsigma = [], []
        for t in range(self.T):
            hz = torch.cat((h[:, t], z_posterior[:, t]), dim=1)
            mu_t, sigma_t = self.Gen_net(hz)
            x_mu.append(mu_t)
            x_logsigma.append(sigma_t)
        return torch.cat(x_mu, dim=1), torch.cat(x_logsigma, dim=1)

    def forward(self, h, z_posterior, cate_posterior):
        z_mean_prior, z_logvar_prior = self.Pz_prior(h, cate_posterior)
        x_mu, x_logsigma = self.gen_px_hz(h, z_posterior)
        return z_mean_prior, z_logvar_prior, x_mu, x_logsigma


class ContVRNN(nn.Module):
    """连续 VRNN（无 Gumbel）— 完整模型"""
    def __init__(self, cate_dim=5, z_dim=10, conv_dim=20, hidden_dim=20,
                 T=20, w=1, n=36, temperature=5.0, min_temperature=0.1,
                 anneal_rate=0.1, device='cuda:0'):
        super().__init__()
        self.cate_dim = cate_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.T = T
        self.n = n
        self.enc_dim = conv_dim
        self.device = device
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate
        self.dec_init_dim = hidden_dim + z_dim

        self.inference = ContVRNN_InferenceNet(
            cate_dim, z_dim, hidden_dim, self.enc_dim,
            T=T, w=w, n=n, device=device)
        self.generation = ContVRNN_GenerationNet(
            hidden_dim, cate_dim, z_dim, self.enc_dim, self.dec_init_dim,
            T=T, w=w, n=n, device=device)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def loss_fn(self, x, z, z_mean_post, z_logvar_post, z_mean_prior, z_logvar_prior,
                x_mu, x_logsigma, cate, c_mean, c_logvar):
        B = x.size(0)
        # Reconstruction (same)
        x_var = torch.exp(x_logsigma.float()) ** 2
        loglikelihood = -0.5 * torch.sum(
            np.log(2 * np.pi) + torch.log(x_var) + (x.float() - x_mu.float()) ** 2 / x_var)
        # KL for z (same)
        z_var_post = torch.exp(z_logvar_post)
        z_var_prior = torch.exp(z_logvar_prior)
        kld_z = 0.5 * torch.sum(z_logvar_prior - z_logvar_post
                        + ((z_var_post + (z_mean_post - z_mean_prior) ** 2) / z_var_prior) - 1)
        # KL for c: KL(N(μ,σ²) || N(0,1))
        kld_c = -0.5 * torch.sum(1 + c_logvar - c_mean ** 2 - c_logvar.exp())

        return (-loglikelihood + kld_c + kld_z) / B, loglikelihood / B, kld_z / B, kld_c / B

    def forward(self, x):
        (z_post, z_mean_post, z_logvar_post, h_out,
         cate_post, c_mean, c_logvar) = self.inference(x, self.temperature)
        z_mean_prior, z_logvar_prior, x_mu, x_logsigma = \
            self.generation(h_out, z_post, cate_post)
        return (z_mean_post, z_logvar_post, z_post,
                z_mean_prior, z_logvar_prior, x_mu, x_logsigma,
                cate_post, c_mean, c_logvar)


# ── 训练 / 测试 ──

def train_model(model, train_loader, epochs=50, lr=0.0002, device='cuda:0',
                checkpoints_path='model', checkpoint_prefix='cont_vrnn',
                log_path='log_trainer', seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    log_file = os.path.join(log_path, f'{checkpoint_prefix}_loss.txt')

    for epoch in range(epochs):
        model.train()
        losses = []
        for dataitem in train_loader:
            _, _, data = dataitem
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss, llh, kld_z, kld_c = model.loss_fn(
                data, out[2], out[0], out[1], out[3], out[4],
                out[5], out[6], out[7], out[8], out[9])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f'Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}')
        ckpt_path = os.path.join(checkpoints_path, f'{checkpoint_prefix}_epochs{epoch+1}.pth')
        torch.save({
            'epoch': epoch+1, 'temperature': model.temperature,
            'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'losses': losses,
        }, ckpt_path)
        model.temperature = max(model.temperature * np.exp(-model.anneal_rate * (epoch+1)),
                                model.min_temperature)
    print('Training complete!')


def test_model(model, test_loader, device='cuda:0',
               checkpoints_path='model', checkpoint_prefix='cont_vrnn', start_epoch=50,
               log_path='log_tester', log_file=None):
    os.makedirs(log_path, exist_ok=True)
    log_file = log_file or f'{checkpoint_prefix}_epochs{start_epoch}_loss.txt'

    ckpt = torch.load(os.path.join(checkpoints_path, f'{checkpoint_prefix}_epochs{start_epoch}.pth'),
                      weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.temperature = ckpt['temperature']
    print(f'Loaded checkpoint epoch {ckpt["epoch"]}')

    model.to(device)
    model.eval()
    log_path_full = os.path.join(log_path, log_file)

    with torch.no_grad(), open(log_path_full, 'w') as f:
        for dataitem in test_loader:
            timestamps, labels, data = dataitem
            data = data.to(device)
            out = model(data)
            x_mu, x_logsigma = out[5], out[6]
            llh = -0.5 * torch.sum(
                torch.pow((data[:, -1, -1, :, -1].float() - x_mu[:, -1, -1, :, -1]) /
                          torch.exp(x_logsigma[:, -1, -1, :, -1]), 2)
                + 2 * x_logsigma[:, -1, -1, :, -1] + np.log(np.pi * 2), dim=1)

            for i in range(len(llh)):
                ts = timestamps[i, -1, -1, -1].item() if timestamps.dim() >= 4 else 0
                lbl = labels[i, -1, -1, -1].item() if labels.dim() >= 4 else 0
                is_anom = 'Anomaly' if lbl >= 0.5 else 'Normaly'
                f.write(f'{ts},{llh[i].item():.6f},{is_anom}\n')

    print(f'Testing complete! Output: {log_path_full}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--n', type=int, default=7)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--cate_dim', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=10)
    parser.add_argument('--conv_dim', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--min_temperature', type=float, default=0.1)
    parser.add_argument('--anneal_rate', type=float, default=0.1)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--checkpoints_path', type=str, default='model')
    parser.add_argument('--checkpoint_prefix', type=str, default='cont_vrnn')
    parser.add_argument('--log_path', type=str, default='log_tester')
    parser.add_argument('--log_file', type=str, default='')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() and args.gpu_id >= 0 else torch.device('cpu')

    dataset = KpiReader(args.dataset_path)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=(args.mode == 'train'), num_workers=4)

    model = ContVRNN(cate_dim=args.cate_dim, z_dim=args.z_dim, conv_dim=args.conv_dim,
                     hidden_dim=args.hidden_dim, T=args.T, n=args.n,
                     temperature=args.temperature, min_temperature=args.min_temperature,
                     anneal_rate=args.anneal_rate, device=device)

    if args.mode == 'train':
        train_model(model, loader, epochs=args.epochs, lr=args.lr, device=device,
                    checkpoints_path=args.checkpoints_path,
                    checkpoint_prefix=args.checkpoint_prefix, seed=args.seed)
    else:
        test_model(model, loader, device=device,
                   checkpoints_path=args.checkpoints_path,
                   checkpoint_prefix=args.checkpoint_prefix,
                   start_epoch=args.start_epoch,
                   log_path=args.log_path)


if __name__ == '__main__':
    main()
