"""
VAE（无 RNN）消融变体 — 去掉 SGmVRNN 中的 LSTM 循环

基于 model.py，核心改动：
  - InferenceNet 去掉 LSTMCell，各时间步独立推断
  - GenerationNet 不使用 hidden state h
  - dec_init_dim = z_dim（原为 hidden_dim + z_dim）

用法对齐 LSTM-NDT:
    python baselines/vae_no_rnn.py --mode train ...
    python baselines/vae_no_rnn.py --mode test  ...
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


# ── 基础组件（移植自 model.py） ──

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
        x = self.conv(x)
        x = x.view(BT, -1)
        x = self.conv_fc(x)
        return x.view(-1, self.T, self.conv_dim)

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

        def make_seq(nonlinearity):
            return nn.Sequential(
                ConvUnitTranspose1d(32, 16, k, s, p),
                ConvUnitTranspose1d(16, 8, k, s, p),
                ConvUnitTranspose1d(8, 1, k, s, p, nonlinearity=nonlinearity))

        self.deconv_fc_mu = nn.Sequential(
            LinearUnit(dec_init_dim, self.conv_dim * 2),
            LinearUnit(self.conv_dim * 2, self.cd[0] * self.cd[1]))
        self.deconv_mu = make_seq(nn.Tanh())
        self.deconv_fc_logsigma = nn.Sequential(
            LinearUnit(dec_init_dim, self.conv_dim * 2),
            LinearUnit(self.conv_dim * 2, self.cd[0] * self.cd[1]))
        self.deconv_logsigma = make_seq(nn.Tanh())

    def _decode(self, x, deconv_fc, deconv):
        x = deconv_fc(x)
        x = x.view(-1, self.cd[0], self.cd[1])
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


class LossFunctions:
    eps = 1e-8
    def log_normal(self, x, mu, var):
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)

    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))


class ReparameterizeTrick:
    def reparameterize_gaussian(self, mean, logvar, random_sampling=True):
        if random_sampling:
            return mean + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        return mean

    def sample_gumbel(self, shape, device, eps=1e-20):
        U = torch.rand(shape).to(device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, device):
        return F.softmax((logits + self.sample_gumbel(logits.size(), device)) / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, device, hard=False):
        y = self.gumbel_softmax_sample(logits, temperature, device)
        if not hard:
            return y
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        return (y_hard - y).detach() + y


# ── VAE（无 RNN）核心模型 ──

class VAENoRNN_InferenceNet(nn.Module):
    """
    去掉 LSTM 的 InferenceNet。
    各时间步独立推断 cate 和 z，无循环状态传递。
    """
    def __init__(self, cate_dim, z_dim, hidden_dim, enc_dim, enc='CNN',
                 T=20, w=1, n=38, hard_gumbel=False, device='cuda:0'):
        super().__init__()
        self.cate_dim = cate_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.T = T
        self.n = n
        self.device = device
        self.hard_gumbel = hard_gumbel
        self.rt = ReparameterizeTrick()
        self.enc_x = EncX(enc_dim, enc=enc, n=n, w=w, T=T)
        # No LSTMCell — removed!
        # x → cate (keep Gumbel-Softmax)
        self.x_c_layer = LinearUnit(enc_dim, hidden_dim)
        self.c_logits = nn.Linear(hidden_dim, cate_dim)
        # x + cate → z
        self.xc_mean = nn.Sequential(
            LinearUnit(enc_dim + cate_dim, hidden_dim),
            nn.Linear(hidden_dim, z_dim))
        self.xc_logvar = nn.Sequential(
            LinearUnit(enc_dim + cate_dim, hidden_dim),
            nn.Linear(hidden_dim, z_dim))

    def forward(self, x, temperature):
        x = x.float()
        x_enc = self.enc_x(x)                     # (B, T, enc_dim)

        z_post_list, z_mean_list, z_logvar_list = [], [], []
        cate_list, logits_list, prob_list = [], [], []

        for t in range(self.T):
            x_t = x_enc[:, t, :]                   # (B, enc_dim)
            # cate via Gumbel-Softmax
            h_c = self.x_c_layer(x_t)
            logits = self.c_logits(h_c)             # (B, cate_dim)
            prob = F.softmax(logits, dim=-1)
            cate_t = self.rt.gumbel_softmax(logits, temperature, self.device, self.hard_gumbel)

            # z via Gaussian
            xc = torch.cat([x_t, cate_t], dim=1)
            z_mean = self.xc_mean(xc)
            z_logvar = self.xc_logvar(xc)
            z_t = self.rt.reparameterize_gaussian(z_mean, z_logvar, self.training)

            z_post_list.append(z_t.unsqueeze(1))
            z_mean_list.append(z_mean.unsqueeze(1))
            z_logvar_list.append(z_logvar.unsqueeze(1))
            cate_list.append(cate_t.unsqueeze(1))
            logits_list.append(logits.unsqueeze(1))
            prob_list.append(prob.unsqueeze(1))

        return (torch.cat(z_post_list, dim=1),
                torch.cat(z_mean_list, dim=1),
                torch.cat(z_logvar_list, dim=1),
                torch.cat(cate_list, dim=1),
                torch.cat(logits_list, dim=1),
                torch.cat(prob_list, dim=1))


class VAENoRNN_GenerationNet(nn.Module):
    """
    去掉 LSTM 的 GenerationNet。
    无需 hidden state h，Pz_prior 只用 cate，gen_px_hz 只用 z。
    """
    def __init__(self, cate_dim, z_dim, enc_dim, dec_init_dim, dec='CNN',
                 T=20, w=1, n=38, device='cuda:0'):
        super().__init__()
        self.cate_dim = cate_dim
        self.z_dim = z_dim
        self.T = T
        self.n = n
        self.device = device
        self.rt = ReparameterizeTrick()
        # Prior: P(z|c) — no h input
        self.Pz_c_mean = nn.Sequential(
            LinearUnit(cate_dim, z_dim))
        self.Pz_c_logvar = nn.Sequential(
            LinearUnit(cate_dim, z_dim))
        # Decoder: P(x|z) — no h input, dec_init_dim = z_dim
        self.Gen_net = DecX(enc_dim, dec_init_dim, dec=dec, n=n, w=w, T=T)

    def forward(self, z_posterior, cate_posterior):
        B = z_posterior.size(0)
        # P(z|c) prior — no h
        z_mean_prior = []
        z_logvar_prior = []
        for t in range(self.T):
            c_t = cate_posterior[:, t, :]
            z_mean_prior.append(self.Pz_c_mean(c_t).unsqueeze(1))
            z_logvar_prior.append(self.Pz_c_logvar(c_t).unsqueeze(1))
        z_mean_prior = torch.cat(z_mean_prior, dim=1)
        z_logvar_prior = torch.cat(z_logvar_prior, dim=1)

        # P(x|z) — no h
        x_mu_list, x_logsigma_list = [], []
        for t in range(self.T):
            z_t = z_posterior[:, t, :]
            mu_t, logsigma_t = self.Gen_net(z_t)
            x_mu_list.append(mu_t)
            x_logsigma_list.append(logsigma_t)
        x_mu = torch.cat(x_mu_list, dim=1)
        x_logsigma = torch.cat(x_logsigma_list, dim=1)

        return z_mean_prior, z_logvar_prior, x_mu, x_logsigma


class VAENoRNN(nn.Module):
    """VAE（无 RNN）— 完整模型"""
    def __init__(self, cate_dim=5, z_dim=10, conv_dim=20, hidden_dim=20,
                 T=20, w=1, n=36, temperature=5.0, min_temperature=0.1,
                 anneal_rate=0.1, hard_gumbel=False, device='cuda:0'):
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
        self.hard_gumbel = hard_gumbel
        # dec_init_dim = z_dim (no h concat)
        self.dec_init_dim = z_dim

        self.losses = LossFunctions()
        self.inference = VAENoRNN_InferenceNet(
            cate_dim, z_dim, hidden_dim, self.enc_dim,
            T=T, w=w, n=n, hard_gumbel=hard_gumbel, device=device)
        self.generation = VAENoRNN_GenerationNet(
            cate_dim, z_dim, self.enc_dim, self.dec_init_dim,
            dec='CNN', T=T, w=w, n=n, device=device)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def loss_fn(self, x, z, z_mean_post, z_logvar_post, z_mean_prior, z_logvar_prior,
                x_mu, x_logsigma, cate, logits, posterior_probs):
        B = x.size(0)
        loglikelihood = self.losses.log_normal(
            x.float(), x_mu.float(), torch.pow(torch.exp(x_logsigma.float()), 2))
        z_var_post = torch.exp(z_logvar_post)
        z_var_prior = torch.exp(z_logvar_prior)
        kld_z = 0.5 * torch.sum(z_logvar_prior - z_logvar_post
                        + ((z_var_post + torch.pow(z_mean_post - z_mean_prior, 2)) / z_var_prior) - 1)
        kld_cate = -self.losses.entropy(logits, posterior_probs) - np.log(1 / self.cate_dim)
        return (-loglikelihood + kld_cate + kld_z) / B, loglikelihood / B, kld_z / B, kld_cate / B

    def forward(self, x):
        z_post, z_mean_post, z_logvar_post, cate_post, logits, probs = \
            self.inference(x, self.temperature)
        z_mean_prior, z_logvar_prior, x_mu, x_logsigma = \
            self.generation(z_post, cate_post)
        return (z_mean_post, z_logvar_post, z_post,
                z_mean_prior, z_logvar_prior, x_mu, x_logsigma,
                cate_post, logits, probs)


# ── 训练 / 测试 ──

def train_model(model, train_loader, epochs=50, lr=0.0002, device='cuda:0',
                checkpoints_path='model', checkpoint_prefix='vae_no_rnn',
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
            outputs = model(data)
            z_mean_post, z_logvar_post, z = outputs[0], outputs[1], outputs[2]
            z_mean_prior, z_logvar_prior = outputs[3], outputs[4]
            x_mu, x_logsigma = outputs[5], outputs[6]
            cate, logits, probs = outputs[7], outputs[8], outputs[9]
            loss, llh, kld_z, kld_pi = model.loss_fn(
                data, z, z_mean_post, z_logvar_post,
                z_mean_prior, z_logvar_prior, x_mu, x_logsigma,
                cate, logits, probs)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f'Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}  Temp: {model.temperature:.4f}')
        with open(log_file, 'a') as f:
            f.write(f'Epoch {epoch+1} Loss: {avg_loss:.4f}\n')

        # save checkpoint
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
               checkpoints_path='model', checkpoint_prefix='vae_no_rnn', start_epoch=50,
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
            outputs = model(data)
            x_mu, x_logsigma = outputs[5], outputs[6]
            # log-likelihood of last timestamp (matching SGmVRNN convention)
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
    parser.add_argument('--checkpoint_prefix', type=str, default='vae_no_rnn')
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

    model = VAENoRNN(cate_dim=args.cate_dim, z_dim=args.z_dim, conv_dim=args.conv_dim,
                     hidden_dim=args.hidden_dim, T=args.T, n=args.n,
                     temperature=args.temperature, min_temperature=args.min_temperature,
                     anneal_rate=args.anneal_rate, hard_gumbel=False, device=device)

    if args.mode == 'train':
        train_model(model, loader, epochs=args.epochs, lr=args.lr, device=device,
                    checkpoints_path=args.checkpoints_path,
                    checkpoint_prefix=args.checkpoint_prefix, seed=args.seed)
    else:
        test_model(model, loader, device=device,
                   checkpoints_path=args.checkpoints_path,
                   checkpoint_prefix=args.checkpoint_prefix,
                   start_epoch=args.start_epoch,
                   log_path=args.log_path,
                   log_file=args.log_file)


if __name__ == '__main__':
    main()
