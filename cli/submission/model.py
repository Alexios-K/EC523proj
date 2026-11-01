"""DCCRN-E seed-paper baseline (Hu et al., Interspeech 2020, arXiv:2008.00264).

Follows the paper's DCCRN-E configuration and the authors' reference code
(github.com/huyanxin/DeepComplexCRN): complex conv encoder/decoder with
channels {32,64,128,256,256,256}, kernel (5,2), stride (2,1); a two-layer
complex LSTM (128 real + 128 imaginary units) with a 1024-unit projection;
and the E (polar) mask. STFT: 25 ms sqrt-Hann window, 6.25 ms hop, 512 FFT.
The model is causal in time; the encoder pads only past frames.

The paper's text lists channels {32,64,128,128,256,256} (2.93M parameters),
but its reported 3.7M parameters match the reference code's
{32,64,128,256,256,256} (3.67M), which is used here.

Differences from the reference code: torch.stft/istft replace its conv STFT,
and the decoder drops the last transposed-conv frame (causal) where the
reference drops the first.

The public contract: build once, then transform one [1,T] waveform without
filenames, targets, SNRs, transcripts, or metadata.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ComplexConv2d(nn.Module):
    """Complex conv on [B, C, F, T]; channels hold [real..., imag...]."""

    def __init__(self, in_ch, out_ch, kernel=(5, 2), stride=(2, 1), padding=(2, 1)):
        super().__init__()
        self.time_pad = padding[1]
        self.real = nn.Conv2d(in_ch // 2, out_ch // 2, kernel, stride, (padding[0], 0))
        self.imag = nn.Conv2d(in_ch // 2, out_ch // 2, kernel, stride, (padding[0], 0))
        for conv in (self.real, self.imag):
            nn.init.normal_(conv.weight, std=0.05)
            nn.init.zeros_(conv.bias)

    def forward(self, x):
        x = F.pad(x, (self.time_pad, 0, 0, 0))  # causal: pad past frames only
        r, i = x.chunk(2, dim=1)
        real = self.real(r) - self.imag(i)
        imag = self.real(i) + self.imag(r)
        return torch.cat([real, imag], dim=1)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(5, 2), stride=(2, 1),
                 padding=(2, 0), output_padding=(1, 0)):
        super().__init__()
        args = (kernel, stride, padding, output_padding)
        self.real = nn.ConvTranspose2d(in_ch // 2, out_ch // 2, *args)
        self.imag = nn.ConvTranspose2d(in_ch // 2, out_ch // 2, *args)
        for conv in (self.real, self.imag):
            nn.init.normal_(conv.weight, std=0.05)
            nn.init.zeros_(conv.bias)

    def forward(self, x):
        r, i = x.chunk(2, dim=1)
        real = self.real(r) - self.imag(i)
        imag = self.real(i) + self.imag(r)
        return torch.cat([real, imag], dim=1)


class ComplexLSTM(nn.Module):
    """Complex LSTM on [T, B, D] real/imag pairs (the reference 'NavieComplexLSTM')."""

    def __init__(self, input_size, hidden_size, projection_dim=None):
        super().__init__()
        self.real = nn.LSTM(input_size // 2, hidden_size // 2)
        self.imag = nn.LSTM(input_size // 2, hidden_size // 2)
        self.proj_r = self.proj_i = None
        if projection_dim is not None:
            self.proj_r = nn.Linear(hidden_size // 2, projection_dim // 2)
            self.proj_i = nn.Linear(hidden_size // 2, projection_dim // 2)

    def forward(self, r, i):
        rr, ii = self.real(r)[0], self.imag(i)[0]
        ri, ir = self.imag(r)[0], self.real(i)[0]
        real, imag = rr - ii, ir + ri
        if self.proj_r is not None:
            real, imag = self.proj_r(real), self.proj_i(imag)
        return real, imag


def complex_cat(a, b):
    """Concatenate along channels, keeping the [real..., imag...] layout."""
    ar, ai = a.chunk(2, dim=1)
    br, bi = b.chunk(2, dim=1)
    return torch.cat([ar, br, ai, bi], dim=1)


class DCCRN(nn.Module):
    def __init__(self, sample_rate=16_000, win_len=400, hop=100, n_fft=512,
                 channels=(32, 64, 128, 256, 256, 256), rnn_units=256, rnn_layers=2):
        super().__init__()
        if sample_rate != 16_000:
            raise ValueError("the course interface is fixed at 16 kHz")
        self.win_len, self.hop, self.n_fft = win_len, hop, n_fft
        self.register_buffer("window", torch.hann_window(win_len) ** 0.5, persistent=True)

        ch = (2,) + tuple(channels)
        self.encoder = nn.ModuleList(
            nn.Sequential(ComplexConv2d(ch[k], ch[k + 1]), nn.BatchNorm2d(ch[k + 1]), nn.PReLU())
            for k in range(len(ch) - 1)
        )
        freq = (n_fft // 2) // 2 ** (len(ch) - 1)  # 256 bins (DC dropped) -> 4
        feat = freq * ch[-1]
        self.rnn = nn.ModuleList(
            ComplexLSTM(feat if k == 0 else rnn_units, rnn_units,
                        projection_dim=feat if k == rnn_layers - 1 else None)
            for k in range(rnn_layers)
        )
        decoder = []
        for k in range(len(ch) - 1, 0, -1):
            layers = [ComplexConvTranspose2d(ch[k] * 2, ch[k - 1])]
            if k > 1:
                layers += [nn.BatchNorm2d(ch[k - 1]), nn.PReLU()]
            decoder.append(nn.Sequential(*layers))
        self.decoder = nn.ModuleList(decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] noisy waveform -> [B, T] enhanced waveform."""
        length = x.shape[-1]
        spec = torch.stft(x, self.n_fft, self.hop, self.win_len, self.window,
                          center=True, return_complex=True)          # [B, 257, M]
        out = torch.stack([spec.real, spec.imag], dim=1)[:, :, 1:]   # [B, 2, 256, M]

        skips = []
        for layer in self.encoder:
            out = layer(out)
            skips.append(out)

        b, c, f, m = out.shape                                       # [B, 256, 4, M]
        seq = out.permute(3, 0, 1, 2)                                # [M, B, C, F]
        r = seq[:, :, : c // 2].reshape(m, b, -1)
        i = seq[:, :, c // 2 :].reshape(m, b, -1)
        for layer in self.rnn:
            r, i = layer(r, i)
        r = r.reshape(m, b, c // 2, f)
        i = i.reshape(m, b, c // 2, f)
        out = torch.cat([r, i], dim=2).permute(1, 2, 3, 0)           # [B, C, F, M]

        for layer, skip in zip(self.decoder, reversed(skips)):
            out = layer(complex_cat(out, skip))[..., :-1]            # drop future frame

        mask = F.pad(out, (0, 0, 1, 0))                              # restore DC: [B, 2, 257, M]
        # E mask: tanh-bounded magnitude, unbounded phase rotation, i.e.
        # tanh(|M|)|X| exp(j(angle X + angle M)) = X * M * tanh(|M|)/|M|.
        # Written without angle() so gradients stay finite where M = 0.
        mag = torch.sqrt(mask[:, 0] ** 2 + mask[:, 1] ** 2 + 1e-8)
        est = spec * torch.complex(mask[:, 0], mask[:, 1]) * (torch.tanh(mag) / mag)
        return torch.istft(est, self.n_fft, self.hop, self.win_len, self.window,
                           center=True, length=length)

    def enhance(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != 16_000 or waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("expected one [1,T] waveform at 16 kHz")
        length = waveform.shape[-1]
        padded = waveform
        if length < self.n_fft:  # reflect padding in the STFT needs a few frames
            padded = F.pad(waveform, (0, self.n_fft - length))
        return self.forward(padded)[..., :length]


def build_enhancer(sample_rate: int) -> nn.Module:
    return DCCRN(sample_rate=sample_rate)
