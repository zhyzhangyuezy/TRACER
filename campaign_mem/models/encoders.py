from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def summarize_prefix(prefix: torch.Tensor) -> torch.Tensor:
    mean = prefix.mean(dim=1)
    std = prefix.std(dim=1, unbiased=False)
    maximum = prefix.max(dim=1).values
    last = prefix[:, -1, :]
    slope = (prefix[:, -1, :] - prefix[:, 0, :]) / max(prefix.shape[1] - 1, 1)
    return torch.cat([mean, std, maximum, last, slope], dim=-1)


class SummaryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        summary_dim = input_dim * 5
        self.network = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.network(summarize_prefix(prefix))


class TCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=8, dilation=4),
            nn.ReLU(),
        )
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        hidden = self.network(prefix.transpose(1, 2))[:, :, : prefix.shape[1]]
        pooled = torch.cat([hidden.mean(dim=-1), hidden[:, :, -1]], dim=-1)
        return self.projection(pooled)


class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        seq_len: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(prefix) + self.pos_embedding[:, : prefix.shape[1], :]
        hidden = self.encoder(hidden)
        pooled = torch.cat([hidden.mean(dim=1), hidden[:, -1, :]], dim=-1)
        return self.projection(pooled)


class LSTMSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=True,
        )
        self.projection = nn.Linear(hidden_dim * 4, embedding_dim)

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(prefix)
        pooled = torch.cat([hidden.mean(dim=1), hidden[:, -1, :]], dim=-1)
        return self.projection(pooled)


class ResidualMLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        hidden = self.dropout(self.fc2(hidden))
        return self.norm(hidden + self.skip(x))


def _moving_average(prefix: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return prefix
    pad = kernel_size // 2
    transposed = prefix.transpose(1, 2)
    padded = F.pad(transposed, (pad, pad), mode="replicate")
    trend = F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
    return trend[:, :, : prefix.shape[1]].transpose(1, 2)


class DLinearSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        seq_len: int,
        moving_avg: int = 3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.moving_avg = moving_avg
        self.seasonal = nn.Linear(seq_len, seq_len)
        self.trend = nn.Linear(seq_len, seq_len)
        nn.init.constant_(self.seasonal.weight, 1.0 / max(seq_len, 1))
        nn.init.constant_(self.trend.weight, 1.0 / max(seq_len, 1))
        nn.init.zeros_(self.seasonal.bias)
        nn.init.zeros_(self.trend.bias)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim * seq_len),
            nn.Linear(input_dim * seq_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        trend = _moving_average(prefix, kernel_size=min(self.moving_avg, max(self.seq_len, 1)))
        seasonal = prefix - trend
        seasonal_out = self.seasonal(seasonal.transpose(1, 2))
        trend_out = self.trend(trend.transpose(1, 2))
        encoded = (seasonal_out + trend_out).reshape(prefix.shape[0], -1)
        return self.projection(encoded)


class HybridDLinearTCNSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.dlinear = DLinearSequenceEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
        self.tcn = TCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )
        summary_dim = input_dim * 5
        self.residual_gate = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_projector = nn.Sequential(
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        final_linear = self.residual_gate[-1]
        if isinstance(final_linear, nn.Linear):
            nn.init.constant_(final_linear.bias, -1.5)

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        dlinear_embedding = self.dlinear(prefix)
        tcn_embedding = self.tcn(prefix)
        summary = summarize_prefix(prefix)
        residual_weight = torch.sigmoid(self.residual_gate(summary))
        residual = self.residual_projector(torch.cat([dlinear_embedding, tcn_embedding], dim=-1))
        return self.output_norm(dlinear_embedding + residual_weight * residual)


class PatchTSTSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        seq_len: int,
        patch_len: int = 2,
        stride: int = 1,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = max(2, min(patch_len, seq_len))
        self.stride = max(1, min(stride, self.patch_len))
        self.patch_proj = nn.Linear(self.patch_len, hidden_dim)
        self.n_patches = max(1, (seq_len - self.patch_len) // self.stride + 1)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim * hidden_dim * 2),
            nn.Linear(input_dim * hidden_dim * 2, embedding_dim),
        )

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        means = prefix.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(prefix, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (prefix - means) / stdev
        tokens = normalized.transpose(1, 2).unfold(dimension=2, size=self.patch_len, step=self.stride)
        # [B, C, P, patch_len]
        tokens = self.patch_proj(tokens)
        batch_size, channels, patches, hidden_dim = tokens.shape
        tokens = tokens.reshape(batch_size * channels, patches, hidden_dim)
        tokens = tokens + self.pos_embedding[:, :patches, :]
        encoded = self.encoder(tokens)
        encoded = encoded.reshape(batch_size, channels, patches, hidden_dim)
        pooled = torch.cat([encoded.mean(dim=2), encoded[:, :, -1, :]], dim=-1)
        return self.projection(pooled.reshape(batch_size, -1))


class InvertedTransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        seq_len: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_proj = nn.Linear(seq_len, hidden_dim)
        self.feature_embedding = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim * hidden_dim),
            nn.Linear(input_dim * hidden_dim, embedding_dim),
        )

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        means = prefix.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(prefix, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (prefix - means) / stdev
        tokens = normalized.transpose(1, 2)
        hidden = self.token_proj(tokens) + self.feature_embedding[:, : tokens.shape[1], :]
        hidden = self.encoder(hidden)
        return self.projection(hidden.reshape(prefix.shape[0], -1))


class TSMixerBlock(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len),
            nn.Dropout(dropout),
        )
        self.channel = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        prefix = prefix + self.temporal(prefix.transpose(1, 2)).transpose(1, 2)
        prefix = prefix + self.channel(prefix)
        return prefix


class TSMixerSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TSMixerBlock(seq_len=seq_len, input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim * seq_len),
            nn.Linear(input_dim * seq_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        means = prefix.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(prefix, dim=1, keepdim=True, unbiased=False) + 1e-5)
        hidden = (prefix - means) / stdev
        for block in self.blocks:
            hidden = block(hidden)
        return self.projection(hidden.reshape(prefix.shape[0], -1))


def _fft_periods(prefix: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    spectrum = torch.fft.rfft(prefix, dim=1)
    frequency_strength = spectrum.abs().mean(dim=0).mean(dim=-1)
    frequency_strength[0] = 0.0
    top_k = max(1, min(top_k, int(frequency_strength.shape[0] - 1)))
    scores, indices = torch.topk(frequency_strength, k=top_k)
    periods = prefix.new_tensor(
        [max(1, int(prefix.shape[1] // max(int(index.item()), 1))) for index in indices],
        dtype=torch.long,
    )
    batch_scores = spectrum.abs().mean(dim=-1)[:, indices]
    return periods, batch_scores


class TimesBlock(nn.Module):
    def __init__(self, hidden_dim: int, seq_len: int, top_k: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.top_k = top_k
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = prefix.shape
        period_list, period_weight = _fft_periods(prefix, top_k=self.top_k)
        outputs: list[torch.Tensor] = []
        target_len = seq_len
        for index in range(period_list.shape[0]):
            period = int(period_list[index].item())
            padded_len = ((target_len + period - 1) // period) * period
            if padded_len > target_len:
                pad = prefix.new_zeros(batch_size, padded_len - target_len, hidden_dim)
                hidden = torch.cat([prefix, pad], dim=1)
            else:
                hidden = prefix
            hidden = hidden.reshape(batch_size, padded_len // period, period, hidden_dim).permute(0, 3, 1, 2)
            hidden = self.conv(hidden)
            hidden = hidden.permute(0, 2, 3, 1).reshape(batch_size, padded_len, hidden_dim)
            outputs.append(hidden[:, :target_len, :])
        stacked = torch.stack(outputs, dim=-1)
        weights = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1)
        return torch.sum(stacked * weights, dim=-1) + prefix


class TimesNetSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([TimesBlock(hidden_dim=hidden_dim, seq_len=seq_len, top_k=min(2, seq_len)) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        means = prefix.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(prefix, dim=1, keepdim=True, unbiased=False) + 1e-5)
        hidden = self.input_proj((prefix - means) / stdev)
        for block in self.blocks:
            hidden = self.norm(block(hidden))
        hidden = self.dropout(hidden)
        pooled = torch.cat([hidden.mean(dim=1), hidden[:, -1, :]], dim=-1)
        return self.projection(pooled)


class TiDESequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        flattened_dim = input_dim * seq_len
        summary_dim = input_dim * 5
        self.summary_encoder = ResidualMLPBlock(summary_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.encoder = nn.Sequential(
            ResidualMLPBlock(flattened_dim + hidden_dim, hidden_dim, hidden_dim, dropout=dropout),
            *[ResidualMLPBlock(hidden_dim, hidden_dim, hidden_dim, dropout=dropout) for _ in range(max(num_layers - 1, 0))],
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        means = prefix.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(prefix, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (prefix - means) / stdev
        summary = self.summary_encoder(summarize_prefix(normalized))
        hidden = torch.cat([normalized.reshape(prefix.shape[0], -1), summary], dim=-1)
        hidden = self.encoder(hidden)
        return self.projection(hidden)


def build_encoder(
    encoder_type: str,
    input_dim: int,
    hidden_dim: int,
    embedding_dim: int,
    seq_len: int,
) -> nn.Module:
    if encoder_type == "summary":
        return SummaryEncoder(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    if encoder_type == "tcn":
        return TCNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    if encoder_type == "transformer":
        return TransformerSequenceEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
    if encoder_type == "lstm":
        return LSTMSequenceEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )
    if encoder_type == "dlinear":
        return DLinearSequenceEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
    if encoder_type == "dlinear_tcn":
        return HybridDLinearTCNSequenceEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
    if encoder_type == "patchtst":
        return PatchTSTSequenceEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
    if encoder_type == "itransformer":
        return InvertedTransformerSequenceEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
    if encoder_type == "timesnet":
        return TimesNetSequenceEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
    if encoder_type == "tide":
        return TiDESequenceEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
    if encoder_type == "tsmixer":
        return TSMixerSequenceEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            seq_len=seq_len,
        )
    raise ValueError(f"Unknown encoder type: {encoder_type}")
