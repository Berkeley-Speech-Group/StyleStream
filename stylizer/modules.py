"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb


# raw wav to mel spec


mel_basis_cache = {}
hann_window_cache = {}


def get_bigvgan_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
    fmin=0,
    fmax=None,
    center=False,
):  # Copy from https://github.com/NVIDIA/BigVGAN/tree/main
    device = waveform.device
    key = f"{n_fft}_{n_mel_channels}_{target_sample_rate}_{hop_length}_{win_length}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        from librosa.filters import mel as librosa_mel_fn

        mel = librosa_mel_fn(sr=target_sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=fmin, fmax=fmax)
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_length).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_length) // 2
    waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec


def get_vocos_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


class MelSpec(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
    ):
        super().__init__()
        assert mel_spec_type in ["vocos", "bigvgan"], print("We only support two extract mel backend: vocos or bigvgan")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate

        if mel_spec_type == "vocos":
            self.extractor = get_vocos_mel_spectrogram
        elif mel_spec_type == "bigvgan":
            self.extractor = get_bigvgan_mel_spectrogram

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mel


# sinusoidal position embedding for time


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# sinusoidal position embedding for general sequence
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Compute positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x


# convolutional position embedding


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16, causal=False):
        super().__init__()
        assert kernel_size % 2 != 0
        self.kernel_size = kernel_size
        self.causal = causal
        if causal:
            self.padding = 0
        else:
            self.padding = kernel_size // 2
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=self.padding),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=self.padding),
            nn.Mish(),
        )

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None): 
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        if self.causal:
            x = F.pad(x, pad=((self.kernel_size-1)*2, 0), mode='constant', value=0)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out
    
    def realtime_forward(self, x, left_context):
        raise NotImplementedError("Realtime forward is not implemented for this class")


# rotary positional embedding related


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # length = length if isinstance(length, int) else length.max()
    scale = scale * torch.ones_like(start, dtype=torch.float32)  # in case scale is a scalar
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


# Global Response Normalization layer (Instance Normalization ?)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x

# Prompt Encoder
class PromptEncoder(nn.Module):
    def __init__(self, mel_dim, content_dim, out_dim, dilations=[1, 2, 4, 8]):
        super().__init__()
        self.input_proj = nn.Linear(mel_dim + content_dim, out_dim)
        self.conv_blocks = nn.ModuleList([
            ConvNeXtV2Block(dim=out_dim, intermediate_dim=out_dim*2, dilation=dilation)
            for dilation in dilations
        ])
        # self.pe = SinusoidalPositionalEncoding(out_dim) 
    def forward(self, mel, content, drop_mel_cond=False):
        """
        Args:
            mel: [B, t, mel_dim]
            content: [B, t, content_dim]
        Returns:
            [B, t, out_dim]
        """
        if drop_mel_cond:
            mel = torch.zeros_like(mel, device=mel.device)

        x = self.input_proj(torch.cat((mel, content), dim=-1))
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        # x = self.pe(x)
        return x
        


# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x):
        if self.native_rms_norm:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        else:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight

        return x


# AdaLayerNorm
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNorm for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# FeedForward


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)

# Cross Attention
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head**(-0.5)
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        """
        Args:
            x: Query tensor of shape [batch_size, seq_len, query_dim]
            context: Context tensor of shape [batch_size, context_len, context_dim]
        
        Returns:
            Output tensor of shape [batch_size, seq_len, query_dim]
            
        """
        h = self.heads

        q = self.to_q(x)  # [batch_size, seq_len, inner_dim]
        k = self.to_k(context)  # [batch_size, context_len, inner_dim]
        v = self.to_v(context)  # [batch_size, context_len, inner_dim]

        # Reshape for multi-head attention
        q, k, v = map(lambda t: t.view(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2), (q, k, v))
        # q: [batch_size, heads, seq_len, dim_head]
        # k: [batch_size, heads, context_len, dim_head]
        # v: [batch_size, heads, context_len, dim_head]

        # Compute attention scores
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # [batch_size, heads, seq_len, context_len]
        attn = sim.softmax(dim=-1)  # [batch_size, heads, seq_len, context_len]
        
        # Apply attention to values
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [batch_size, heads, seq_len, dim_head]
        
        # Reshape and project output
        out = out.transpose(1, 2).reshape(out.shape[0], -1, out.shape[-1] * h)  # [batch_size, seq_len, inner_dim]
        return self.to_out(out)  # [batch_size, seq_len, query_dim]


# Attention with possible joint part
# modified from diffusers/src/diffusers/models/attention_processor.py


class Attention(nn.Module):
    def __init__(
        self,
        processor: JointAttnProcessor | AttnProcessor | AlibiAttnProcessor | FlashAlibiAttnProcessor | ChunkedCausalAttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,  # if not None -> joint attention
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        if self.context_dim is not None:
            self.to_q_c = nn.Linear(context_dim, self.inner_dim)
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if qk_norm is None:
                self.c_q_norm = None
                self.c_k_norm = None
            elif qk_norm == "rms_norm":
                self.c_q_norm = RMSNorm(dim_head, eps=1e-6)
                self.c_k_norm = RMSNorm(dim_head, eps=1e-6)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_dim is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, context_dim)

    def forward(
        self,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b n d"] = None,  # context c  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
        chunk_mask=None,
    ) -> torch.Tensor:
        if c is not None:
            if isinstance(self.processor, ChunkedCausalAttnProcessor):
                return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope, chunk_mask=chunk_mask)
            else:
                return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            if isinstance(self.processor, ChunkedCausalAttnProcessor):
                return self.processor(self, x, mask=mask, rope=rope, chunk_mask=chunk_mask)
            else:
                return self.processor(self, x, mask=mask, rope=rope)


# Attention processor
class AttnProcessor:
    def __init__(
        self,
        pe_attn_head: int | None = None,  # number of attention head to apply rope, None for all
    ):
        self.pe_attn_head = pe_attn_head

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(query[:, :pn, :, :], freqs, q_xpos_scale)
                key[:, :pn, :, :] = apply_rotary_pos_emb(key[:, :pn, :, :], freqs, k_xpos_scale)
            else:
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = mask
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x

# Alibi attention processor
class AlibiAttnProcessor:
    def __init__(self):
        pass  # ALiBi has no learnable parameters

    def __call__(
        self,
        attn: Attention,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope=None,  # Ignored (ALiBi replaces RoPE)
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        n_heads = attn.heads

        # Project to query/key/value
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # Reshape to [batch_size, heads, seq_len, dim_head]
        head_dim = attn.inner_dim // n_heads
        query = query.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)

        # Apply qk_norm if present
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        # 1. Compute ALiBi biases -------------------------------------------------
        # Head-specific slopes (decreasing exponentially)
        slopes = 1.0 / (2 ** (torch.arange(n_heads, device=x.device) * (8 / n_heads)))
        # Distance matrix |i - j|
        pos = torch.arange(seq_len, device=x.device)
        distance = torch.abs(pos[:, None] - pos[None, :])  # [seq_len, seq_len]
        # ALiBi bias: -slope * distance (shape: [n_heads, seq_len, seq_len])
        alibi = -slopes[:, None, None] * distance[None, ...]
        alibi = alibi.unsqueeze(0).expand(batch_size, -1, -1, -1)  # Add batch dim

        # 2. Combine with mask ----------------------------------------------------
        if mask is not None:
            # Convert boolean mask to additive mask (-inf for masked positions)
            mask_expanded = mask.unsqueeze(1).unsqueeze(1).expand_as(alibi)
            mask_float = torch.zeros_like(alibi, dtype=torch.float32, device=x.device)
            mask_float = mask_float.masked_fill(~mask_expanded, float("-inf"))
            attn_mask = alibi + mask_float
        else:
            attn_mask = alibi

        # 3. Compute attention -----------------------------------------------------
        x = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,  # ALiBi + mask injected here
            dropout_p=attn.dropout,
            is_causal=False
        )

        # Reshape back and project
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.inner_dim)
        x = attn.to_out[0](x)
        x = attn.to_out[1](x)

        # Optional final masking (matches original code behavior)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        return x


# Chunked attention processor
class ChunkedCausalAttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding
        chunk_mask=None,
    ) -> torch.FloatTensor:
        batch_size, seq_len, _ = x.shape
        n_heads = attn.heads

        # Project to query/key/value
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # Reshape to [batch_size, heads, seq_len, dim_head]
        head_dim = attn.inner_dim // n_heads
        query = query.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)

        # Apply qk_norm if present
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        # Compute ALiBi biases -------------------------------------------------
        # Head-specific slopes (decreasing exponentially)
        slopes = 1.0 / (2 ** (torch.arange(n_heads, device=x.device) * (8 / n_heads)))
        # Distance matrix |i - j|
        pos = torch.arange(seq_len, device=x.device)
        distance = torch.abs(pos[:, None] - pos[None, :])  # [seq_len, seq_len]
        # ALiBi bias: -slope * distance (shape: [n_heads, seq_len, seq_len])
        alibi = -slopes[:, None, None] * distance[None, ...]
        alibi = alibi.unsqueeze(0).expand(batch_size, -1, -1, -1)  # Add batch dim

        # Expand chunk mask to match attention shape
        chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads, -1, -1)

        # Combine masks --------------------------------------------------------
        if mask is not None:
            # Convert boolean mask to additive mask (-inf for masked positions)
            mask_expanded = mask.unsqueeze(1).unsqueeze(1).expand_as(alibi)
            mask_float = torch.zeros_like(alibi, dtype=torch.float32, device=x.device)
            mask_float = mask_float.masked_fill(~mask_expanded, float("-inf"))
            # Combine ALiBi, chunk mask, and padding mask
            attn_mask = alibi + mask_float
            attn_mask = attn_mask.masked_fill(~chunk_mask, float("-inf"))
        else:
            # Just combine ALiBi and chunk mask
            attn_mask = alibi.masked_fill(~chunk_mask, float("-inf"))

        # 4. Compute attention -----------------------------------------------------
        x = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=attn.dropout,
            is_causal=False  # We handle causality through our custom mask
        )

        # Reshape back and project
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.inner_dim)
        x = attn.to_out[0](x)
        x = attn.to_out[1](x)

        # Optional final masking (matches original code behavior)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        return x



# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py


class JointAttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        c: float["b nt d"] = None,  # context c, here text # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.FloatTensor:
        residual = x

        batch_size = c.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # `context` projections
        c_query = attn.to_q_c(c)
        c_key = attn.to_k_c(c)
        c_value = attn.to_v_c(c)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_query = c_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_key = c_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_value = c_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)
        if attn.c_q_norm is not None:
            c_query = attn.c_q_norm(c_query)
        if attn.c_k_norm is not None:
            c_key = attn.c_k_norm(c_key)

        # apply rope for context and noised input independently
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            c_query = apply_rotary_pos_emb(c_query, freqs, q_xpos_scale)
            c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

        # joint attention
        query = torch.cat([query, c_query], dim=2)
        key = torch.cat([key, c_key], dim=2)
        value = torch.cat([value, c_value], dim=2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = F.pad(mask, (0, c.shape[1]), value=True)  # no mask for c (text)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # Split the attention outputs.
        x, c = (
            x[:, : residual.shape[1]],
            x[:, residual.shape[1] :],
        )

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)
        if not attn.context_pre_only:
            c = attn.to_out_c(c)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
            # c = c.masked_fill(~mask, 0.)  # no mask for c (text)

        return x, c


# DiT Block


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, chunked=False, use_flash_attn=False):
        super().__init__()

        self.attn_norm = AdaLayerNorm(dim)
        # self.attn = Attention(
        #     processor=AttnProcessor(pe_attn_head=pe_attn_head),
        #     dim=dim,
        #     heads=heads,
        #     dim_head=dim_head,
        #     dropout=dropout,
        #     qk_norm=qk_norm,
        # )
        
        # Using ALiBi bias for extrapolation - Flash Attention if available
        if not chunked:
            if use_flash_attn:
                self.attn = Attention(
                    processor=FlashAlibiAttnProcessor(),
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                )
            else:
                self.attn = Attention(
                    processor=AlibiAttnProcessor(),
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                )
        else:
            self.attn = Attention(
                processor=ChunkedCausalAttnProcessor(),
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
            )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None, chunk_mask=None):  # x: noised input, t: time embedding, global_tokens: [B, T, C]
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope, chunk_mask=chunk_mask)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x
    
# XAttnDiT Block


class XAttnDiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = AdaLayerNorm(dim)

        
        # Using ALiBi bias for extrapolation
        self.attn = Attention(
            processor=ChunkedCausalAttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.xattn = CrossAttention(
            query_dim=dim,
            context_dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, prompt, t, mask=None, rope=None, chunk_mask=None):  # x: noised input, prompt: prompt, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope, chunk_mask=chunk_mask)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        # xattn
        xattn_output = self.xattn(x=x, context=prompt)
        x = x + xattn_output

        # ffn
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


# MMDiT Block https://arxiv.org/abs/2403.03206


class MMDiTBlock(nn.Module):
    r"""
    modified from diffusers/src/diffusers/models/attention.py

    notes.
    _c: context related. text, cond, etc. (left part in sd3 fig2.b)
    _x: noised input related. (right part)
    context_pre_only: last layer only do prenorm + modulation cuz no more ffn
    """

    def __init__(
        self, dim, heads, dim_head, ff_mult=4, dropout=0.1, context_dim=None, context_pre_only=False, qk_norm=None
    ):
        super().__init__()
        if context_dim is None:
            context_dim = dim
        self.context_pre_only = context_pre_only

        self.attn_norm_c = AdaLayerNorm_Final(context_dim) if context_pre_only else AdaLayerNorm(context_dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=JointAttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=context_dim,
            context_pre_only=context_pre_only,
            qk_norm=qk_norm,
        )

        if not context_pre_only:
            self.ff_norm_c = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)
            self.ff_c = FeedForward(dim=context_dim, mult=ff_mult, dropout=dropout, approximate="tanh")
        else:
            self.ff_norm_c = None
            self.ff_c = None
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, c, t, mask=None, rope=None, c_rope=None):  # x: noised input, c: context, t: time embedding
        # pre-norm & modulation for attention input
        if self.context_pre_only:
            norm_c = self.attn_norm_c(c, t)
        else:
            norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)

        # attention
        x_attn_output, c_attn_output = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope)

        # process attention output for context c
        if self.context_pre_only:
            c = None
        else:  # if not last layer
            c = c + c_gate_msa.unsqueeze(1) * c_attn_output

            norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            c_ff_output = self.ff_c(norm_c)
            c = c + c_gate_mlp.unsqueeze(1) * c_ff_output

        # process attention output for input x
        x = x + x_gate_msa.unsqueeze(1) * x_attn_output

        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x_ff_output = self.ff_x(norm_x)
        x = x + x_gate_mlp.unsqueeze(1) * x_ff_output

        return c, x


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float["b"]):  # noqa: F821
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time

import torch
import math

class StreamingCrossfader:
    def __init__(self, overlap: int, mode: str = "equal_power", device=None, dtype=None):
        """
        overlap: number of samples used for crossfade
        mode: "equal_power" or "linear"
        """
        self.overlap = int(overlap)
        self.mode = mode
        self.device = device
        self.dtype = dtype

        self.tail = None  # last `overlap` samples kept from previous chunk
        self._build_windows()

    def _build_windows(self):
        if self.overlap <= 0:
            self.fade_out = None
            self.fade_in = None
            return

        dev = self.device or "cpu"
        dt = self.dtype or torch.float32

        if self.mode == "linear":
            w_out = torch.linspace(1.0, 0.0, self.overlap, device=dev, dtype=dt)
            w_in  = 1.0 - w_out
        else:  # equal_power
            t = torch.linspace(0.0, math.pi / 2, self.overlap, device=dev, dtype=dt)
            w_out = torch.cos(t) ** 2
            w_in  = torch.sin(t) ** 2

        self.fade_out = w_out
        self.fade_in  = w_in

    def reset(self):
        self.tail = None

    @torch.no_grad()
    def push(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        chunk: 1D tensor [T]
        Returns the part you can output *now*.
        Keeps the last `overlap` samples internally.
        """
        assert chunk.dim() == 1, "chunk must be 1D [T]"
        if self.overlap == 0:
            # No crossfade, just stream everything
            return chunk

        T = chunk.shape[0]
        if T <= self.overlap:
            # Not enough to safely split; just accumulate in tail
            self.tail = chunk if self.tail is None else torch.cat([self.tail, chunk], dim=0)[-self.overlap:]
            return chunk.new_zeros(0)  # emit nothing

        # First chunk: no previous tail to crossfade with
        if self.tail is None:
            # Keep last overlap, output everything before that
            self.tail = chunk[-self.overlap:]
            return chunk[:-self.overlap]

        # Normal case: we have a tail from previous chunk
        # 1) crossfade tail with first `overlap` samples of current chunk
        head = chunk[:self.overlap]
        mixed = self.tail * self.fade_out.to(chunk) + head * self.fade_in.to(chunk)

        # 2) middle part (no overlap)
        if T > 2 * self.overlap:
            middle = chunk[self.overlap:-self.overlap]
        else:
            middle = chunk.new_zeros(0)

        # 3) keep new tail
        self.tail = chunk[-self.overlap:]

        # Emit mixed + middle; tail is held back for next call
        return torch.cat([mixed, middle], dim=0)

    def flush(self) -> torch.Tensor:
        """
        Call at the end of the stream to emit the remaining tail.
        """
        if self.tail is None:
            return torch.tensor([], dtype=self.dtype or torch.float32, device=self.device or "cpu")
        tail = self.tail
        self.tail = None
        return tail
