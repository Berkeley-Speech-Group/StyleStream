"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from stylizer.modules import (
    TimestepEmbedding,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNorm_Final,
)

# noised input mel and context mixing embedding
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, content_dim, out_dim, use_causal_conv=False):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + content_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim, causal=use_causal_conv)

    def forward(self, x: float["b n d"], cond: float["b n d"], content: float["b n d"], drop_mel_cond=False):  # noqa: F722
        if drop_mel_cond:  # cfg for cond mel
            cond = torch.zeros_like(cond, device=cond.device)

        x = self.proj(torch.cat((x, cond, content), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x

class GlobalTokenTransformer(nn.Module):
    def __init__(self, hidden_dim=768, depth=4, heads=12, dim_head=64, ff_mult=4, dropout=0.1, n_tokens=32, token_dim=128):
        super().__init__()
        self.token_pos_embed = nn.Embedding(n_tokens, hidden_dim)
        self.in_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * ff_mult,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=depth,
        )

    def forward(self, global_tokens: float["b t c"]):
        B, T, C = global_tokens.shape
        global_tokens = self.in_mlp(global_tokens)
        global_tokens = global_tokens + self.token_pos_embed(torch.arange(T, device=global_tokens.device))
        global_tokens = self.transformer(global_tokens)
        return global_tokens
    
class GlobalTokenMLP(nn.Module):
    def __init__(self, hidden_dim=768, n_tokens=32, token_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, global_tokens: float["b t c"]):
        global_tokens = self.mlp(global_tokens)
        return global_tokens

class GlobalVectorMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, global_vector: float["b d"]):
        global_vector = self.mlp(global_vector)
        return global_vector

# Transformer backbone using DiT blocks
class DiT(nn.Module):
    def __init__(
        self,
        dim=768,
        depth=12,
        heads=12,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        content_dim=768,
        use_ssl_style_emb=False,
        chunked=False,
    ):
        super().__init__()
        self.use_ssl_style_emb = use_ssl_style_emb
        self.time_embed = TimestepEmbedding(dim)
        self.null_content = nn.Parameter(torch.zeros(content_dim))
        self.input_embed = InputEmbedding(mel_dim, content_dim, dim, use_causal_conv=chunked)
        self.dim = dim
        self.depth = depth

        if use_ssl_style_emb:
            self.ssl_style_emb_mlp = nn.Sequential(
                nn.Linear(768, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            self.null_ssl_style_emb = nn.Parameter(torch.zeros(768))

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    chunked=chunked,
                )
                for _ in range(depth)
            ]
        )

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(
        self,
        x: float["b t d_mel"],  # noisy input mel 
        cond: float["b t d_mel"],  # masked cond mel  
        content: int["b t d_content"],  # pretrained content features
        time: float["b"] | float[""],  # time step  
        drop_mel_cond,  # bool, cfg for cond mel
        drop_content,  # bool, cfg for content
        mask: bool["b t"] | None = None, 
        ssl_style_emb: float["b d"] | None = None,  # ssl style embedding
        chunk_mask: bool["t t"] | None = None,  # chunk mask for attention
    ):
        B, seq_len, _ = x.shape
        
        # Time embedding
        if time.ndim == 0:
            time = time.repeat(B)
        t = self.time_embed(time) # [B, dim]

        # SSL style embedding
        if self.use_ssl_style_emb:
            if drop_mel_cond:
                ssl_style_emb = self.null_ssl_style_emb.expand(B, -1) # [B, 768]
                ssl_style_emb = self.ssl_style_emb_mlp(ssl_style_emb) # [B, dim]
            else:
                ssl_style_emb = self.ssl_style_emb_mlp(ssl_style_emb) # [B, dim]
            t = t + ssl_style_emb

        # CFG for content
        if drop_content:
            content = self.null_content.expand(B, seq_len, -1)
        
        x = self.input_embed(x, cond, content, drop_mel_cond=drop_mel_cond) # [B, seq_len, dim]

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, chunk_mask=chunk_mask)

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output


