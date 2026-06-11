from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from stylizer.vocos_causal.modules import ConvNeXtBlock, AdaLayerNorm
from stylizer.vocos_causal.stft_util import istft_vocos

class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

class VocosPretrainBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        output_dim: int = None,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
        use_realtime: bool = False,
        causal: bool = False
    ):
        super().__init__()
        output_dim = dim if output_dim is None else output_dim
        self.input_channels = input_channels
        self.use_realtime = use_realtime
        self.causal = causal
        
        if use_realtime or causal:
            self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=0)
        else:
            self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)

        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                    use_realtime=use_realtime,
                    causal=causal
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.out = nn.Linear(dim, output_dim)
        self.apply(self._init_weights)

        if causal:
            self.embed_left_context = None
            
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get('bandwidth_id', None)
        if self.causal:
            x = self.embed(F.pad(x, pad=(6, 0), mode='constant', value=0))
        else:
            x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        x = self.out(x).transpose(1, 2)
        return x
    
    @torch.no_grad()
    def decode_mel(self, x):
        # x: [B, d_mel, T]
        out = self.forward(x)
        log_mag, raw_phase = torch.chunk(out, 2, dim=1)
        wav_hat = istft_vocos(log_mag, raw_phase)
        return wav_hat
    
    @torch.no_grad()
    def realtime_decode_mel(self, x):
        # x: [B, d_mel, t_chunk]
        if self.embed_left_context is None:
            self.embed_left_context = torch.zeros(x.size(0), self.input_channels, 6).to(x.device)
        
        x_cat = torch.cat([self.embed_left_context, x], dim=2)  # [B, C, 6 + t_chunk]
        x_out = self.embed(x_cat)  # [B, dim, t_chunk]
        self.embed_left_context = x[:, :, -6:]  # update left context
        
        x_out = self.norm(x_out.transpose(1, 2)).transpose(1, 2)
        for conv_block in self.convnext:
            x_out = conv_block.realtime_forward(x_out)
        x_out = self.final_layer_norm(x_out.transpose(1, 2))
        x_out = self.out(x_out).transpose(1, 2)
        log_mag, raw_phase = torch.chunk(x_out, 2, dim=1)
        wav_hat = istft_vocos(log_mag, raw_phase)
        return wav_hat
    
    def clear_cache(self):
        self.embed_left_context = None
        for conv_block in self.convnext:
            conv_block.clear_cache()
        
        
        
        