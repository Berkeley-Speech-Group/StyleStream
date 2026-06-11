import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from transformers import AutoFeatureExtractor, AutoModel, AutoConfig, WhisperProcessor, WhisperModel, HubertModel
from lightning import LightningModule
from collections import deque
from typing import Optional, Any, Union, Callable
from torch.nn import Module, MultiheadAttention, Linear, Dropout, LayerNorm
from inference.hf_cache import cached_from_pretrained

class SSLExtractor(nn.Module):
    def __init__(self, model_id: str, layer_idx: int, num_hidden_layers: int = None):
        """
        Args:
            model_id (str): Hugging Face model ID (e.g., "facebook/hubert-base-ls960", "facebook/hubert-large-ls960-ft", "openai/whisper-large-v3"
                            "facebook/wav2vec2-base-960h", "microsoft/wavlm-base-plus", "microsoft/wavlm-large", "facebook/data2vec-audio-base", "facebook/data2vec-audio-large-960h",
                            "lengyue233/content-vec-best").
            layer_idx (int): Index of the hidden state layer to extract.
                             Note that the hidden_states tuple includes the initial embedding output as index 0.
            num_hidden_layers (int, optional): If provided, override the model configuration's number
                                               of hidden layers. This effectively truncates the model
                                               so that only up to the specified number of transformer encoder layers are computed.
                                               
        e.g. if you want to extract features of the 9th layer from SSL, you should set layer_idx = num_hidden_layers = 9. 
        
        """
        super().__init__()
        self.model_id = model_id
        self.layer_idx = layer_idx
        self.downsampling_factor = 320

        self.feature_extractor = cached_from_pretrained(AutoFeatureExtractor, model_id)
        print("Using HuggingFace SSL encoder from transformers")
        config = cached_from_pretrained(AutoConfig, model_id)
        config.output_hidden_states = True
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers
        self.model = cached_from_pretrained(AutoModel, model_id, config=config)
            
        self.model.eval()
        self.model.requires_grad_(False)
        self.hidden_states_key = "hidden_states"

    def forward(self, audio_inputs, downstream_mode=False):
        """
        Args:
            audio_inputs (List[np.ndarray] or Tensor): A batch of raw audio signals @16kHz.
                This can be a list of 1D numpy arrays or a tensor of shape [batch_size, sequence_length].
        
        Returns:
            features (Tensor): Extracted features from the specified layer.
                Shape: [batch_size, seq_length, hidden_size]
            feature_lengths (Tensor): The true lengths of the unpadded features, in shape [B, ]
        """
        # audio_inputs: list of 1D numpy arrays or [B, T] tensor @16kHz
        input_lengths = torch.tensor([len(x) for x in audio_inputs])  # [B]
        feature_lengths = (input_lengths // self.downsampling_factor).to(self.model.device).long()

        # deal with downstream mode, when True, turn off normalization and scale up the audio by 15.0
        if downstream_mode:
            self.feature_extractor.do_normalize = False
            audio_inputs = [x * 15.0 for x in audio_inputs]

        # Preprocess
        inputs = self.feature_extractor(
            audio_inputs,
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
        )
        encoder_inputs = inputs

        # Move to device
        encoder_inputs = {k: v.to(self.model.device) for k, v in encoder_inputs.items()}

        # Forward: ensure output_hidden_states
        with torch.no_grad():
            outputs = self.model(**encoder_inputs, output_hidden_states=True)

        # Fetch hidden states tuple
        hidden_states = getattr(outputs, self.hidden_states_key)

        # Default to last layer
        idx = self.layer_idx if self.layer_idx is not None else len(hidden_states) - 1
        if idx < 0 or idx >= len(hidden_states):
            raise ValueError(f"layer_idx {idx} out of range [0, {len(hidden_states)-1}]")
        features = hidden_states[idx]  # [B, T_enc, D]

        # Adjust feature_lengths to match features shape
        max_len = features.shape[1]
        feature_lengths = torch.clamp(feature_lengths, max=max_len)
         
        return {
            'features': features,
            'feature_lengths': feature_lengths
        }
    
    def clear_cache(self):
        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
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
        return self.dropout(x)

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.

    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    TransformerEncoderLayer can handle either traditional torch.tensor inputs,
    or Nested Tensor inputs.  Derived classes are expected to similarly accept
    both input formats.  (Not all combinations of inputs are currently
    supported by TransformerEncoderLayer while Nested Tensor is in prototype
    state.)

    If you are implementing a custom layer, you may derive it either from
    the Module or TransformerEncoderLayer class.  If your custom layer
    supports both torch.Tensors and Nested Tensors inputs, make its
    implementation a derived class of TransformerEncoderLayer. If your custom
    Layer supports only torch.Tensor inputs, derive its implementation from
    Module.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.nhead = nhead
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
            prepare_streaming: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        why_not_sparsity_fast_path = ''
        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                              f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = src
        
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """Generate ALiBi slopes for attention heads"""
        # Standard ALiBi formula: 1 / 2^(8*i/H) for head i
        slopes = 1.0 / (2 ** (torch.arange(num_heads) * (8.0 / num_heads)))
        return slopes

    def _compute_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute ALiBi bias matrix: -slope_h * |i-j| for each head h"""        
        # Create position indices and distance matrix
        pos = torch.arange(seq_len, device=device)
        distance = torch.abs(pos[:, None] - pos[None, :])  # [seq_len, seq_len]
        
        # Get slopes and move to device
        slopes = self._get_alibi_slopes(self.nhead).to(device)  # [num_heads]
        
        # Compute bias: -slope * distance for each head
        alibi_bias = -slopes[:, None, None] * distance[None, :, :]  # [num_heads, seq_len, seq_len]
        
        return alibi_bias
    
    @torch.no_grad()
    def realtime_forward(self, chunk):
        
        # chunk in shape [t_chunk, 1, D]
        chunk_size = chunk.shape[0]
        if not hasattr(self, 'left_context'):
            self.left_context = deque(maxlen=450 // chunk_size + 1)
            k = chunk
            v = k.clone()
        else:
            k = torch.cat(list(self.left_context) + [chunk], dim=0)  # [t_left + t_chunk, 1, D]
            v = k.clone()
        seq_len = k.shape[0]
            
        # update left context
        self.left_context.append(chunk)
        
        # prepare alibi bias
        src_mask = self._compute_alibi_bias(seq_len, chunk.device)  # [num_heads, seq_len, seq_len]
        src_mask = src_mask[:, -chunk_size:, :]  # only need the last chunk_size rows for current chunk
        
        # going through encoder layer
        x = chunk
        if self.norm_first:
            x = x + self.dropout1(self.self_attn(self.norm1(x), self.norm1(k), self.norm1(v), attn_mask=src_mask, key_padding_mask=None, need_weights=False, is_causal=False)[0])
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self.dropout1(self.self_attn(x, k, v, attn_mask=src_mask, key_padding_mask=None, need_weights=False, is_causal=False)[0]))
            x = self.norm2(x + self._ff_block(x))

        return x
    
    def clear_cache(self):
        if hasattr(self, 'left_context'):
            del self.left_context

class ASRTransformer(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, max_seq_length=5000):
        """
        Args:
            input_dim (int): Dimension of the input Hubert features.
            vocab_size (int): Size of the target vocabulary.
            d_model (int): Dimension of the model.
            nhead (int): Number of heads in multi-head attention.
            num_encoder_layers (int): Number of transformer encoder layers.
            num_decoder_layers (int): Number of transformer decoder layers.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout value.
            max_seq_length (int): Maximum sequence length for positional encoding.
        """
        super().__init__()
        # Project Hubert features to model dimension
        self.input_linear = nn.Linear(input_dim, d_model)
        
        # Token embedding for the target text tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Final linear layer to project to vocabulary logits
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            src : Input features of shape [batch_size, src_seq_len, input_dim].
            tgt : Target token indices of shape [batch_size, tgt_seq_len].
            (Optional masks and key padding masks for transformer attention)
            src_mask : used for encoder self attention (causal / chunked), in shape [S, S]
            tgt_mask : used for decoder self attention (causal / chunked), REQUIRED for asr, in shape [T, T]
            memory_mask : used to further restricting cross attention, in shape [T, S]
            src_key_padding_mask : used for indicating which elements in the encoder's input are padding (encoder self attention), REQUIRED for asr, in shape [B, S]
            tgt_key_padding_mask : used for indicating which elements in the decoder's input are padding (decoder self attention), REQUIRED for asr, in shape [B, T]
            memory_key_padding_mask : used for preventing attending to padded positions during cross attention, REQUIRED for asr, in shape [B, S]
            
        Returns:
            logits: in shape [batch_size, tgt_seq_len, vocab_size].
        """
        # Encode the input features
        src = self.input_linear(src)                    # [B, src_seq_len, d_model]
        src = self.pos_encoder(src)                     # add positional encodings
        src = src.transpose(0, 1)                       # [src_seq_len, B, d_model]
        
        # Embed target tokens and add positional encoding
        tgt = self.embedding(tgt)                       # [B, tgt_seq_len, d_model]
        tgt = self.pos_decoder(tgt)
        tgt = tgt.transpose(0, 1)                       # [tgt_seq_len, B, d_model]
        
        # Pass through the Transformer encoder-decoder
        output = self.transformer(
            src, tgt, 
            src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Project transformer outputs to vocabulary logits
        output = self.output_linear(output)             # [tgt_seq_len, B, vocab_size]
        logits = output.transpose(0, 1)                 # [B, tgt_seq_len, vocab_size]
        
        return logits
    
class ASRTransformerDecoderOnly(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, max_seq_length=5000):
        """
        Args:
            input_dim (int): Dimension of the input Hubert features.
            vocab_size (int): Size of the target vocabulary.
            d_model (int): Dimension of the model.
            nhead (int): Number of heads in multi-head attention.
            num_decoder_layers (int): Number of transformer decoder layers.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout value.
            max_seq_length (int): Maximum sequence length for positional encoding.
        """
        super().__init__()
        # Project Hubert features to model dimension
        self.input_linear = nn.Linear(input_dim, d_model)
        
        # Token embedding for the target text tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_decoder_layers
        )
        
        # Final linear layer to project to vocabulary logits
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            src : Input features of shape [batch_size, src_seq_len, input_dim].
            tgt : Target token indices of shape [batch_size, tgt_seq_len].
            (Optional masks and key padding masks for transformer attention)
            tgt_mask : used for decoder self attention (causal / chunked), REQUIRED for asr, in shape [T, T]
            memory_mask : used to further restricting cross attention, in shape [T, S]
            tgt_key_padding_mask : used for indicating which elements in the decoder's input are padding (decoder self attention), REQUIRED for asr, in shape [B, T]
            memory_key_padding_mask : used for preventing attending to padded positions during cross attention, REQUIRED for asr, in shape [B, S]
            
        Returns:
            logits: in shape [batch_size, tgt_seq_len, vocab_size].
        """
        # Encode the input features
        src = self.input_linear(src)                    # [B, src_seq_len, d_model]
        src = self.pos_encoder(src)                     # add positional encodings
        src = src.transpose(0, 1)                       # [src_seq_len, B, d_model]
        
        # Embed target tokens and add positional encoding
        tgt = self.embedding(tgt)                       # [B, tgt_seq_len, d_model]
        tgt = self.pos_decoder(tgt)
        tgt = tgt.transpose(0, 1)                       # [tgt_seq_len, B, d_model]
        
        # Pass through the Transformer encoder-decoder
        output = self.transformer_decoder(
            tgt, src, 
            tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Project transformer outputs to vocabulary logits
        output = self.output_linear(output)             # [tgt_seq_len, B, vocab_size]
        logits = output.transpose(0, 1)                 # [B, tgt_seq_len, vocab_size]
        
        return logits

def round_ste(x: Tensor) -> Tensor:
    """Straight-Through Estimator for rounding: forward x.round(), backward identity."""
    return x + (x.round() - x).detach()

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        During forward pass, GRL acts as identity.
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In backward pass, gradients are reversed (multiplied by -lambda).
        """
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_=1.0):
    """
    Convenience function to apply the GradientReversalFunction.
    """
    return GradientReversalFunction.apply(x, lambda_)

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return grad_reverse(x, self.lambda_)

class FSQ(nn.Module):
    """
    Finite Scalar Quantization with symmetric bounding and index output.
    
    Args:
        levels (List[int]): Factorized levels for each codebook dimension (e.g. [5, 5, 5]).
        input_dim (int): Dimension of the input features.
    """
    def __init__(
        self,
        levels: list[int],
        input_dim: int,
    ):
        super().__init__()
        _levels = levels
        levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("levels", levels)
        basis = torch.cumprod(torch.tensor([1] + _levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("basis", basis)
        
        self.codebook_dim = len(levels)
        self.input_dim = input_dim
        
        self.project_in = nn.Linear(input_dim, self.codebook_dim)
        self.project_out = nn.Linear(self.codebook_dim, input_dim)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Maps z into a symmetric range based on levels."""
        half_l = (self.levels - 1) * (1 + eps) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def _scale_and_shift(self, zhat: Tensor) -> Tensor:
        """
        Converts zhat from [-1,1] to [0, levels-1] for each dimension.
        """
        half_width = self.levels.to(zhat.dtype) // 2
        return zhat * half_width + half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """
        Converts quantized code (shape [..., codebook_dim] in [-1,1]) into a single integer index.
        """
        assert zhat.shape[-1] == self.codebook_dim
        zhat_scaled = self._scale_and_shift(zhat)
        return (zhat_scaled * self.basis.to(zhat_scaled.dtype)).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """
        Inverse: converts a single index into factorized codes of shape [..., codebook_dim].
        """
        device = indices.device
        out_codes = []
        for d in range(self.codebook_dim):
            divisor = self.basis[d].to(device)
            level_d = self.levels[d].to(device)
            out_codes.append(((indices // divisor) % level_d).unsqueeze(-1))
        return torch.cat(out_codes, dim=-1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: [B, T, input_dim]
        Returns:
            quantized   : Quantized output, shape [B, T, input_dim] if project_back else [B, T, codebook_dim].
            indices     : Integer indices, shape [B, T], each in [0, num_codes).
        """
        # Project input to codebook space: [B, T, codebook_dim]
        z = self.project_in(x)

        # Apply symmetric bounding into [-(level-1)/2, (level-1)/2]
        z_bound = self.bound(z)
        
        # Compute half_width per dimension (as float) so that dividing maps z_bound to [-1,1]
        half_width = self.levels.to(z.dtype) // 2
        z_norm = round_ste(z_bound) / half_width  # now in [-1, 1]
        
        # Convert normalized codes to indices
        indices = self.codes_to_indices(z_norm)  # [B, T]
        
        # Project back to the hidden dim
        quantized = self.project_out(z_norm)
        
        return quantized, indices

class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6, ema_momentum=0.06, ch_momentum=None):
        super().__init__()
        # Trainable params (unchanged; ckpt-compatible)
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta  = nn.Parameter(torch.zeros(1, 1, dim))

        # Streaming (EMA) hyperparams
        self.eps = eps
        self.ema_m = ema_momentum
        self._unused_ch_m = ch_momentum  # kept only for ctor compatibility; not used

        # Non-persistent streaming state (won't appear in state_dict)
        self.register_buffer("ema_sq", torch.zeros(1, 1, dim), persistent=False)   # [B,1,C]
        self.register_buffer("initialized", torch.tensor(False), persistent=False)

        self._state_shape = None  # (B, C) for lazy (re)allocation

    # ---------- Non-streaming path (unchanged math) ----------
    def forward(self, x: torch.Tensor, prepare_streaming: bool = False) -> torch.Tensor:
        """
        x: [B, T, C]
        If prepare_streaming=True, we also seed EMA state from x for later realtime_forward.
        Output uses the original non-streaming GRN (full-chunk norm).
        """
        if prepare_streaming:
            self._bootstrap_from(x)

        Gx = torch.norm(x, p=2, dim=1, keepdim=True)             # [B,1,C]
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)         # [B,1,C]
        return self.gamma * (x * Nx) + self.beta + x

    # ---------- Streaming path (causal EMA over per-channel RMS) ----------
    def realtime_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Causal, chunk-by-chunk GRN using EMA stats over time per channel.
        x: [B, T, C] for the current streaming chunk.
        Call forward(..., prepare_streaming=True) once before first chunk, or let it lazy-init here.
        """
        B, T, C = x.shape
        self._ensure_state_shape(B, C, device=x.device, dtype=x.dtype)

        # Per-chunk per-channel mean-square (no future beyond this chunk)
        chunk_sq = (x * x).mean(dim=1, keepdim=True)  # [B,1,C]

        # Lazy init if needed
        if not bool(self.initialized.item()):
            with torch.no_grad():
                self.ema_sq = chunk_sq.detach().clone()
                self.initialized.fill_(True)

        # EMA update (detach stats)
        with torch.no_grad():
            self.ema_sq.mul_(1 - self.ema_m).add_(self.ema_m * chunk_sq.detach())

        # EMA-based per-channel RMS
        G = torch.sqrt(self.ema_sq + self.eps)                     # [B,1,C]
        ch_mean_inst = G.mean(dim=-1, keepdim=True)                # [B,1,1] (instantaneous across channels)
        Nx = G / (ch_mean_inst + self.eps)                         # [B,1,C]

        return x + self.gamma * (x * Nx) + self.beta

    @torch.no_grad()
    def clear_cache(self):
        """Call at utterance boundaries before a new stream."""
        if self._state_shape is not None:
            B, C = self._state_shape
            device, dtype = self.ema_sq.device, self.ema_sq.dtype
            self.ema_sq = torch.zeros(B, 1, C, device=device, dtype=dtype)
        self.initialized.fill_(False)

    # ---------- Helpers ----------
    @torch.no_grad()
    def _bootstrap_from(self, x: torch.Tensor):
        """Seed EMA state from a pre-utterance so realtime_forward starts stable."""
        B, T, C = x.shape
        self._ensure_state_shape(B, C, device=x.device, dtype=x.dtype)
        sq = (x * x).mean(dim=1, keepdim=True)  # [B,1,C]
        self.ema_sq.copy_(sq)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _ensure_state_shape(self, B, C, device, dtype):
        """Ensure streaming buffers match current (B, C)/device/dtype."""
        need_reset = (
            self._state_shape != (B, C) or
            self.ema_sq.device != device or
            self.ema_sq.dtype  != dtype
        )
        if need_reset:
            self.ema_sq = torch.zeros(B, 1, C, device=device, dtype=dtype)
            self.initialized = torch.tensor(False, device=device)
            self._state_shape = (B, C)





# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108

class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
        post_norm: bool = False,
        causal: bool = False
    ):
        super().__init__()
        self.use_post_norm = post_norm
        self.causal = causal
        self.dilation = dilation
        padding = (dilation * (7 - 1)) // 2 if not causal else 0
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        if self.use_post_norm:
            self.post_norm = nn.LayerNorm(dim)
        self.left_context_len = dilation * (7 - 1) 
        self.left_context = None

    def forward(self, x: torch.Tensor, prepare_streaming=False) -> torch.Tensor:
        # x shape: [B, T, C]
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        if self.causal:
            x = F.pad(x, (self.dilation * (7 - 1), 0), mode="constant", value=0)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x, prepare_streaming=prepare_streaming)
        x = self.pwconv2(x)
        if self.use_post_norm:
            return self.post_norm(residual + x)
        else:
            return residual + x
    
    @torch.no_grad()
    # NOTE: GRN is the cause of streaming issue!
    def realtime_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T_chunk, C]
        B, T_chunk, C = x.shape
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        if self.left_context is None:
            self.left_context = torch.zeros(B, C, self.left_context_len, device=x.device)
        x = torch.cat([self.left_context, x], dim=-1)  # pad left context
        self.left_context = x[:, :, -self.left_context_len:]  # update left context for next chunk
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn.realtime_forward(x)
        x = self.pwconv2(x)
        if self.use_post_norm:
            return self.post_norm(residual + x)
        else:
            return residual + x
        
    def clear_cache(self):
        self.left_context = None
        self.grn.clear_cache()

class StatsPoolingClassifierLayer(nn.Module):  
    def __init__(self, input_dim, emb_dim, n_classes, use_std=True):
        super().__init__()
        self.use_std = use_std
        factor = 2 if use_std else 1
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_dim*factor, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.mapping = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim, n_classes)
        )
        
    def forward(self, x):
        # x in shape [B, T, input_dim]
        if self.use_std:
            in_feat = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1) # [B, 2*input_dim]
        else:
            in_feat = x.mean(dim=1) # [B, input_dim]
        emb = self.embedding_layer(in_feat) # [B, emb_dim] 
        logits = self.mapping(emb) # [B, n_classes]
        return logits

class FSQTransformer(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, fsq_layer_idx=4,
                 dim_feedforward=2048, dropout=0.1, max_seq_length=5000, use_convnext=False,
                 fsq_levels=[5, 5, 5]):
        super().__init__()
        self.nhead = nhead
        self.use_convnext = use_convnext
        self.num_encoder_layers = num_encoder_layers
        self.fsq_layer_idx = fsq_layer_idx
        assert num_encoder_layers >= fsq_layer_idx
        
        self.input_linear = nn.Linear(input_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_length)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len=max_seq_length)

        # Build encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        if self.use_convnext:
            self.convnext_blocks = nn.ModuleList([
                ConvNeXtV2Block(dim=d_model, intermediate_dim=d_model*2, dilation=1, post_norm=True, causal=False) for _ in range(fsq_layer_idx)
            ])

        # Insert FSQ
        self.fsq = FSQ(levels=fsq_levels, input_dim=d_model)

        # Build decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # final linear
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, return_layers_features=False, return_logits=True, return_features=False, prepare_streaming=False):
        """
        Args:
            src : Input features of shape [batch_size, src_seq_len, input_dim].
            tgt : Target token indices of shape [batch_size, tgt_seq_len].
            (Optional masks and key padding masks for transformer attention)
            src_mask : used for encoder self attention, in shape [S, S]
            tgt_mask : used for decoder self attention, REQUIRED for asr, in shape [T, T]
            memory_mask : used to further restricting cross attention, in shape [T, S]
            src_key_padding_mask : used for indicating which elements in the encoder's input are padding (encoder self attention), REQUIRED for asr, in shape [B, S]
            tgt_key_padding_mask : used for indicating which elements in the decoder's input are padding (decoder self attention), REQUIRED for asr, in shape [B, T]
            memory_key_padding_mask : used for preventing attending to padded positions during cross attention, REQUIRED for asr, in shape [B, S]
            
        Returns:
            logits: in shape [batch_size, tgt_seq_len, vocab_size].
        """
        if return_layers_features:
            layers_features = []
        
        x = self.input_linear(src)        # [B, T, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)                 # [T, B, d_model]

        # Pass first part of encoder layers
        for i in range(self.fsq_layer_idx):
            if prepare_streaming:
                x = self.encoder.layers[i](x, src_mask, src_key_padding_mask=src_key_padding_mask, prepare_streaming=prepare_streaming)
            else:
                x = self.encoder.layers[i](x, src_mask, src_key_padding_mask=src_key_padding_mask)
            if self.use_convnext:
                x = self.convnext_blocks[i](x.transpose(0, 1), prepare_streaming=prepare_streaming).transpose(0, 1)
            if return_layers_features:
                layers_features.append(x.transpose(0, 1)) # [B, T, d_model]
        
        if return_features:
            return x.transpose(0, 1)  # [B, T, d_model]

        if not return_logits:
            return layers_features

        # FSQ 
        x = x.transpose(0, 1) # [B, T, d_model]
        x_quant, x_indices = self.fsq(x) # [B, T, d_model] plus indices [B, T]
        x_quant = self.pos_encoder(x_quant).transpose(0, 1) # [T, B, d_model]

        # Pass the rest of encoder layers
        for i in range(self.fsq_layer_idx, self.num_encoder_layers):
            x_quant = self.encoder.layers[i](x_quant, src_mask=None, src_key_padding_mask=src_key_padding_mask)

        memory = x_quant

        # Decoder
        y = self.embedding(tgt)
        y = self.pos_decoder(y)
        y = y.transpose(0, 1) # [tgt_seq_len, B, d_model]

        output = self.decoder(
            y, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = self.output_linear(output)             # [tgt_seq_len, B, vocab_size]
        logits = output.transpose(0, 1)                 # [B, tgt_seq_len, vocab_size]
        
        if return_layers_features:
            return logits, layers_features
        else:
            return logits
    
    def realtime_forward(self, src_chunk):
        # src_chunk: [B, T_chunk, input_dim]
        x = self.input_linear(src_chunk).transpose(0, 1) # [T_chunk, B, d_model]

        # Pass through encoder layers
        for i in range(self.fsq_layer_idx):
            x = self.encoder.layers[i].realtime_forward(x)
            if self.use_convnext:
                x = self.convnext_blocks[i].realtime_forward(x.transpose(0, 1)).transpose(0, 1)
        
        x = x.transpose(0, 1) # [B, T_chunk, d_model]
        return x
    
    def clear_cache(self):
        if self.use_convnext:
            for block in self.convnext_blocks:
                block.clear_cache()
        for layer in self.encoder.layers:
            layer.clear_cache()


def generate_masks_from_lengths(src_lengths, tgt_lengths, device="cuda", max_src_len=None, max_tgt_len=None):
    """
    Generate transformer masks given true lengths of source and target sequences.
    
    Args:
        src_lengths (List[int] or Tensor): True lengths of source sequences (SSL features)
            for each example in the batch.
        tgt_lengths (List[int] or Tensor): True lengths of target text token sequences
            for each example in the batch.
        device (str or torch.device): Device on which to create the masks.
    
    Returns:
        tgt_mask (Tensor): [max_tgt_len, max_tgt_len] square subsequent mask for decoder self-attention.
        src_key_padding_mask (Tensor): [batch_size, max_src_len] mask for encoder input (True = padding).
        tgt_key_padding_mask (Tensor): [batch_size, max_tgt_len] mask for decoder input (True = padding).
        memory_key_padding_mask (Tensor): [batch_size, max_src_len] mask for cross-attention (same as src_key_padding_mask).
    """
    # Convert lists to tensors if needed.
    if not torch.is_tensor(src_lengths):
        src_lengths = torch.tensor(src_lengths, dtype=torch.long, device=device)
    if not torch.is_tensor(tgt_lengths):
        tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.long, device=device)
    
    batch_size = src_lengths.size(0)
    max_src_len = max_src_len if max_src_len is not None else int(src_lengths.max().item())
    max_tgt_len = max_tgt_len if max_tgt_len is not None else int(tgt_lengths.max().item())
        
    # Create source key padding mask.
    # For each example, positions greater than or equal to its true length are padding (True).
    src_range = torch.arange(max_src_len, device=device).unsqueeze(0).expand(batch_size, max_src_len)
    src_key_padding_mask = src_range >= src_lengths.unsqueeze(1)
    
    # Create target key padding mask.
    tgt_range = torch.arange(max_tgt_len, device=device).unsqueeze(0).expand(batch_size, max_tgt_len)
    tgt_key_padding_mask = tgt_range >= tgt_lengths.unsqueeze(1)
    
    # Generate the causal (subsequent) mask for the target.
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(max_tgt_len).to(device)
    
    # Memory key padding mask is the same as src_key_padding_mask.
    memory_key_padding_mask = src_key_padding_mask.clone()
    
    return {
        'tgt_mask': tgt_mask,
        'src_key_padding_mask': src_key_padding_mask,
        'tgt_key_padding_mask': tgt_key_padding_mask,
        'memory_key_padding_mask': memory_key_padding_mask
    }
    
def generate_src_padding_masks(src_lengths, device="cuda"):
    """
    Generate transformer masks given true lengths of source and target sequences.
    
    Args:
        src_lengths (List[int] or Tensor): True lengths of source sequences (SSL features)
            for each example in the batch.
        device (str or torch.device): Device on which to create the masks.
    
    Returns:
        src_key_padding_mask (Tensor): [batch_size, max_src_len] mask for encoder input (True = padding).
        memory_key_padding_mask (Tensor): [batch_size, max_src_len] mask for cross-attention (same as src_key_padding_mask).
    """
    # Convert lists to tensors if needed.
    if not torch.is_tensor(src_lengths):
        src_lengths = torch.tensor(src_lengths, dtype=torch.long, device=device)
    
    batch_size = src_lengths.size(0)
    max_src_len = int(src_lengths.max().item())
    
    # Create source key padding mask.
    # For each example, positions greater than or equal to its true length are padding (True).
    src_range = torch.arange(max_src_len, device=device).unsqueeze(0).expand(batch_size, max_src_len)
    src_key_padding_mask = src_range >= src_lengths.unsqueeze(1)
    
    # Memory key padding mask is the same as src_key_padding_mask.
    memory_key_padding_mask = src_key_padding_mask.clone()
    
    return {
        'src_key_padding_mask': src_key_padding_mask,
        'memory_key_padding_mask': memory_key_padding_mask
    }
