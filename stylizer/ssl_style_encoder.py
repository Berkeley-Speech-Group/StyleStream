import math
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoFeatureExtractor, AutoModel, AutoConfig
from inference.hf_cache import cached_from_pretrained

class SSLStackedFeatureExtractor(nn.Module):
    def __init__(self, model_id: str):
        """
        Args:
            model_id (str): Hugging Face model ID (e.g., "facebook/hubert-base-ls960", "facebook/hubert-large-ls960-ft",
                            "facebook/wav2vec2-base-960h", "microsoft/wavlm-base-plus", "microsoft/wavlm-large", "facebook/data2vec-audio-base", "facebook/data2vec-audio-large-960h").
        
        """
        super().__init__()
        self.model_id = model_id

        # Load the corresponding feature extractor.
        self.feature_extractor = cached_from_pretrained(AutoFeatureExtractor, model_id)

        # Load the model configuration and set output_hidden_states to True.
        config = cached_from_pretrained(AutoConfig, model_id)
        config.output_hidden_states = True

        # Load the model using the modified configuration.
        self.model = cached_from_pretrained(AutoModel, model_id, config=config).eval()
        self.model.requires_grad_(False)

    def forward(self, audio_inputs):
        """
        Args:
            audio_inputs (List[np.ndarray]): A batch of raw audio signals @16kHz.
                This can be a list of 1D numpy arrays.
        
        Returns:
            features (Tensor): Extracted features from all layers stacked @50Hz.
                Shape: [batch_size, num_layers, seq_length, hidden_size]
        """
        # Preprocess raw audio signals to the format expected by the SSL model.
        inputs = self.feature_extractor(audio_inputs, return_tensors="pt", padding=True, sampling_rate=16000)
        
        # Move inputs to the same device as the model.
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        # Forward pass through the model.
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # outputs.hidden_states is a tuple containing the hidden states for each layer,
        # with index 0 usually being the embedding output.
        hidden_states = outputs.hidden_states
        features = torch.stack(hidden_states[1:], dim=1)  # Shape: [batch_size, num_layers, seq_length, hidden_size]
        
        return features

class Conv1dLN(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int):
        super().__init__()
        kernel_size = 5
        padding = (kernel_size - 1) // 2 * dilation 
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.ln   = nn.LayerNorm(hidden_dim)
        self.act  = nn.GELU()

    def forward(self, x):
        # x: [B, C, T]
        res = x
        x = self.conv(x)             # → [B, C, T]
        x = x.transpose(1, 2)        # → [B, T, C]
        x = self.ln(x)               # LN over C
        x = self.act(x)
        return x.transpose(1, 2) + res     # → [B, C, T]
    
class Conv1dLNStack(nn.Module):
    def __init__(self, hidden_dim: int, dilations: list[int]):
        super().__init__()
        self.layers = nn.ModuleList([Conv1dLN(hidden_dim, dilation) for dilation in dilations])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ASTP(nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        self.in_dim = in_dim

        self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        """
        x: a 3-dimensional tensor (B,C,T)
        """
        context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
        context_std = torch.sqrt(
            torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
        x_in = torch.cat((x, context_mean, context_std), dim=1) # (B, 3C, T)

        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)

        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1) # (B, 2C)

    def get_out_dim(self):
        self.out_dim = 2 * self.in_dim
        return self.out_dim

class SSLStyleEncoder(nn.Module):
    def __init__(self, n_ssl_layers: int, input_dim: int, hidden_dim: int, dilations: list[int], nstacks: int):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(n_ssl_layers))
        self.input_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv_encoder = nn.Sequential(*[Conv1dLNStack(hidden_dim, dilations) for _ in range(nstacks)])
        self.pool = ASTP(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, ssl_features):
        # ssl_features: [B, num_layers, seq_length, hidden_dim]
        # weighted sum of ssl_features
        normed_w = torch.softmax(self.layer_weights, dim=0)  # (L,)
        weighted_hidden = torch.einsum("b l s d, l -> b s d", ssl_features, normed_w) # [B, seq_length, hidden_dim]

        temp = self.input_conv(weighted_hidden.transpose(1, 2)) # [B, hidden_dim, seq_length]
        temp = self.conv_encoder(temp) # [B, hidden_dim, seq_length]
        temp = self.pool(temp) # [B, hidden_dim * 2]
        emb = self.out_linear(temp) # [B, hidden_dim]
        return emb
