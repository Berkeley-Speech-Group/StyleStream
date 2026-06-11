from __future__ import annotations
from random import random
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from lightning.pytorch.utilities import rank_zero_only
from torchdiffeq import odeint
from lightning import LightningModule
import soundfile as sf
import librosa
from collections import deque
from pathlib import Path
import numpy as np

from stylizer.utils import (
    exists,
    mask_from_frac_lengths,
    mask_from_start_end_indices,
    take_safe_log,
)
from stylizer.dit import DiT
from destylizer.model import SSLExtractor
from destylizer.asr_lightning_module import ASRLightningModule
from stylizer.ssl_style_encoder import SSLStackedFeatureExtractor
from stylizer.vocos_causal.models import VocosPretrainBackbone
from stylizer.vocos_causal.stft_util import get_mel

CKPT_DIR = Path(__file__).resolve().parent.parent / "assets" / "ckpts"
DESTYLIZER_CKPT = CKPT_DIR / "destylizer.ckpt"
VOCODER_CKPT = CKPT_DIR / "vocos_causal_best.ckpt"

class CFMLightningModule(LightningModule):
    def __init__(
        self,
        dim=768,
        depth=16,
        heads=12,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        content_dim=768,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            method="euler" 
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        frac_lengths_mask: tuple[float, float] = (0.2, 0.6),
        vocab_size=None,
        use_ssl_style_emb=True,
        use_style_encoder=True,
        ssl_style_encoder_config: dict = dict(
            n_ssl_layers=12,
            input_dim=768,
            hidden_dim=768,
            dilations=[1, 2, 4, 8, 16],
            nstacks=2,
        ),
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        warmup_steps: int = 2000,
        total_train_steps: int = 400000,
        initial_lr: float = 1e-6,
        **_unused_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["_unused_kwargs"])
        
        # Store checkpoint path for later loading in setup
        self.content_dim = content_dim
        self.vocab_size = vocab_size
        
        # Set up no_context_mel flag
        if sum(self.hparams['frac_lengths_mask']) != 2:
            self.no_context_mel = False
        else:
            self.no_context_mel = True
        
        # Track if setup has been called
        self._setup_called = False

        # DiT
        self.transformer = DiT(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            ff_mult=ff_mult,
            mel_dim=mel_dim,
            content_dim=content_dim,
            use_ssl_style_emb=use_ssl_style_emb,
            chunked=False,
        )
        
        if use_ssl_style_emb and use_style_encoder:
            from stylizer.ssl_style_encoder import SSLStyleEncoder
            self.ssl_style_encoder = SSLStyleEncoder(**ssl_style_encoder_config)

        # Initialize external models dict to store models without registering them as submodules
        self._external_models = {}

    def _ensure_setup(self):
        """Ensure setup has been called, call it if not"""
        if not self._setup_called:
            self.setup()

    def setup(self, stage=None):
        """Setup method called after the model is moved to the correct device"""
        if self._setup_called:
            return
            
        # Load destylizer based on configuration
        self._external_models['destylizer'] = ASRLightningModule.load_from_checkpoint(
            str(DESTYLIZER_CKPT),
            map_location=self.device
        ).eval()
        self._external_models['destylizer'].requires_grad_(False)
    
        # Load vocoder on GPU for faster inference and less CPU memory usage
        vocoder = VocosPretrainBackbone(
            input_channels=100,
            dim=512,
            intermediate_dim=1536,
            num_layers=8,
            output_dim=1026,
            causal=True,
        )
        vocoder.load_state_dict(torch.load(str(VOCODER_CKPT), map_location=self.device))
        vocoder.to(self.device)
        vocoder.eval()
        vocoder.requires_grad_(False)
        self._external_models['vocoder'] = vocoder
        
        # Load ssl feature extractor only when this checkpoint has local style encoder weights.
        if self.hparams['use_ssl_style_emb'] and self.uses_style_encoder:
            self._external_models['ssl_stacked_features_extractor'] = SSLStackedFeatureExtractor("microsoft/wavlm-base-plus").to(self.device)
            self._external_models['ssl_stacked_features_extractor'].eval()
            self._external_models['ssl_stacked_features_extractor'].requires_grad_(False)

        self._setup_called = True

    @property
    def destylizer(self):
        return self._external_models.get('destylizer')
        
    @property
    def vocoder(self):
        return self._external_models.get('vocoder')
    
    @property
    def ssl_stacked_features_extractor(self):
        return self._external_models.get('ssl_stacked_features_extractor')

    @property
    def uses_style_encoder(self):
        return bool(self.hparams.get("use_style_encoder", True))

    @classmethod
    def load_for_inference(cls, checkpoint_path, device="cuda:0"):
        """Convenience method to load model for inference with setup"""
        model = cls.load_from_checkpoint(checkpoint_path, map_location=device)
        model.setup("predict")
        model.eval()
        return model

    def load_wav(self, path):
        x, fs = sf.read(path)
        if fs != 16000:
            x = librosa.resample(x, orig_sr=fs, target_sr=16000)
        return x

    def _style_file_from_dir(self, style_dir: Path, suffix: str, required=True):
        preferred = style_dir / f"{style_dir.name}{suffix}"
        if preferred.exists():
            return preferred

        matches = sorted(style_dir.glob(f"*{suffix}"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Multiple {suffix} files found in style folder: {style_dir}")
        if required:
            raise FileNotFoundError(f"No {suffix} file found in style folder: {style_dir}")
        return None

    def resolve_style_reference(self, style_ref):
        style_path = Path(style_ref)
        if style_path.is_dir():
            wav_path = self._style_file_from_dir(style_path, ".wav", required=True)
            emb_path = self._style_file_from_dir(style_path, ".npy", required=False)
            return wav_path, emb_path
        return style_path, None

    def load_wav_reference(self, wav_ref):
        if isinstance(wav_ref, (str, Path)):
            wav_path = Path(wav_ref)
            if wav_path.is_dir():
                wav_path = self._style_file_from_dir(wav_path, ".wav", required=True)
            return self.load_wav(wav_path)
        return wav_ref

    def load_style_reference(self, style_ref, style_emb=None):
        external_style_emb = style_emb
        if isinstance(style_ref, (str, Path)):
            wav_path, package_style_emb = self.resolve_style_reference(style_ref)
            style_wav = self.load_wav(wav_path)
            if external_style_emb is None:
                external_style_emb = package_style_emb
        else:
            style_wav = style_ref
        return style_wav, external_style_emb

    def load_style_embedding(self, style_emb):
        if style_emb is None:
            return None
        if isinstance(style_emb, (str, Path)):
            style_emb = np.load(style_emb)
        if isinstance(style_emb, np.ndarray):
            style_emb = torch.from_numpy(style_emb)
        if not torch.is_tensor(style_emb):
            raise TypeError("style_emb must be a .npy path, numpy array, or torch tensor")

        style_emb = style_emb.float()
        if style_emb.ndim == 2 and style_emb.shape[0] == 1:
            style_emb = style_emb.squeeze(0)
        if style_emb.ndim != 1:
            raise ValueError(f"style_emb must have shape [768], got {tuple(style_emb.shape)}")
        if style_emb.shape[0] != 768:
            raise ValueError(f"style_emb must have 768 values, got {tuple(style_emb.shape)}")
        return style_emb.unsqueeze(0).to(self.device)

    def get_ssl_style_embedding(self, style_wav, external_style_emb=None):
        if not self.hparams['use_ssl_style_emb']:
            return None
        if self.uses_style_encoder:
            ssl_features = self.ssl_stacked_features_extractor([style_wav])
            return self.ssl_style_encoder(ssl_features)

        ssl_style_emb = self.load_style_embedding(external_style_emb)
        if ssl_style_emb is None:
            raise ValueError(
                "This checkpoint was loaded without style encoder weights. "
                "Pass a style folder containing a .wav and matching .npy style embedding, "
                "or pass style_emb explicitly."
            )
        return ssl_style_emb

    def create_chunk_mask(self, seq_len, chunk_size):
        # Create a mask that allows attention to:
        # 1. All previous positions (causal)
        # 2. All positions within the same chunk
        # True means allow attention, False means block attention
        
        # Create position indices
        i = torch.arange(seq_len, device=self.device)
        j = torch.arange(seq_len, device=self.device)
        
        # Create chunk indices using integer division
        chunk_i = i // chunk_size
        chunk_j = j // chunk_size
        
        # Create causal mask (j <= i)
        causal_mask = j.unsqueeze(0) <= i.unsqueeze(1)
        
        # Create chunk mask (chunk_j == chunk_i)
        chunk_mask = chunk_j.unsqueeze(0) == chunk_i.unsqueeze(1)
        
        # Combine masks
        final_mask = causal_mask | chunk_mask
        
        return final_mask

    @torch.no_grad()
    def sample(
        self,
        cond: float["t_cond"] | str,  # 1d numpy array, wav path, or style folder
        content: float["t_content"] | str,  # 1d numpy array, wav path, or folder containing a wav
        steps=32,
        cfg_strength=2.0,
        sway_sampling_coef=None,
        style_emb=None,
    ):   
        self._ensure_setup()
        self.eval()
        
        cond, external_style_emb = self.load_style_reference(cond, style_emb=style_emb)
        cond_wav = cond
        
        # extract cond's content
        cond_out_dict = self.destylizer.extract_indices([cond])
        cond_content = cond_out_dict['features'] # [1, t, content_dim]
        
        # extract cond mel
        cond = torch.from_numpy(cond).float().to(self.device).unsqueeze(0) # [1, t']
        cond_mel = take_safe_log(get_mel(cond)) # [1, d_mel, t]
        cond_mel = cond_mel.permute(0, 2, 1) # [1, t, d_mel]
        
        # align cond_content and cond_mel
        cond_len = min(cond_content.shape[1], cond_mel.shape[1]) # scalar t
        cond_content = cond_content[:, :cond_len] # [1, t, d_content]
        cond_mel = cond_mel[:, :cond_len] # [1, t, d_mel]
            
        # extract source content features
        content = self.load_wav_reference(content)
        
        out_dict = self.destylizer.extract_indices([content])
        content = out_dict['features'] # [1, T, d_content]
            
        source_len = content.shape[1] # scalar T
        
        # concat content features of source and condition
        content_concat = torch.concat([cond_content, content], dim=1) # [1, t+T, d_content]
        
        # expand mel 
        cond_mel_concat = torch.concat([cond_mel, torch.zeros(1, source_len, cond_mel.shape[-1], device=self.device)], dim=1) # [1, t+T, d_mel]

        # extract ssl style embeddings
        ssl_style_emb = self.get_ssl_style_embedding(cond_wav, external_style_emb=external_style_emb)
        
        # chunk mask
        seq_len = source_len + cond_len
        chunk_mask = self.create_chunk_mask(seq_len, seq_len)

        # neural ode
        def fn(t, x):
            # predict flow
            pred = self.transformer(
                x=x, cond=cond_mel_concat, content=content_concat, time=t, drop_mel_cond=False, drop_content=False, chunk_mask=chunk_mask, ssl_style_emb=ssl_style_emb
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=cond_mel_concat, content=content_concat, time=t, drop_mel_cond=True, drop_content=True, chunk_mask=chunk_mask, ssl_style_emb=ssl_style_emb
            )
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        y0 = torch.randn_like(cond_mel_concat, device=self.device) # [1, t+T, d_mel]

        # time steps
        t = torch.linspace(0, 1, steps + 1, device=self.device).float()
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # sample through odeint
        trajectory = odeint(fn, y0, t, **self.hparams['odeint_kwargs'])
        out_mel = trajectory[-1] # [1, t+T, d_mel]
        out_mel = out_mel[:, cond_len:].transpose(-2, -1) # [1, d_mel, T] 

        out_mel = torch.exp(out_mel)
        out = self.vocoder.decode_mel(out_mel).squeeze()

        return {
            'pred_mel': out_mel,
            'pred_audio': out
        }
        
    @torch.no_grad()
    def prepare_streaming(self, target_path, chunk_size_16k, style_emb=None):
        # NOTE: chunk_size_16k @16kHz
        self._ensure_setup()
        self.eval()
        tgt, external_style_emb = self.load_style_reference(target_path, style_emb=style_emb)
        
        # fill destylizer deque with tgt chunks
        num_chunks = len(tgt) // chunk_size_16k
        tgt_trim = tgt[-num_chunks*chunk_size_16k:]
        self.destylizer_buffer = deque(maxlen=(16000*6)//chunk_size_16k+1) # about 6s of history
        for i in range(0, len(tgt_trim), chunk_size_16k):
            self.destylizer_buffer.append(tgt_trim[i:i+chunk_size_16k])
                
        # extract target's content
        tgt_out_dict = self.destylizer.extract_indices([tgt])
        tgt_content = tgt_out_dict['features'] # [1, t, content_dim]
        
        # extract target mel
        tgt = torch.from_numpy(tgt).float().to(self.device).unsqueeze(0) # [1, t']
        tgt_mel = take_safe_log(get_mel(tgt)) # [1, d_mel, t]
        tgt_mel = tgt_mel.permute(0, 2, 1) # [1, t, d_mel]
        
        # align tgt_content and tgt_mel
        tgt_len = min(tgt_content.shape[1], tgt_mel.shape[1]) # scalar t
        tgt_content = tgt_content[:, :tgt_len] # [1, t, d_content]
        tgt_mel = tgt_mel[:, :tgt_len] # [1, t, d_mel]
        
        # extract or load ssl style emb
        ssl_style_emb = self.get_ssl_style_embedding(tgt.squeeze().cpu().numpy(), external_style_emb=external_style_emb)
        
        # only store 3s of tgt for CFM past context
        self.tgt_content = tgt_content[:, :50*3]
        self.tgt_mel = tgt_mel[:, :50*3]
        self.ssl_style_emb = ssl_style_emb
    
    @torch.no_grad()
    def realtime_sample(
        self,
        chunk: float["t_chunk"],  # source input chunk, 1d numpy array @16kHz WITH right context, e.g. 9600 (600ms) + 80 (5ms)
        steps=16,
        cfg_strength=2.0,
        chunk_as_content=False,
        chunk_size=None, # @50Hz
    ):
        self.eval()
        if chunk_size is None:
            chunk_size = (len(chunk) - 80) // 320
        if chunk_as_content:
            chunk_content = chunk # [1, t_chunk, content_dim]
        else:
            chunk_cat = np.concatenate(list(self.destylizer_buffer) + [chunk])
            chunk_content = self.destylizer.extract_indices([chunk_cat])['features'][:, -(chunk_size+1):] # [1, t_chunk, content_dim]
            self.destylizer_buffer.append(chunk[:-80])
        if not hasattr(self, 'left_content'):
            self.left_content = deque(maxlen=150 // chunk_size + 1) # keep about 3s of left context
            input_content = chunk_content
        else:
            input_content = torch.cat(list(self.left_content) + [chunk_content], dim=1) # [1, t_left + t_chunk, content_dim]
            
        # update left content
        self.left_content.append(chunk_content)
        
        # concat with target content    
        content_concat = torch.cat([self.tgt_content, input_content], dim=1) # [1, t_tgt + t_left + t_chunk, content_dim]
        
        # expand mel
        mel_concat = torch.cat([self.tgt_mel, torch.zeros(1, input_content.shape[1], self.tgt_mel.shape[-1], device=self.device)], dim=1) # [1, t_tgt + t_left + t_chunk, d_mel]
        
        # chunk mask, here full attention
        seq_len = content_concat.shape[1]
        chunk_mask = self.create_chunk_mask(seq_len, seq_len)
        
        # neural ode
        def fn(t, x):
            # predict flow
            pred = self.transformer(
                x=x, cond=mel_concat, content=content_concat, time=t, drop_mel_cond=False, drop_content=False, chunk_mask=chunk_mask, ssl_style_emb=self.ssl_style_emb
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=mel_concat, content=content_concat, time=t, drop_mel_cond=True, drop_content=True, chunk_mask=chunk_mask, ssl_style_emb=self.ssl_style_emb
            )
            return pred + (pred - null_pred) * cfg_strength

        # noise input, use all zeros as init noise to have deterministic behavior
        y0 = torch.zeros_like(mel_concat, device=self.device) # [1, t_tgt + t_left + t_chunk, d_mel]

        # time steps
        t = torch.linspace(0, 1, steps + 1, device=self.device).float()

        # sample through odeint
        trajectory = odeint(fn, y0, t, **self.hparams['odeint_kwargs'])
        out_mel = trajectory[-1] # [1, t_tgt + t_left + t_chunk, d_mel]
        out_mel = out_mel[:, -(chunk_size+1):].transpose(-2, -1) # [1, d_mel, t_chunk] 
        out_mel = torch.exp(out_mel)

        # vocoding
        out = self.vocoder.realtime_decode_mel(out_mel).squeeze()

        return out
    
    def clear_cache(self):
        if hasattr(self, 'destylizer_buffer'):
            del self.destylizer_buffer
        if hasattr(self, 'left_content'):
            del self.left_content
        self.vocoder.clear_cache()
    
    @torch.no_grad()
    def resynthesize_with_vocos(self, path):
        self._ensure_setup()
        x = self.load_wav(path)
        x = torch.from_numpy(x).float().unsqueeze(0).to(self.device) # [1, t]
        x_mel = get_mel(x) # [1, d_mel, t]
        x_hat = self.vocoder.decode_mel(x_mel).squeeze()
        return x_hat
