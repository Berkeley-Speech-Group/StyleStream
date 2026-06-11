import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule
import math
import itertools

from destylizer.model import SSLExtractor, FSQTransformer, generate_src_padding_masks
from destylizer.text_tokenizer import GraphemeTokenizer

class ASRLightningModule(LightningModule):
    def __init__(
        self,
        ssl_model_id: str = "facebook/hubert-large-ls960-ft",
        ssl_layer_idx: int = 18,
        ssl_num_hidden_layers: int = 18,
        input_dim: int = 1024,
        d_model: int = 768,
        nhead: int = 12,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        use_convnext: bool = True,
        fsq_levels: list = [5, 3, 3],
        fsq_layer_idx: int = 6,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        warmup_steps: int = 4000,
        total_train_steps: int = 100000,
        initial_lr: float = 1e-6,
        **_unused_kwargs,
    ):
        """
        Args:
            ssl_model_id (str): HuggingFace model ID for SSL extractor (e.g., "facebook/hubert-base-ls960").
            ssl_layer_idx (int): Which layer's hidden states to use as encoder output.
            ssl_num_hidden_layers (int, optional): Limit the number of SSL encoder layers if desired.
            vocab_size (int): Size of the target vocabulary (graphemes + PAD/BOS/EOS).
            d_model (int): Dimension of the ASRTransformer.
            nhead (int): Number of attention heads.
            num_encoder_layers (int): Number of transformer encoder layers.
            num_decoder_layers (int): Number of transformer decoder layers.
            dim_feedforward (int): Dim of FFN in transformer.
            dropout (float): Dropout probability.
            fsq_levels (list): a list of FSQ levels.
            lr (float): Learning rate.
            weight_decay (float): Weight decay for AdamW.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["_unused_kwargs"]) 

        # Track if setup has been called
        self._setup_called = False
        self._external_models = {}
        
        # ------------------
        # 1) SSL Extractor
        # ------------------
        self.ssl_extractor = SSLExtractor(
            model_id=ssl_model_id,
            layer_idx=ssl_layer_idx,
            num_hidden_layers=ssl_num_hidden_layers,
        )
        
        # ------------------
        # 2) Tokenizer
        # ------------------
        self.text_tokenizer = GraphemeTokenizer(
            include_space=True,
            include_apostrophe=True,
            include_dash=False,
        )
        
        # Match the model vocabulary to the tokenizer.
        self.vocab_size = self.text_tokenizer.vocab_size
        
        # ------------------
        # 3) ASR Transformer
        # ------------------
        self.asr_model = FSQTransformer(
            input_dim=input_dim,               
            vocab_size=self.vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            fsq_layer_idx=fsq_layer_idx,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=5000,
            use_convnext=use_convnext,
            fsq_levels=fsq_levels,
        )
        
        # ------------------
        # 4) Loss
        # ------------------
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.text_tokenizer.pad_id()
        )
        
        # ------------------
        # 5) Hyperparams
        # ------------------
        self.lr = lr
        self.weight_decay = weight_decay

    def _ensure_setup(self):
        """Ensure setup has been called, call it if not"""
        if not self._setup_called:
            self.setup()

    def setup(self, stage: str):
        if self._setup_called:
            return
        
        if self.hparams.get('use_distillation', False):
            print(f"Loading teacher model from: {self.hparams['distillation_config']['teacher_checkpoint_path']}")
            self._external_models['teacher'] = ASRLightningModule.load_from_checkpoint(
                self.hparams['distillation_config']['teacher_checkpoint_path'],
                map_location=self.device
            )
            self._external_models['teacher'].requires_grad_(False)
            self._external_models['teacher'].eval()
            self._external_models['teacher'].to(self.device)
            print("Teacher model loaded successfully")
            
        self._setup_called = True

    @property
    def teacher(self):
        return self._external_models.get('teacher', None)

    @torch.no_grad()
    def extract_indices(self, wavs, prepare_streaming=False):
        """
        #### Input ####
        wavs (List[np.ndarray] or Tensor): A batch of raw audio signals @16kHz. 
            This can be a list of 1D numpy arrays or a tensor of shape [batch_size, sequence_length].
            
        #### Output ####
        indices: Integer indices, shape [B, T], each in [0, level_1*level_2*...*level_L).
        features: in shape [B, T, d_model], last layer output before quantization
        feature_lengths: in shape [B, ]
        
        """ 
        self.asr_model.eval()
        self.ssl_extractor.eval()
        
        # 1) Extract SSL features
        out_dict = self.ssl_extractor(wavs, downstream_mode=True) 
        features, feature_lengths = out_dict["features"], out_dict["feature_lengths"]
        
        # 2) Generate masks
        mask_dict = generate_src_padding_masks(
            src_lengths=feature_lengths,
            device=features.device
        )
        
        # 3) Forward through first part of encoder
        x = self.asr_model.input_linear(features)       # [B, T, d_model]
        x = self.asr_model.pos_encoder(x)
        x = x.transpose(0, 1)                 # [T, B, d_model]
        src_mask = None

        for i in range(self.asr_model.fsq_layer_idx):
            if prepare_streaming:
                x = self.asr_model.encoder.layers[i](x, src_mask=src_mask, src_key_padding_mask=mask_dict['src_key_padding_mask'], prepare_streaming=True)
            else:
                x = self.asr_model.encoder.layers[i](x, src_mask=src_mask, src_key_padding_mask=mask_dict['src_key_padding_mask'])
            if self.asr_model.use_convnext:
                x = self.asr_model.convnext_blocks[i](x.transpose(0, 1), prepare_streaming=prepare_streaming).transpose(0, 1)
            
        # 4) FSQ 
        x = x.transpose(0, 1) # [B, T, d_model]
        out_features = x
        
        quant, indices = self.asr_model.fsq(x) # [B, T, d_model] plus indices [B, T]
        
        return {
            'indices': indices,
            'features': out_features,
            'feature_lengths': feature_lengths
        }
    
    def clear_cache(self):
        self.ssl_extractor.clear_cache()
        self.asr_model.clear_cache()
