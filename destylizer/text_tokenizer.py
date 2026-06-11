import torch.nn as nn
import torchaudio
import torch
import string
import numpy as np
import jiwer
import soundfile as sf
import librosa
import re

class GraphemeTokenizer(nn.Module):
    def __init__(self, include_space=True, include_apostrophe=True, include_dash=False, **kwargs):
        super().__init__()
        self.charactors = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
                           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
                           'u', 'v', 'w', 'x', 'y', 'z']
        self.punctuations = string.punctuation + " "
        
        if include_space:
            self.charactors.append(' ')
            self.punctuations = self.punctuations.replace(" ", "")
        if include_apostrophe:
            self.charactors.append("'")
            self.punctuations = self.punctuations.replace("'", "")
        if include_dash:
            self.charactors.append("-")
            self.punctuations = self.punctuations.replace("-", "")
            
        self.ch2idx = {ch:i for i, ch in enumerate(self.charactors)}
        self._pad_id = len(self.charactors) 
        self._bos_id = len(self.charactors) + 1
        self._eos_id = len(self.charactors) + 2

        self.charactors += ["PAD", "BOS", "EOS"]
        self.vocab = self.charactors
        self.vocab_size = len(self.charactors)
        self.normalization = jiwer.Compose([
            jiwer.ToLowerCase(),  # Convert text to lowercase
            jiwer.RemoveWhiteSpace(replace_by_space=True),  # Normalize whitespace
            jiwer.RemoveMultipleSpaces(),  # Replace multiple spaces with a single space
            jiwer.Strip()  # Strip leading and trailing whitespace
        ])

    def convert_numbers_to_words(self, text):
        import num2words
        return re.sub(r'\d+', lambda match: num2words(match.group(0)), text)
    
    def text_normalization(self, text):
        text = self.convert_numbers_to_words(text)
        for char in self.punctuations:
            text = text.replace(char, "")
        return self.normalization(text)
    
    def __call__(self, texts, **kwargs):
        texts = [self.text_normalization(text) for text in texts]
        token_idxs = [[self.ch2idx[t] for t in text] for text in texts]
        token_idxs = [torch.tensor([self.bos_id()] + idxs + [self.eos_id()]) for idxs in token_idxs]
        token_lengths = torch.tensor([len(idxs) for idxs in token_idxs])
        
        token_idxs = nn.utils.rnn.pad_sequence(token_idxs, batch_first=True, padding_value=self._pad_id)
        return token_idxs, token_lengths, texts
    
    def get_decoder(self):
        return lambda tokens: ''.join([self.charactors[i] for i in tokens])
    
    def get_remove_tokens(self):
        return [self._pad_id, self._bos_id, self._eos_id]
    
    def pad_id(self):
        return self._pad_id
    
    def bos_id(self):
        return self._bos_id
    
    def eos_id(self):
        return self._eos_id
