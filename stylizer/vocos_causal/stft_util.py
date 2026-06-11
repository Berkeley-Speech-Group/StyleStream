
import torch
import torchaudio

# Set parameters to obtain the desired shape
n_fft = 1024
hop_length = 320
win_length = 1024
spec_abs_exponent = 0.5
spec_factor = 0.15

sample_rate = 16000
# number of spectral bins in your original mag: n_fft//2 + 1
n_freq = n_fft // 2 + 1
n_mels = 100

# Create a window; here we use a Hann window as an example.
window = torch.hann_window(win_length)


# build a MelScale that maps from linear-freq (n_freq bins) → n_freq mel bins
mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
).eval()

# 1) pseudo-inverse mel → linear spec magnitude
inverse_mel = torchaudio.transforms.InverseMelScale(
    n_stft=n_fft//2 + 1,
    n_mels=n_mels,
    sample_rate=sample_rate,
).eval()

def stft_ri(wf):
    """
    Compute the STFT of waveform `wf` using self.n_fft, self.hop_length, self.win_length, and self.window,
    returning a tensor with real and imaginary parts concatenated along the frequency axis.
    Input: wf of shape [B, 1, T]; Output: [B, 2*F, T].
    """
    wf_stft = torch.view_as_real(torch.stft(
        wf,
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length, 
        window=window.to(wf.device),
        center=True, 
        onesided=True, 
        return_complex=True,
    ))
    return torch.cat((wf_stft[..., 0], wf_stft[..., 1]), dim=1)


def istft_ri(wf_stft):
    """
    Compute the iSTFT of spectrogram `wf_stft` using self.n_fft, self.hop_length, self.win_length, and self.window,
    returning the original waveform wf.
    Input: wf of shape [B, 2*F, T]; Output: [B, 1, T].
    """
    wf = torch.istft(
        torch.view_as_complex(torch.stack(torch.chunk(wf_stft, 2, dim=1), dim=-1)),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length).to(wf_stft.device),
        center=True,
        onesided=True,
    )
    return wf

def stft_mp(wf):
    """
    Compute the STFT of waveform `wf` using self.n_fft, self.hop_length, self.win_length, and self.window,
    returning a tensor with magnitude and phase parts concatenated along the frequency axis.
    Input: wf of shape [B, 1, T]; Output: [B, 2*F, T].
    """
    wf_stft = torch.stft(
        wf,
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length, 
        window=window.to(wf.device),
        center=True, 
        onesided=True, 
        return_complex=True,
    )
    return torch.cat((torch.log(torch.clamp(torch.abs(wf_stft), min=1e-6)), torch.atanh(torch.clamp(torch.angle(wf_stft) / torch.pi, min=-1+1e-6, max=1-1e-6))), dim=1)


def istft_mp(wf_stft):
    """
    Compute the iSTFT of spectrogram `wf_stft` using self.n_fft, self.hop_length, self.win_length, and self.window,
    returning the original waveform wf.
    Input: wf of shape [B, 2*F, T]; Output: [B, 1, T].
    """
    M, P = torch.chunk(wf_stft, 2, dim=1)
    M = torch.exp(M)
    P = torch.tanh(P) * torch.pi

    wf = torch.istft(
        torch.polar(M, P),
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length, 
        window=torch.hann_window(win_length).to(wf_stft.device),
        center=True,
        onesided=True,
    )
    return wf

def stft_vocos(wf):
    """
    Compute the STFT of waveform `wf` using self.n_fft, self.hop_length, self.win_length, and self.window,
    returning a tensor with magnitude and phase parts concatenated along the frequency axis.
    Input: wf of shape [B, t]; Output: [log_mag, phase] both have shape [B, F, T]
    """
    wf_stft = torch.stft(
        wf,
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length, 
        window=window.to(wf.device),
        center=True, 
        onesided=True, 
        return_complex=True,
    )

    log_mag = torch.log(torch.clamp(torch.abs(wf_stft), min=1e-6))
    phase = torch.angle(wf_stft)

    return log_mag, phase

def stft_mel(wf):
    """
    wf: Tensor of shape [B, 1, T] or [B, T]
    returns: log-mel spec of shape [B, n_mels, T_frames], on same device as wf
    """
    mel_spec_device = mel_spec.to(wf.device)
    mel = mel_spec_device(wf)
    mel = torch.log(torch.clamp(mel, min=1e-6))

    return mel

def get_log_mel(wf):
    """
    wf: Tensor of shape [B, T]
    returns: log-mel spec of shape [B, n_mels, T_frames], on same device as wf
    """
    mel_spec_device = mel_spec.to(wf.device)
    mel = mel_spec_device(wf)
    log_mel = torch.log(torch.clamp(mel, min=1e-6))

    return log_mel

def get_mel(wf):
    """
    wf: Tensor of shape [B, T]
    returns: mel spec of shape [B, n_mels, T_frames], on same device as wf
    """
    mel_spec_device = mel_spec.to(wf.device)
    mel = mel_spec_device(wf)

    return mel

def get_mel_transform():
    return torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        ).eval()

def istft_vocos(log_mag, raw_phase):
    """
    Compute the iSTFT, returns the original waveform wf.
    Input: [log_mag, raw_phase] both have shape [B, F, T]; Output: [B, t].
    """
    mag = torch.exp(log_mag)

    wf = torch.istft(
        torch.polar(mag, raw_phase),
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length, 
        window=torch.hann_window(win_length).to(mag.device),
        center=True,
        onesided=True,
    )
    
    return wf

def stft_complex(wf):
    """
    Compute the STFT of waveform `wf` using self.n_fft, self.hop_length, self.win_length, and self.window,
    returning a complex tensor with scaling according to FlowAVSE.
    Input: wf of shape [B, 1, T]; Output: wf_stft of shape [B, F, T]
    """
    e = spec_abs_exponent

    wf_stft = torch.stft(
        wf,
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length, 
        window=window.to(wf.device),
        center=True, 
        onesided=True, 
        return_complex=True,
    )
    wf_stft = wf_stft.abs()**e * torch.exp(1j * wf_stft.angle())

    return wf_stft

def istft_complex(wf_stft):
    """
    Compute the iSTFT of spectrogram `wf_stft` using self.n_fft, self.hop_length, self.win_length, and self.window,
    returning the original waveform wf.
    Input: wf_stft of shape [B, F, T]; Output: [B, 1, T].
    """
    e = spec_abs_exponent
    wf_stft = wf_stft.abs()**(1/e) * torch.exp(1j * wf_stft.angle())

    wf = torch.istft(
        wf_stft,
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length, 
        window=torch.hann_window(win_length).to(wf_stft.device),
        center=True,
        onesided=True,
    )

    return wf


def mel_to_log_magnitude(log_mel: torch.Tensor) -> torch.Tensor:
    """
    log_mel:  Tensor of shape [B, n_mels, T] (or [n_mels, T])  
            assumed to be natural-log of mel-amplitude: log(mel_spec + eps)
    returns:  log-magnitude spec of shape [B, n_stft, T] on same device
    """
    device = log_mel.device
    # 1) undo the log → get mel-amplitude
    mel_amp = torch.exp(log_mel)

    # 2) invert mel → linear-freq amplitude
    inverse_mel_device = inverse_mel.to(device)
    spec_amp = inverse_mel_device(mel_amp)

    # 3) log & clamp → log-magnitude
    log_mag = torch.log(torch.clamp(spec_amp, min=1e-6))

    return log_mag

