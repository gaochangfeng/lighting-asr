import torch
import torchaudio
import librosa
import numpy
from lasr.utils.register import Register
from lasr.utils.specaugment import time_mask, time_warp, freq_mask

register_trans = Register()

@register_trans.register("avgchannel")
def AverageChanl(wav):
    if len(wav.shape) == 2:
        return numpy.average(wav, axis = 1)
    return wav

@register_trans.register("resample:16k")
def ReSample(wav, ssr, tsr=16000):
    if ssr != tsr:
        wav = librosa.resample(wav, ssr, tsr, res_type="kaiser_fast")
    return wav

@register_trans.register("norm")
def VoiceNorm(wav):
    max_value = max([abs(x) for x in wav])
    wav = [x/(max_value + 1e-9) for x in wav]
    wav = numpy.array(wav)    
    return wav

@register_trans.register("soxspeed")
def SoxSpeedPt(wav, sp = [1, 1.1, 0.9], sample_rate=16000):
    sp_ratio = numpy.random.choice(sp)
    if sp_ratio == 1:
        return wav
    else:
        import sox
        tfm = sox.Transformer()
        tfm.speed(sp_ratio)
        array_out = tfm.build_array(input_array=wav, sample_rate_in=sample_rate)
        return array_out


@register_trans.register("fbank:80")
def WavToKaldiFbank(
    wavform,
    blackman_coeff: float = 0.42, 
    channel: int = -1, 
    dither: float = 0.0, 
    energy_floor: float = 1.0, 
    frame_length: float = 25.0, 
    frame_shift: float = 10.0, 
    high_freq: float = 0.0, 
    htk_compat: bool = False, 
    low_freq: float = 20.0, 
    min_duration: float = 0.0, 
    num_mel_bins: int = 80, 
    preemphasis_coefficient: float = 0.97, 
    raw_energy: bool = True, 
    remove_dc_offset: bool = True, 
    round_to_power_of_two: bool = True, 
    sample_frequency: float = 16000.0, 
    snip_edges: bool = True, 
    subtract_mean: bool = False, 
    use_energy: bool = False, 
    use_log_fbank: bool = True, 
    use_power: bool = True, 
    vtln_high: float = -500.0, 
    vtln_low: float = 100.0, 
    vtln_warp: float = 1.0, 
    window_type: str = 'povey',
    audio_bit: int = 16
    ):

    wavform = torch.from_numpy(wavform.copy()).unsqueeze(0).float()
    wavform = wavform * 2 ** (audio_bit - 1)
    fbank = torchaudio.compliance.kaldi.fbank(
        wavform, 
        blackman_coeff=blackman_coeff, 
        channel=channel, 
        dither=dither, 
        energy_floor=energy_floor, 
        frame_length=frame_length, 
        frame_shift=frame_shift, 
        high_freq=high_freq, 
        htk_compat=htk_compat, 
        low_freq=low_freq, 
        min_duration=min_duration, 
        num_mel_bins=num_mel_bins, 
        preemphasis_coefficient=preemphasis_coefficient, 
        raw_energy=raw_energy, 
        remove_dc_offset=remove_dc_offset, 
        round_to_power_of_two=round_to_power_of_two, 
        sample_frequency=sample_frequency, 
        snip_edges=snip_edges, 
        subtract_mean=subtract_mean, 
        use_energy=use_energy, 
        use_log_fbank=use_log_fbank, 
        use_power=use_power, 
        vtln_high=vtln_high, 
        vtln_low=vtln_low, 
        vtln_warp=vtln_warp, 
        window_type=window_type,
        )
    
    return fbank.detach().numpy()

@register_trans.register("specaug")
def SpecAugment(
    x,
    resize_mode="PIL",
    max_time_warp=5,
    max_freq_width=27,
    n_freq_mask=2,
    max_time_width=40,
    n_time_mask=2,
    inplace=True,
    replace_with_zero=False,
):
    """spec agument

    apply random time warping and time/freq masking
    default setting is based on LD (Librispeech double) in Table 2
        https://arxiv.org/pdf/1904.08779.pdf

    :param numpy.ndarray x: (time, freq)
    :param str resize_mode: "PIL" (fast, nondifferentiable) or "sparse_image_warp"
        (slow, differentiable)
    :param int max_time_warp: maximum frames to warp the center frame in spectrogram (W)
    :param int freq_mask_width: maximum width of the random freq mask (F)
    :param int n_freq_mask: the number of the random freq mask (m_F)
    :param int time_mask_width: maximum width of the random time mask (T)
    :param int n_time_mask: the number of the random time mask (m_T)
    :param bool inplace: overwrite intermediate array
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    assert x.ndim == 2
    x = time_warp(x, max_time_warp, inplace=inplace, mode=resize_mode)
    x = freq_mask(
        x,
        max_freq_width,
        n_freq_mask,
        inplace=inplace,
        replace_with_zero=replace_with_zero,
    )
    x = time_mask(
        x,
        max_time_width,
        n_time_mask,
        inplace=inplace,
        replace_with_zero=replace_with_zero,
    )
    return x

