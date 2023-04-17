import random
import numpy

def time_warp(x, max_time_warp=80, inplace=False, mode="PIL"):
    """time warp for spec augment

    move random center frame by the random width ~ uniform(-window, window)
    :param numpy.ndarray x: spectrogram (time, freq)
    :param int max_time_warp: maximum time frames to warp
    :param bool inplace: overwrite x with the result
    :param str mode: "PIL" (default, fast, not differentiable) or "sparse_image_warp"
        (slow, differentiable)
    :returns numpy.ndarray: time warped spectrogram (time, freq)
    """
    window = max_time_warp
    if mode == "PIL":
        from PIL import Image
        from PIL.Image import BICUBIC
        t = x.shape[0]
        if t - window <= window:
            return x
        # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
        center = random.randrange(window, t - window)
        warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1

        left = Image.fromarray(x[:center]).resize((x.shape[1], warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped), BICUBIC)
        if inplace:
            x[:warped] = left
            x[warped:] = right
            return x
        return numpy.concatenate((left, right), 0)
    elif mode == "sparse_image_warp":
        import torch

        from espnet.utils import spec_augment

        # TODO(karita): make this differentiable again
        return spec_augment.time_warp(torch.from_numpy(x), window).numpy()
    else:
        raise NotImplementedError(
            "unknown resize mode: "
            + mode
            + ", choose one from (PIL, sparse_image_warp)."
        )

def freq_mask(x, F=30, n_mask=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument

    :param numpy.ndarray x: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = x
    else:
        cloned = x.copy()

    num_mel_channels = cloned.shape[1]
    fs = numpy.random.randint(0, F, size=(n_mask, 2))

    for f, mask_end in fs:
        f_zero = random.randrange(0, num_mel_channels - f)
        mask_end += f_zero

        # avoids randrange error if values are equal and range is empty
        if f_zero == f_zero + f:
            continue

        if replace_with_zero:
            cloned[:, f_zero:mask_end] = 0
        else:
            cloned[:, f_zero:mask_end] = cloned.mean()
    return cloned

def time_mask(spec, T=40, n_mask=2, replace_with_zero=True, inplace=False):
    """freq mask for spec agument

    :param numpy.ndarray spec: (time, freq)
    :param int n_mask: the number of masks
    :param bool inplace: overwrite
    :param bool replace_with_zero: pad zero on mask if true else use mean
    """
    if inplace:
        cloned = spec
    else:
        cloned = spec.copy()
    len_spectro = cloned.shape[0]
    ts = numpy.random.randint(0, T, size=(n_mask, 2))
    for t, mask_end in ts:
        # avoid randint range error
        if len_spectro - t <= 0:
            continue
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            continue

        mask_end += t_zero
        if replace_with_zero:
            cloned[t_zero:mask_end] = 0
        else:
            cloned[t_zero:mask_end] = cloned.mean()
    return cloned

