import numpy as np
import torch
import torch.nn.functional as F


def inverse_softplus(x, beta, scale=1):
    # log(exp(scale*x)-1)/scale
    out = x / scale
    mask = x * beta < 20 * scale
    out[mask] = torch.log(torch.exp(beta * out[mask]) - 1 + 1e-10) / beta
    return out


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(-1, img1.shape[-1]).mean(0, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def get_expon_lr_func(
    lr_init,
    lr_final,
    warmup_steps=0,
    max_steps=1_000,
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if warmup_steps and step < warmup_steps:
            return lr_init * step / warmup_steps
        elif step > max_steps:
            return 0
        t = np.clip((step - warmup_steps) / (max_steps - warmup_steps), 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp

    return helper


def get_cosine_lr_func(
    lr_init,
    lr_final,
    warmup_steps=0,
    max_steps=10_000,
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if warmup_steps and step < warmup_steps:
            return lr_init * step / warmup_steps
        elif step > max_steps:
            return 0.0
        lr_cos = lr_final + 0.5 * (lr_init - lr_final) * (
            1
            + np.cos(np.pi * (step - warmup_steps) / (max_steps - warmup_steps))
        )
        return lr_cos

    return helper


def gauss_conv3d_separable(x, gauss_1d, pad):
    """Apply separable 3D Gaussian smoothing via three 1D conv3d passes.

    Uses replicate boundary padding to avoid darkening at volume borders.

    Args:
        x: (1, 1, D, H, W) float tensor on any device
        gauss_1d: (ks,) normalized 1D Gaussian kernel
        pad: kernel half-width (ks // 2)

    Returns:
        Smoothed tensor, same shape as x.
    """
    ws = gauss_1d.shape[0]
    # F.pad order for 5-D: (W_left, W_right, H_top, H_bottom, D_front, D_back)
    kx = gauss_1d.reshape(1, 1, ws, 1, 1)   # convolves along D
    ky = gauss_1d.reshape(1, 1, 1, ws, 1)   # convolves along H
    kz = gauss_1d.reshape(1, 1, 1, 1, ws)   # convolves along W
    x = F.conv3d(F.pad(x, (0, 0, 0, 0, pad, pad), mode="replicate"), kx)
    x = F.conv3d(F.pad(x, (0, 0, pad, pad, 0, 0), mode="replicate"), ky)
    x = F.conv3d(F.pad(x, (pad, pad, 0, 0, 0, 0), mode="replicate"), kz)
    return x
