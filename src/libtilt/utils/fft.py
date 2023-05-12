from typing import Sequence, List, Tuple

import einops
import numpy as np
import torch


def rfft_shape_from_signal_shape(input_shape: Sequence[int]) -> Tuple[int]:
    """Get the output shape of an rfft on an input with input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)


def fft_center(
    grid_shape: Tuple[int, ...], fftshifted: bool, rfft: bool
) -> torch.Tensor:
    """Return the indices of the fftshifted DFT center."""
    fft_center = torch.zeros(size=(len(grid_shape),))
    grid_shape = torch.as_tensor(grid_shape).float()
    if rfft is True:
        grid_shape = torch.tensor(rfft_shape_from_signal_shape(grid_shape))
    if fftshifted is True:
        fft_center = torch.divide(grid_shape, 2, rounding_mode='floor')
    if rfft is True:
        fft_center[-1] = 0
    return fft_center


def construct_fftfreq_grid_2d(image_shape: Sequence[int], rfft: bool) -> torch.Tensor:
    """Construct a grid of DFT sample frequencies for a 2D image.

    Parameters
    ----------
    image_shape: Sequence[int]
        A 2D shape `(h, w)` of the input image for which a grid of DFT sample frequencies
        should be calculated.
    rfft: bool
        Indicates whether the frequency grid is for a real fft (rfft).

    Returns
    -------
    frequency_grid: torch.Tensor
        `(h, w, 2)` array of DFT sample frequencies.
        Order of frequencies in the last dimension corresponds to the order of
        the two dimensions of the grid.
    """
    last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    h, w = image_shape
    freq_y = torch.fft.fftfreq(h)
    freq_x = last_axis_frequency_func(w)
    h, w = rfft_shape_from_signal_shape(image_shape) if rfft is True else image_shape
    freq_yy = einops.repeat(freq_y, 'h -> h w', w=w)
    freq_xx = einops.repeat(freq_x, 'w -> h w', h=h)
    return einops.rearrange([freq_yy, freq_xx], 'freq h w -> h w freq')


def construct_fftfreq_grid_3d(image_shape: Sequence[int], rfft: bool) -> torch.Tensor:
    """Construct a grid of DFT sample frequencies for a 3D image.

    Parameters
    ----------
    image_shape: Sequence[int]
        A 3D shape `(d, h, w)` of the input image for which a grid of DFT sample frequencies
        should be calculated.
    rfft: bool
        Controls Whether the frequency grid is for a real fft (rfft).

    Returns
    -------
    frequency_grid: torch.Tensor
        `(h, w, 3)` array of DFT sample frequencies.
        Order of frequencies in the last dimension corresponds to the order of dimensions
        of the grid.
    """
    last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    d, h, w = image_shape
    freq_z = torch.fft.fftfreq(d)
    freq_y = torch.fft.fftfreq(h)
    freq_x = last_axis_frequency_func(w)
    d, h, w = rfft_shape_from_signal_shape(image_shape) if rfft is True else image_shape
    freq_zz = einops.repeat(freq_z, 'd -> d h w', h=h, w=w)
    freq_yy = einops.repeat(freq_y, 'h -> d h w', d=d, w=w)
    freq_xx = einops.repeat(freq_x, 'w -> d h w', d=d, h=h)
    return einops.rearrange([freq_zz, freq_yy, freq_xx], 'freq h w -> h w freq')


def rfft_to_symmetrised_dft_2d(rfft: torch.Tensor) -> torch.Tensor:
    """Construct a symmetrised discrete Fourier transform from an rfft.

    The symmetrised discrete Fourier transform contains a full FFT with components at
    the Nyquist frequency repeated on both sides. This yields a spectrum which is
    symmetric around the DC component of the FFT, useful for some applications.

    This is only valid for rffts of cubic inputs with even sidelength.
    Input should be the result of calling rfftn on input data.

    1D example:
    - rfftfreq: `[0.0000, 0.1667, 0.3333, 0.5000]`
    - fftshifted fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333]`
    - symmetrised fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333,  0.5000]`

    Parameters
    ----------
    rfft: torch.Tensor
        `(h, w)` or `(b, h, w)` array containing an rfft of square 2D image data
        with an even sidelength.

    Returns
    -------
    output: torch.Tensor
        `(h+1, w+1)` or `(b, h+1, w+1)` symmetrised DFT constructed from the input
        `rfft`.
    """
    r = rfft.shape[-2]  # lenght of h is unmodified by rfft
    if rfft.ndim == 2:
        output = torch.zeros((r + 1, r + 1), dtype=torch.complex64)
    elif rfft.ndim == 3:
        b = rfft.shape[0]
        output = torch.zeros((b, r + 1, r + 1), dtype=torch.complex64)
    # fftshift full length dims to center DC component
    dc = r // 2
    rfft = torch.fft.fftshift(rfft, dim=(-2,))
    output[..., :-1, dc:] = rfft  # place rfft in output
    output[..., -1, dc:] = rfft[..., 0, :]  # replicate components at Nyquist
    # fill redundant half
    output[..., :, :dc] = torch.flip(torch.conj(output[..., :, dc + 1:]), dims=(-2, -1))
    return output


def rfft_to_symmetrised_dft_3d(rfft: torch.Tensor) -> torch.Tensor:
    """Construct a symmetrised discrete Fourier transform from an rfft.

    The symmetrised discrete Fourier transform contains a full FFT with components at
    the Nyquist frequency repeated on both sides. This yields a spectrum which is
    symmetric around the DC component of the FFT.

    This is only valid for rffts of cubic inputs with even sidelength.
    Input should be the result of calling rfft on input data.

    1D example:
    - rfftfreq: `[0.0000, 0.1667, 0.3333, 0.5000]`
    - fftshifted fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333]`
    - symmetrised fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333,  0.5000]`
    """
    r = rfft.shape[-3]  # input dim length
    if rfft.ndim == 3:
        output = torch.zeros((r + 1, r + 1, r + 1), dtype=torch.complex64)
    elif rfft.ndim == 4:
        b = rfft.shape[0]
        output = torch.zeros((b, r + 1, r + 1, r + 1), dtype=torch.complex64)
    # fftshift full length dims (i.e. not -1) to center DC component
    rfft = torch.fft.fftshift(rfft, dim=(-3, -2))
    # place rfft in output
    dc = r // 2  # index for DC component
    output[..., :-1, :-1, dc:] = rfft
    # replicate components at nyquist (symmetrise)
    output[..., :-1, -1, dc:] = rfft[..., :, 0, :]
    output[..., -1, :-1, dc:] = rfft[..., 0, :, :]
    output[..., -1, -1, dc:] = rfft[..., 0, 0, :]
    # fill redundant half-spectrum
    output[..., :, :, :dc] = torch.flip(torch.conj(output[..., :, :, dc + 1:]),
                                        dims=(-3, -2, -1))
    return output


def symmetrised_dft_to_dft_2d(dft: torch.Tensor, inplace: bool = True):
    """Desymmetrise a symmetrised 2D discrete Fourier transform.

    Turn a symmetrised DFT into a normal DFT by averaging duplicated
    components at the Nyquist frequency.

    1D example:
    - fftshifted fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333]`
    - symmetrised fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333,  0.5000]`
    - desymmetrised fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333]`

    Parameters
    ----------
    dft: torch.Tensor
        `(b, h, w)` or `(h, w)` array containing symmetrised discrete Fourier transform(s)
    inplace: bool
        Controls whether the operation is applied in place on the existing array.

    Returns
    -------

    """
    if inplace is False:
        dft = dft.clone()
    dft[..., :, 0] = (0.5 * dft[..., :, 0]) + (0.5 * dft[..., :, -1])
    dft[..., 0, :] = (0.5 * dft[..., 0, :]) + (0.5 * dft[..., -1, :])
    return dft[..., :-1, :-1]


def symmetrised_dft_to_rfft_2d(dft: torch.Tensor, inplace: bool = True):
    if dft.ndim == 2:
        dft = einops.rearrange(dft, 'h w -> 1 h w')
    _, h, w = dft.shape
    dc = h // 2
    r = dc + 1  # start of right half-spectrum
    rfft = dft if inplace is True else torch.clone(dft)
    # average hermitian symmetric halves
    rfft[..., :, r:] *= 0.5
    rfft[..., :, r:] += 0.5 * torch.flip(torch.conj(rfft[..., :, :dc]), dims=(-2, -1))
    # average leftover redundant nyquist
    rfft[..., 0, r:] = (0.5 * rfft[..., 0, r:]) + (0.5 * rfft[..., -1, r:])
    # return without redundant nyquist
    rfft = rfft[..., :-1, dc:]
    return torch.fft.ifftshift(rfft, dim=-2)


def symmetrised_dft_to_dft_3d(dft: torch.Tensor, inplace: bool = True):
    """Desymmetrise a symmetrised 3D discrete Fourier transform.

    Turn a symmetrised DFT into a normal DFT by averaging duplicated
    components at the Nyquist frequency.

    1D example:
    - fftshifted fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333]`
    - symmetrised fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333,  0.5000]`
    - desymmetrised fftfreq: `[-0.5000, -0.3333, -0.1667,  0.0000,  0.1667,  0.3333]`

    Parameters
    ----------
    dft: torch.Tensor
        `(b, h, w)` or `(h, w)` array containing symmetrised discrete Fourier transform(s)
    inplace: bool
        Controls whether the operation is applied in place on the existing array.

    Returns
    -------

    """
    if inplace is False:
        dft = dft.clone()
    dft[..., :, :, 0] = (0.5 * dft[..., :, :, 0]) + (0.5 * dft[..., :, :, -1])
    dft[..., :, 0, :] = (0.5 * dft[..., :, 0, :]) + (0.5 * dft[..., :, -1, :])
    dft[..., 0, :, :] = (0.5 * dft[..., 0, :, :]) + (0.5 * dft[..., -1, :, :])
    return dft[..., :-1, :-1, :-1]


def _indices_centered_on_dc_for_shifted_rfft(
    rfft_shape: Sequence[int]
) -> torch.Tensor:
    rfft_shape = torch.tensor(rfft_shape)
    rfftn_dc_idx = torch.div(rfft_shape, 2, rounding_mode='floor')
    rfftn_dc_idx[-1] = 0
    rfft_indices = torch.tensor(np.indices(rfft_shape))  # (c, (d), h, w)
    rfft_indices = einops.rearrange(rfft_indices, 'c ... -> ... c')
    return rfft_indices - rfftn_dc_idx


def _distance_from_dc_for_shifted_rfft(rfft_shape: Sequence[int]) -> torch.Tensor:
    centered_indices = _indices_centered_on_dc_for_shifted_rfft(rfft_shape)
    return einops.reduce(centered_indices ** 2, '... c -> ...', reduction='sum') ** 0.5


def _indices_centered_on_dc_for_shifted_dft(
    dft_shape: Sequence[int], rfft: bool
) -> torch.Tensor:
    if rfft is True:
        return _indices_centered_on_dc_for_shifted_rfft(dft_shape)
    dft_indices = torch.tensor(np.indices(dft_shape)).float()
    dft_indices = einops.rearrange(dft_indices, 'c ... -> ... c')
    dc_idx = fft_center(dft_shape, fftshifted=True, rfft=False)
    return dft_indices - dc_idx


def _distance_from_dc_for_shifted_dft(
    dft_shape: Sequence[int], rfft: bool
) -> torch.Tensor:
    idx = _indices_centered_on_dc_for_shifted_dft(dft_shape, rfft=rfft)
    return einops.reduce(idx ** 2, '... c -> ...', reduction='sum') ** 0.5


def indices_centered_on_dc_for_dft(
    dft_shape: Sequence[int], rfft: bool, fftshifted: bool
) -> torch.Tensor:
    dft_indices = _indices_centered_on_dc_for_shifted_dft(dft_shape, rfft=rfft)
    dft_indices = einops.rearrange(dft_indices, '... c -> c ...')
    if fftshifted is False:
        dims_to_shift = tuple(torch.arange(start=-1 * len(dft_shape), end=0, step=1))
        dims_to_shift = dims_to_shift[:-1] if rfft is True else dims_to_shift
        dft_indices = torch.fft.ifftshift(dft_indices, dim=dims_to_shift)
    return einops.rearrange(dft_indices, 'c ... -> ... c')


def distance_from_dc_for_dft(
    dft_shape: Sequence[int], rfft: bool, fftshifted: bool
) -> torch.Tensor:
    idx = indices_centered_on_dc_for_dft(dft_shape, rfft=rfft, fftshifted=fftshifted)
    return einops.reduce(idx**2, '... c -> ...', reduction='sum') ** 0.5

