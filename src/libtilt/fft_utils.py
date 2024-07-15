import functools
from typing import Sequence, Tuple
from itertools import combinations, permutations

import einops
import numpy as np
import torch
from torch.nn import functional as F


def rfft_shape(input_shape: Sequence[int]) -> Tuple[int]:
    """Get the output shape of an rfft on an input with input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)


def dft_center(
    image_shape: Tuple[int, ...],
    rfft: bool,
    fftshifted: bool,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Return the position of the DFT center for a given input shape."""
    _rfft_shape = rfft_shape(image_shape)
    fft_center = torch.zeros(size=(len(image_shape),), device=device)
    image_shape = torch.as_tensor(image_shape).float()
    if rfft is True:
        image_shape = torch.tensor(_rfft_shape, device=device)
    if fftshifted is True:
        fft_center = torch.divide(image_shape, 2, rounding_mode='floor')
    if rfft is True:
        fft_center[-1] = 0
    return fft_center.long()


def fftshift_2d(input: torch.Tensor, rfft: bool):
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-2, -1))
    else:
        output = torch.fft.fftshift(input, dim=(-2,))
    return output


def ifftshift_2d(input: torch.Tensor, rfft: bool):
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-2, -1))
    else:
        output = torch.fft.ifftshift(input, dim=(-2,))
    return output


def fftshift_3d(input: torch.Tensor, rfft: bool):
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-3, -2, -1))
    else:
        output = torch.fft.fftshift(input, dim=(-3, -2,))
    return output


def ifftshift_3d(input: torch.Tensor, rfft: bool):
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-3, -2, -1))
    else:
        output = torch.fft.ifftshift(input, dim=(-3, -2,))
    return output


@functools.lru_cache(maxsize=1)
def fft_sizes(lower_bound: int = 0) -> torch.LongTensor:
    """FFT input sizes which are factorisable into small primes."""
    # powers of two
    powers_of_two = torch.pow(2, torch.arange(2, 33))

    # numbers factorisable into the smallest primes 2, 3, and 5
    powers = [
        torch.tensor(list(set(permutations(combination))))
        for combination
        in combinations(range(6), 3)
    ]
    i, j, k = einops.rearrange(powers, 'b1 b2 powers -> powers (b1 b2)')
    prime_factors = [
        torch.pow(2, exponent=i),
        torch.pow(3, exponent=j),
        torch.pow(5, exponent=k),
    ]
    prime_factors = einops.rearrange(prime_factors, 'f b -> b f')
    fft_sizes = einops.reduce(prime_factors, 'b f -> b', reduction='prod')
    fft_sizes = torch.cat([powers_of_two, fft_sizes])
    fft_sizes, _ = torch.sort(fft_sizes)
    return fft_sizes[fft_sizes >= lower_bound]


def best_fft_size(
    lower_bound: int, target_fftfreq: float, maximum_relative_error: float
) -> int:
    good_sizes = fft_sizes(lower_bound=lower_bound)
    for fft_size in good_sizes:
        delta_fftfreq = 1 / fft_size
        relative_error = (target_fftfreq % delta_fftfreq) / target_fftfreq
        if relative_error < maximum_relative_error:
            return fft_size
    raise ValueError("No best size found.")


def fftfreq_to_spatial_frequency(
    frequencies: torch.Tensor, spacing: float
) -> torch.Tensor:
    """Convert frequencies in cycles per pixel to cycles per unit distance."""
    # cycles/px * px/distance = cycles/distance
    return torch.as_tensor(frequencies, dtype=torch.float32) * (1 / spacing)


def spatial_frequency_to_fftfreq(
    frequencies: torch.Tensor, spacing: float
) -> torch.Tensor:
    """Convert frequencies in cycles per unit distance to cycles per pixel."""
    # cycles/distance * distance/px = cycles/px
    return torch.as_tensor(frequencies, dtype=torch.float32) * spacing


def _rfft_to_symmetrised_dft_2d(rfft: torch.Tensor) -> torch.Tensor:
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


def _rfft_to_symmetrised_dft_3d(dft: torch.Tensor) -> torch.Tensor:
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
    r = dft.shape[-3]  # input dim length
    if dft.ndim == 3:
        output = torch.zeros((r + 1, r + 1, r + 1), dtype=torch.complex64)
    elif dft.ndim == 4:
        b = dft.shape[0]
        output = torch.zeros((b, r + 1, r + 1, r + 1), dtype=torch.complex64)
    # fftshift full length dims (i.e. not -1) to center DC component
    dft = torch.fft.fftshift(dft, dim=(-3, -2))
    # place rfft in output
    dc = r // 2  # index for DC component
    output[..., :-1, :-1, dc:] = dft
    # replicate components at nyquist (symmetrise)
    output[..., :-1, -1, dc:] = dft[..., :, 0, :]
    output[..., -1, :-1, dc:] = dft[..., 0, :, :]
    output[..., -1, -1, dc:] = dft[..., 0, 0, :]
    # fill redundant half-spectrum
    output[..., :, :, :dc] = torch.flip(
        torch.conj(output[..., :, :, dc + 1:]), dims=(-3, -2, -1)
    )
    return output


def _symmetrised_dft_to_dft_2d(dft: torch.Tensor, inplace: bool = True):
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


def _symmetrised_dft_to_dft_3d(dft: torch.Tensor, inplace: bool = True):
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


def rfft_to_dft_2d(
    dft: torch.Tensor,
    symmetrise: bool = False,
) -> torch.Tensor:
    dft = _rfft_to_symmetrised_dft_2d(dft)
    if symmetrise is False:
        dft = _symmetrised_dft_to_dft_2d(dft)
    return dft


def rfft_to_dft_3d(
    dft: torch.Tensor,
    symmetrise: bool = False,
) -> torch.Tensor:
    dft = _rfft_to_symmetrised_dft_3d(dft)
    if symmetrise is False:
        dft = _symmetrised_dft_to_dft_3d(dft)
    return dft


def dft_to_rfft_2d(
    dft: torch.Tensor,
    symmetrised: bool = False,
) -> torch.Tensor:
    if symmetrised is True:
        dft = _symmetrised_dft_to_dft_2d(dft)
    else:
        raise NotImplementedError()
    return dft


def dft_to_rfft_3d(
    dft: torch.Tensor,
    symmetrised: bool = False,
) -> torch.Tensor:
    if symmetrised is True:
        dft = _symmetrised_dft_to_dft_3d(dft)
    else:
        raise NotImplementedError()
    return dft


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
    dc_idx = dft_center(dft_shape, fftshifted=True, rfft=False)
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
    return einops.reduce(idx ** 2, '... c -> ...', reduction='sum') ** 0.5


def _target_fftfreq_from_spacing(
    source_spacing: Sequence[float],
    target_spacing: Sequence[float],
) -> Sequence[float]:
    target_fftfreq = [
        (_src / _target) * 0.5
        for _src, _target
        in zip(source_spacing, target_spacing)
    ]
    return target_fftfreq


def _best_fft_shape(
    image_shape: Sequence[int],
    target_fftfreq: Sequence[float],
    maximum_relative_error: float = 0.0005,
) -> tuple[int, int]:
    best_fft_shape = [
        best_fft_size(
            lower_bound=dim_length,
            target_fftfreq=_target_fftfreq,
            maximum_relative_error=maximum_relative_error
        )
        for dim_length, _target_fftfreq
        in zip(image_shape, target_fftfreq)
    ]
    return best_fft_shape


def _pad_to_best_fft_shape_2d(
    image: torch.Tensor,
    target_fftfreq: tuple[float, float]
):
    fft_size_h, fft_size_w = _best_fft_shape(
        image_shape=image.shape[-2:], target_fftfreq=target_fftfreq
    )
    # padding is not supported for arrays with large ndim, pack
    image, ps = einops.pack([image], pattern='* h w')

    # pad to best fft size
    h, w = image.shape[-2:]
    ph, pw = fft_size_h - h, fft_size_w - w
    too_much_padding = ph > h or pw > w
    if too_much_padding:
        image_means = einops.reduce(
            image, '... h w -> ... 1 1', reduction='mean'
        )
        image -= image_means
        image = F.pad(image, pad=(0, pw, 0, ph), mode='constant', value=0)
        image += image_means
    else:
        image = F.pad(image, pad=(0, pw, 0, ph), mode='reflect')
    [image] = einops.unpack(image, pattern='* h w', packed_shapes=ps)
    return image


def fftfreq_to_dft_coordinates(
    frequencies: torch.Tensor, image_shape: tuple[int, ...], rfft: bool
):
    """Convert DFT sample frequencies into array coordinates in a fftshifted DFT.

    Parameters
    ----------
    frequencies: torch.Tensor
        `(..., d)` array of multidimensional DFT sample frequencies
    image_shape: tuple[int, ...]
        Length `d` array of image dimensions.
    rfft: bool
        Whether output should be compatible with an rfft (`True`) or a
        full DFT (`False`)

    Returns
    -------
    coordinates: torch.Tensor
        `(..., d)` array of coordinates into a fftshifted DFT.
    """
    _image_shape = image_shape
    image_shape = torch.as_tensor(
        _image_shape, device=frequencies.device, dtype=frequencies.dtype
    )
    _rfft_shape = rfft_shape(_image_shape)
    _rfft_shape = torch.as_tensor(
        _rfft_shape, device=frequencies.device, dtype=frequencies.dtype
    )
    coordinates = torch.empty_like(frequencies)
    coordinates[..., :-1] = frequencies[..., :-1] * image_shape[:-1]
    if rfft is True:
        coordinates[..., -1] = frequencies[..., -1] * 2 * (_rfft_shape[-1] - 1)
    else:
        coordinates[..., -1] = frequencies[..., -1] * image_shape[-1]
    dc = dft_center(_image_shape, rfft=rfft, fftshifted=True, device=frequencies.device)
    return coordinates + dc
