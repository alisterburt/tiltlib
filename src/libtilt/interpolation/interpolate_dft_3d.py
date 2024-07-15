from typing import Tuple, Literal

import einops
import torch
from torch.nn import functional as F

from libtilt.coordinate_utils import array_to_grid_sample


def sample_dft_3d(
    dft: torch.Tensor,
    coordinates: torch.Tensor
) -> torch.Tensor:
    """Sample a complex volume with linear interpolation.


    Parameters
    ----------
    dft: torch.Tensor
        `(d, h, w)` complex valued volume.
    coordinates: torch.Tensor
        `(..., zyx)` array of coordinates at which `dft` should be sampled.
        Coordinates should be ordered zyx, aligned with image dimensions `(d, h, w)`.
        Coordinates should be array coordinates, spanning `[0, N-1]` for a
        dimension of length N.
    Returns
    -------
    samples: torch.Tensor
        `(..., )` array of complex valued samples from `dft`.
    """
    coordinates, ps = einops.pack([coordinates], pattern='* zyx')
    n_samples = coordinates.shape[0]

    # cannot sample complex tensors directly with grid_sample
    # c.f. https://github.com/pytorch/pytorch/issues/67634
    # workaround: treat real and imaginary parts as separate channels
    dft = einops.rearrange(torch.view_as_real(dft), 'd h w complex -> complex d h w')
    dft = einops.repeat(dft, 'complex d h w -> b complex d h w', b=n_samples)
    coordinates = einops.rearrange(coordinates, 'b zyx -> b 1 1 1 zyx')  # b d h w zyx

    samples = F.grid_sample(
        input=dft,
        grid=array_to_grid_sample(coordinates, array_shape=dft.shape[-3:]),
        mode='bilinear',  # this is trilinear when input is volumetric
        padding_mode='border',  # this increases sampling fidelity at nyquist
        align_corners=True,
    )
    samples = einops.rearrange(samples, 'b complex 1 1 1 -> b complex')
    samples = torch.view_as_complex(samples.contiguous())  # (b, )

    # pack data back up and return
    # [samples] = einops.unpack(samples, pattern='*', packed_shapes=ps)
    samples = samples.reshape(*ps)  # replaces commented line above, for performance
    return samples  # (...)


def insert_into_dft_3d(
    data: torch.Tensor,
    coordinates: torch.Tensor,
    dft: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Insert values into a 3D DFT with trilinear interpolation (rasterisation).

    Parameters
    ----------
    data: torch.Tensor
        `(...)` array of values to be inserted into the DFT.
    coordinates: torch.Tensor
        `(..., 3)` array of 3D coordinates for each value in `data`.
    dft: torch.Tensor
        `(d, d, d)` volume containing the discrete Fourier transform into which data will be inserted.
    weights: torch.Tensor
        `(d, d, d)` volume containing the weights associated with each voxel of `dft`

    Returns
    -------
    dft, weights: Tuple[torch.Tensor, torch.Tensor]
        The dft and weights after updating with data from `slices` at `slice_coordinates`.
    """
    if data.shape != coordinates.shape[:-1]:
        raise ValueError('One coordinate triplet is required for each value in data.')

    # linearise data and coordinates
    data, _ = einops.pack([data], pattern='*')
    coordinates, _ = einops.pack([coordinates], pattern='* zyx')
    coordinates = coordinates.float()

    # only keep data and coordinates inside the volume
    in_volume_idx = (coordinates >= 0) & (coordinates <= torch.tensor(dft.shape) - 1)
    in_volume_idx = torch.all(in_volume_idx, dim=-1)
    data, coordinates = data[in_volume_idx], coordinates[in_volume_idx]

    # pre-compute corner coordinates around each sampling point
    corners = torch.empty(size=(data.shape[0], 2, 3), dtype=torch.long)
    corners[:, 0] = torch.floor(coordinates)  # for lower corners
    corners[:, 1] = torch.ceil(coordinates)  # for upper corners

    # pre-compute linear interpolation weights for each value being inserted
    _weights = torch.empty(size=(data.shape[0], 2, 3))  # (b, 2, zyx)
    _weights[:, 1] = coordinates - corners[:, 0]  # upper corner weights
    _weights[:, 0] = 1 - _weights[:, 1]  # lower corner weights

    def add_data_at_corner(z: Literal[0, 1], y: Literal[0, 1], x: Literal[0, 1]):
        w = _weights[:, [z, y, x], [0, 1, 2]]
        w = einops.reduce(w, 'b zyx -> b', reduction='prod')
        zc, yc, xc = einops.rearrange(corners[:, [z, y, x], [0, 1, 2]],
                                      'b zyx -> zyx b')
        dft.index_put_(indices=(zc, yc, xc), values=w * data, accumulate=True)
        weights.index_put_(indices=(zc, yc, xc), values=w, accumulate=True)

    add_data_at_corner(0, 0, 0)
    add_data_at_corner(0, 0, 1)
    add_data_at_corner(0, 1, 0)
    add_data_at_corner(0, 1, 1)
    add_data_at_corner(1, 0, 0)
    add_data_at_corner(1, 0, 1)
    add_data_at_corner(1, 1, 0)
    add_data_at_corner(1, 1, 1)

    return dft, weights
