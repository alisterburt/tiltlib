import einops
import torch
import torch.nn.functional as F

from libtilt.coordinate_utils import (
    array_to_grid_sample,
)
from libtilt.grids.coordinate_grid import coordinate_grid


def project_volume_real(volume: torch.Tensor, rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Make 2D projections of a 3D volume in specific orientations.

    Projections are made by

    1. generating a grid of coordinates sufficient to cover
       the volume in any orientation.

    2. left-multiplying `rotation matrices` and coordinate grids to
       produce rotated coordinates.

    3. sampling `volume` at rotated coordinates.

    4. summing samples along depth dimension of a `(d, h, w)` volume.

    The rotation center of `volume` is taken to be `torch.tensor(volume.shape) // 2`.

    Parameters
    ----------
    volume: torch.Tensor
        `(d, h, w)` volume from which projections will be made.
    rotation_matrices: torch.Tensor
        `(batch, 3, 3)` array of rotation matrices

    Returns
    -------
    projection_images: torch.Tensor
        `(batch, h, w)` array of 2D projection images sampled from `volume`.
    """
    volume = torch.as_tensor(volume)
    rotation_matrices = torch.as_tensor(rotation_matrices, dtype=torch.float)
    volume_shape = torch.tensor(volume.shape)
    ps = padded_sidelength = int(3 ** 0.5 * torch.max(volume_shape))
    shape_difference = torch.abs(padded_sidelength - volume_shape)
    padding = torch.empty(size=(3, 2), dtype=torch.int16)
    padding[:, 0] = torch.div(shape_difference, 2, rounding_mode='floor')
    padding[:, 1] = shape_difference - padding[:, 0]
    torch_padding = torch.flip(padding, dims=(0,))  # dhw -> whd for torch.nn.functional.pad
    torch_padding = einops.rearrange(torch_padding, 'whd pad -> (whd pad)')
    volume = F.pad(volume, pad=tuple(torch_padding), mode='constant', value=0)
    padded_volume_shape = (ps, ps, ps)
    volume_coordinates = coordinate_grid(image_shape=padded_volume_shape, device=volume.device)
    volume_coordinates -= padded_sidelength // 2  # (d, h, w, zyx)
    volume_coordinates = torch.flip(volume_coordinates, dims=(-1,))  # (d, h, w, zyx)
    volume_coordinates = einops.rearrange(volume_coordinates, 'd h w zyx -> d h w zyx 1')

    def _project_volume(rotation_matrix) -> torch.Tensor:
        rotated_coordinates = rotation_matrix @ volume_coordinates
        rotated_coordinates += padded_sidelength // 2
        rotated_coordinates = einops.rearrange(rotated_coordinates, 'd h w zyx 1 -> 1 d h w zyx')
        rotated_coordinates = torch.flip(rotated_coordinates, dims=(-1,))  # zyx -> xyz
        rotated_coordinates = array_to_grid_sample(
            rotated_coordinates, array_shape=padded_volume_shape
        )
        samples = F.grid_sample(
            input=einops.rearrange(volume, 'd h w -> 1 1 d h w'),  # add batch and channel dims
            grid=rotated_coordinates,
            mode='bilinear',  # trilinear for volume
            padding_mode='zeros',
            align_corners=True,
        )
        return einops.reduce(samples, '1 1 d h w -> h w', reduction='sum')

    yl, yh = padding[1, 0], -padding[1, 1]
    xl, xh = padding[2, 0], -padding[2, 1]
    images = [_project_volume(matrix)[yl:yh, xl:xh] for matrix in rotation_matrices] #TODO: This can probabaly optimized using vmap
    return torch.stack(images, dim=0)


def project_image_real(image: torch.Tensor, rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Make 1D projections of a 2D image in specific orientations.

    Projections are made by

    1. generating a line of coordinates sufficient to cover
       the image in any orientation.

    2. left-multiplying `rotation matrices` and coordinate grids to
       produce rotated coordinates.

    3. sampling `image` at rotated coordinates.

    4. summing samples along 'h' dimension of a `(h, w)` image.

    The rotation center of `image` is taken to be `torch.tensor(image.shape) // 2`.

    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` image from which projections will be made.
    rotation_matrices: torch.Tensor
        `(batch, 2, 2)` array of rotation matrices

    Returns
    -------
    projection_lines: torch.Tensor
        `(batch, w)` array of 1D projection lines sampled from `image`.
    """
    image = torch.as_tensor(image)
    rotation_matrices = torch.as_tensor(rotation_matrices, dtype=torch.float)
    image_shape = torch.tensor(image.shape)
    ps = padded_sidelength = int(3 ** 0.5 * torch.max(image_shape))
    shape_difference = torch.abs(padded_sidelength - image_shape)
    padding = torch.empty(size=(2, 2), dtype=torch.int16)
    padding[:, 0] = torch.div(shape_difference, 2, rounding_mode='floor')
    padding[:, 1] = shape_difference - padding[:, 0]
    torch_padding = torch.flip(padding, dims=(0,))  # hw -> wh for torch.nn.functional.pad
    torch_padding = einops.rearrange(torch_padding, 'wh pad -> (wh pad)')
    image = F.pad(image, pad=tuple(torch_padding), mode='constant', value=0)
    padded_image_shape = (ps, ps)
    image_coordinates = coordinate_grid(image_shape=padded_image_shape)
    image_coordinates -= padded_sidelength // 2  # (h, w, yx)
    image_coordinates = torch.flip(image_coordinates, dims=(-1,))  # (h, w, yx)
    image_coordinates = einops.rearrange(image_coordinates, 'h w yx -> h w yx 1')

    def _project_image(rotation_matrix) -> torch.Tensor:
        rotated_coordinates = rotation_matrix @ image_coordinates
        rotated_coordinates += padded_sidelength // 2
        rotated_coordinates = einops.rearrange(rotated_coordinates, 'h w yx 1 -> 1 h w yx')
        rotated_coordinates = torch.flip(rotated_coordinates, dims=(-1,))  # yx -> xy
        rotated_coordinates = array_to_grid_sample(
            rotated_coordinates, array_shape=padded_image_shape
        )
        samples = F.grid_sample(
            input=einops.rearrange(image, 'h w -> 1 1 h w'),  # add batch and channel dims
            grid=rotated_coordinates,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        return einops.reduce(samples, '1 1 h w -> w', reduction='sum')

    xl, xh = padding[1, 0], -padding[1, 1]
    lines = [_project_image(matrix)[xl:xh] for matrix in rotation_matrices]
    return torch.stack(lines, dim=0)
