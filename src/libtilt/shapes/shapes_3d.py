import einops
import torch

from libtilt.grids import coordinate_grid
from libtilt.shapes.soft_edge import add_soft_edge_3d
from libtilt.utils.fft import dft_center


def sphere(
    radius: float,
    image_shape: tuple[int, int, int] | int,
    center: tuple[float, float, float] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape, image_shape)
    if center is None:
        center = dft_center(image_shape, rfft=False, fftshifted=True)
    distances = coordinate_grid(
        image_shape=image_shape,
        center=center,
        norm=True,
        device=device,
    )
    mask = torch.zeros_like(distances, dtype=torch.bool)
    mask[distances < radius] = 1
    return add_soft_edge_3d(mask, smoothing_radius=smoothing_radius)


def cuboid(
    dimensions: tuple[float, float, float],
    image_shape: tuple[int, int, int] | int,
    center: tuple[float, float] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape, image_shape)
    if center is None:
        center = dft_center(image_shape, rfft=False, fftshifted=True)
    coordinates = coordinate_grid(
        image_shape=image_shape,
        center=center,
        device=device,
    )
    dd, dh, dw = dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2
    depth_mask = torch.logical_and(coordinates[..., 0] > -dh, coordinates[..., 0] < dh)
    height_mask = torch.logical_and(coordinates[..., 1] > -dw, coordinates[..., 1] < dw)
    width_mask = torch.logical_and(coordinates[..., 2] > -dw, coordinates[..., 2] < dw)
    mask = einops.rearrange([depth_mask, height_mask, width_mask], 'dhw d h w -> d h w dhw')
    mask = torch.all(mask, dim=-1)
    return add_soft_edge_3d(mask, smoothing_radius=smoothing_radius)