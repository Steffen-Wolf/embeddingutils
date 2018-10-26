from embeddingutils.visualizers.base import BaseVisualizer, ContainerVisualizer
from embeddingutils.visualization import pca
from embeddingutils.visualizers.dimspec import convert_dim
import torch
import torch.nn.functional as F


def _to_img_grid(tensor, spec, return_spec=False):
    collapsing_rules = [('D', 'B'), ('T', 'B')]
    out_spec = ['B', 'C', 'H', 'W', 'Color']
    tensor, spec = convert_dim(tensor, in_spec=spec, out_spec=out_spec, collapsing_rules=collapsing_rules,
                               return_spec=True)
    if return_spec:
        return tensor, spec
    else:
        return tensor


def _padded_concatenate(tensors, dim, pad_width, pad_value):
    tensors = list(tensors)
    if pad_width != 0:
        pad_shape = list(tensors[0].shape)
        pad_shape[dim] = pad_width
        pad_tensor = torch.ones(pad_shape).to(tensors[0].device) * pad_value
        [tensors.insert(i, pad_tensor) for i in range(len(tensors)-1, 0, -1)]
    return torch.cat(tensors, dim=dim)


class RowVisualizer(ContainerVisualizer):
    def __init__(self, pad_width=1, pad_value=.2, upsampling_factor=1, *super_args, **super_kwargs):
        super(RowVisualizer, self).__init__(in_spec=['B', 'C', 'Color', 'T', 'D', 'H', 'W'],
                                            out_spec=['H', 'W', 'Color'],
                                            *super_args, **super_kwargs)
        self.pad_width = pad_width
        self.pad_value = pad_value
        assert isinstance(upsampling_factor, int)
        self.upsampling_factor = upsampling_factor


    def combine(self, *image_grids, **_):
        assert all(all(grid.shape[i] == image_grids[0].shape[i] for grid in image_grids[1:]) for i in [0, 2, 4]), \
            f'Batch, Color and Width dimension must have same length, {[grid.shape for grid in image_grids]}'

        image_grids = [_to_img_grid(grid, self.in_spec) for grid in image_grids]
        # shape now ['B', 'C', 'H', 'W', 'Color']
        image_grids = [_padded_concatenate(grid, dim=2, pad_width=self.pad_width, pad_value=self.pad_value)
                       for grid in image_grids]
        # shape now ['C', 'H', 'W', 'Color']
        image_grids = [_padded_concatenate(grid, dim=0, pad_width=self.pad_width, pad_value=self.pad_value)
                       for grid in image_grids]
        # shape now [B, C, Color] == [H, W, Color] as images
        result = _padded_concatenate(image_grids, dim=0, pad_width=self.pad_width, pad_value=self.pad_value)

        if self.upsampling_factor is not 1:
            result = F.upsample(
                result.permute(2, 0, 1)[None],
                scale_factor=self.upsampling_factor,
                mode='nearest')
            result = result[0].permute(1, 2, 0)

        return result


class ColumnVisualizer(RowVisualizer):
    def combine(self, *image_grids, **_):
        assert all(all(grid.shape[i] == image_grids[0].shape[i] for grid in image_grids[1:]) for i in [0, 2, 4]), \
            f'Batch, Color and Width dimension must have same length, {[grid.shape for grid in image_grids]}'

        image_grids = [_to_img_grid(grid, self.in_spec) for grid in image_grids]
        # shape now ['B', 'C', 'H', 'W', 'Color']
        image_grids = [_padded_concatenate(grid, dim=1, pad_width=self.pad_width, pad_value=self.pad_value)
                       for grid in image_grids]
        # shape now ['C', 'H', 'W', 'Color']
        image_grids = [_padded_concatenate(grid, dim=1, pad_width=self.pad_width, pad_value=self.pad_value)
                       for grid in image_grids]
        # shape now ['H', 'W', 'Color']
        result = _padded_concatenate(image_grids, dim=1, pad_width=self.pad_width, pad_value=self.pad_value)

        if self.upsampling_factor is not 1:
            result = F.upsample(
                result.permute(2, 0, 1)[None],
                scale_factor=self.upsampling_factor,
                mode='nearest')
            result = result[0].permute(1, 2, 0)

        return result


class OverlayVisualizer(ContainerVisualizer):
    def __init__(self, *super_args, **super_kwargs):
        super(OverlayVisualizer, self).__init__(
            in_spec=['Color', 'B'],
            out_spec=['Color', 'B'],
            *super_args, **super_kwargs
        )

    def combine(self, *visualizations, **_):
        result = visualizations[-1]
        for overlay in reversed(visualizations[:-1]):
            a = (overlay[3] + result[3] * (1 - overlay[3]))[None]
            rgb = overlay[:3] * overlay[3][None] + result[:3] * result[3][None] * (1 - overlay[3][None])
            rgb /= a
            result = torch.cat([rgb, a], dim=0)
        return result