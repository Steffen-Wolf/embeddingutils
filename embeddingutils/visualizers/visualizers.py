from embeddingutils.visualizers.base import BaseVisualizer, ContainerVisualizer
from embeddingutils.visualization import pca
from embeddingutils.visualizers.dimspec import convert_dim
import torch
import torch.nn.functional as F


class InputVisualizer(BaseVisualizer):
    def __init__(self, **super_kwargs):
        super(InputVisualizer, self).__init__(
            in_specs={'input': 'B'},
            out_spec='B',
            **super_kwargs
        )

    def visualize(self, input, **_):
        return input


class TargetVisualizer(BaseVisualizer):
    def __init__(self, **super_kwargs):
        super(TargetVisualizer, self).__init__(
            in_specs={'target': 'B'},
            out_spec='B',
            **super_kwargs
        )

    def visualize(self, target, **_):
        return target


class PredictionVisualizer(BaseVisualizer):
    def __init__(self, **super_kwargs):
        super(PredictionVisualizer, self).__init__(
            in_specs={'prediction': 'B'},
            out_spec='B',
            **super_kwargs
        )

    def visualize(self, prediction, **_):
        return prediction


class SegmentationVisualizer(BaseVisualizer):
    def __init__(self, **super_kwargs):
        super(SegmentationVisualizer, self).__init__(
            in_specs={'segmentation': 'B'},
            out_spec='B',
            **super_kwargs
        )

    def visualize(self, segmentation, **_):
        return segmentation


class PcaVisualizer(BaseVisualizer):
    def __init__(self, **super_kwargs):
        super(PcaVisualizer, self).__init__(in_specs={'embedding': ['B', 'C', 'D', 'H', 'W']},
                                            out_spec=['B', 'Color', 'D', 'H', 'W'],
                                            **super_kwargs)

    def visualize(self, embedding, **_):
        return pca(embedding, output_dimensions=3)


class MaskedPcaVisualizer(BaseVisualizer):
    def __init__(self, ignore_label=None, n_components=3, background_brightness=.8, **super_kwargs):
        super(MaskedPcaVisualizer, self).__init__(
            in_specs={'embedding': 'BCDHW', 'segmentation': 'BCDHW'},
            out_spec=['B', 'C', 'Color', 'D', 'H', 'W'],
            **super_kwargs)
        self.ignore_label = ignore_label
        self.background_brightness = background_brightness
        assert n_components % 3 == 0, f'{n_components} is not divisible by 3.'
        self.n_images = n_components // 3

    def visualize(self, embedding, segmentation, **_):
        if self.ignore_label is None:
            mask = torch.ones((embedding.shape[0],) + embedding.shape[2:])
        else:
            mask = segmentation != self.ignore_label
        if len(mask.shape) == len(embedding.shape):
            assert mask.shape[1] == 1, f'{mask.shape}'
            mask = mask[:, 0]
        mask = mask.byte()
        masked = [embedding[i, :, m] for i, m in enumerate(mask)]
        masked = [pca(d[None], 3 * self.n_images, center_data=True)[0] for d in masked]
        output_shape = list(embedding.shape)
        output_shape[1] = 3 * self.n_images
        result = torch.ones(output_shape) * self.background_brightness
        for i, m in enumerate(mask):
            result[i, :, m] = masked[i]
        result = result.contiguous().view((result.shape[0], self.n_images, 3) + result.shape[2:])
        return result


class GridVisualizer(ContainerVisualizer):
    def __init__(self, pad_width=1, pad_value=.2, upsampling_factor=1, *super_args, **super_kwargs):
        super(GridVisualizer, self).__init__(*super_args, **super_kwargs)
        self.pad_width = pad_width
        self.pad_value = pad_value
        assert isinstance(upsampling_factor, int)
        self.upsampling_factor = upsampling_factor

    def padded_concatenate(self, tensors, dim):
        tensors = list(tensors)
        if self.pad_width != 0:
            pad_shape = list(tensors[0].shape)
            pad_shape[dim] = self.pad_width
            pad_tensor = torch.ones(pad_shape).to(tensors[0].device) * self.pad_value
            [tensors.insert(i, pad_tensor) for i in range(len(tensors)-1, 0, -1)]
        return torch.cat(tensors, dim=dim)

    def combine(self, *image_grids, **_):
        assert all(all(grid.shape[i] == image_grids[0].shape[i] for grid in image_grids[1:]) for i in [0, 2, 4]), \
            f'Batch, Color and Width dimension must have same length, {[grid.shape for grid in image_grids]}'

        spec = ['B', 'C', 'Color', 'H', 'W']
        new_spec = ['B', 'C', 'H', 'W', 'Color']
        image_grids = [convert_dim(grid, spec, new_spec) for grid in image_grids]
        image_grids = [self.padded_concatenate(grid, dim=2) for grid in image_grids]
        # now ['C', 'H', 'W', 'Color']
        image_grids = [self.padded_concatenate(grid, dim=0) for grid in image_grids]

        #images = [convert_dim(grid, in_spec=spec, out_spec=['C', 'B', 'Color'], collapsing_rules=collapsing_rules)
        #          for grid in image_grids]

        # shape now [B, C, Color] == [H, W, Color] as images
        result = self.padded_concatenate(image_grids, dim=0)

        if self.upsampling_factor is not 1:
            result = F.upsample(
                result.permute(2, 0, 1)[None],
                scale_factor=self.upsampling_factor,
                mode='nearest')
            result = result[0].permute(1, 2, 0)

        return result


if __name__ == '__main__':
    v0 = PcaVisualizer(input_mapping=None)
    v = GridVisualizer((v0, v0))
    import numpy as np
    tensor = torch.Tensor(np.random.randn(10, 2, 5, 5))
    result = v(embedding=(tensor, ['C', 'D', 'H', 'W']))
    print(result[..., 0])
    print(result.shape)
