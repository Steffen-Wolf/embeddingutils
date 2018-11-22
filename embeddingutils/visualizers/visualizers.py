from embeddingutils.visualizers.base import BaseVisualizer, ContainerVisualizer
from embeddingutils.visualization import pca
from embeddingutils.visualizers.dimspec import convert_dim
import torch
import torch.nn.functional as F


class ImageVisualizer(BaseVisualizer):
    def __init__(self, **super_kwargs):
        super(ImageVisualizer, self).__init__(
            in_specs={'image': 'B'},
            out_spec='B',
            **super_kwargs
        )

    def visualize(self, image, **_):
        return image


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


class SigmoidVisualizer(BaseVisualizer):
    def __init__(self, **super_kwargs):
        super(SigmoidVisualizer, self).__init__(
            in_specs={'image': ['B']},
            out_spec=['B'],
            **super_kwargs)

    def visualize(self, image, **_):
        return F.sigmoid(image)


class RGBVisualizer(BaseVisualizer):
    def __init__(self, **super_kwargs):
        super(RGBVisualizer, self).__init__(
            in_specs={'image': ['B', 'C']},
            out_spec=['B', 'C', 'Color'],
            **super_kwargs
        )

    def visualize(self, image, **_):
        n_channels = image.shape[1]
        assert n_channels % 3 == 0, f'the number of channels {image.shape[1]} has to be divisible by 3'
        image = image.contiguous().view(image.shape[0], n_channels // 3, 3)
        return image


class MaskVisualizer(BaseVisualizer):
    def __init__(self, mask_label, **super_kwargs):
        super(MaskVisualizer, self).__init__(
            in_specs={'image': ['B']},
            out_spec=['B'],
            **super_kwargs
        )
        self.mask_label = mask_label

    def visualize(self, image, **states):
        return (image == self.mask_label).float()


class PcaVisualizer(BaseVisualizer):
    def __init__(self, **super_kwargs):
        super(PcaVisualizer, self).__init__(
            in_specs={'embedding': ['B', 'C', 'D', 'H', 'W']},
            out_spec=['B', 'Color', 'D', 'H', 'W'],
            **super_kwargs)

    def visualize(self, embedding, **_):
        return pca(embedding, output_dimensions=3)


class MaskedPcaVisualizer(BaseVisualizer):
    def __init__(self, ignore_label=None, n_components=3, background_label=0, **super_kwargs):
        super(MaskedPcaVisualizer, self).__init__(
            in_specs={'embedding': 'BCDHW', 'segmentation': 'BCDHW'},
            out_spec=['B', 'C', 'Color', 'D', 'H', 'W'],
            background_label=background_label,
            **super_kwargs)
        self.ignore_label = ignore_label
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
        masked = [None if d.nelement() == 0 else pca(d[None], 3 * self.n_images, center_data=True)[0]
                  for d in masked]
        output_shape = list(embedding.shape)
        output_shape[1] = 3 * self.n_images
        result = torch.zeros(output_shape)
        for i, m in enumerate(mask):
            if masked[i] is not None:
                result[i, :, m] = masked[i]
        result = result.contiguous().view((result.shape[0], self.n_images, 3) + result.shape[2:])
        return result


class NormVisualizer(BaseVisualizer):
    def __init__(self, order=2, dim='C'):
        super(NormVisualizer, self).__init__(
            in_specs={'tensor': ['B'] + [dim]},
            out_spec='B'
        )
        self.order = order

    def visualize(self, tensor):
        return tensor.norm(p=self.order, dim=1)
