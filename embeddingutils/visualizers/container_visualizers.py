from embeddingutils.visualizers.base import ContainerVisualizer
from embeddingutils.visualizers.dimspec import convert_dim
import torch
import torch.nn.functional as F


def _to_rgba(color):
    if isinstance(color, (int, float)):  # color given as brightness
        result = [color, color, color, 1]
    elif isinstance(color, list):
        if len(color) == 3:  # color given as RGB
            result = color + [1]
        elif len(color) == 4:
            result = color.copy()
        else:
            assert False, f'len({color}) = {len(color)} has to be in [3, 4]'
    else:
        assert False, f'color specification not understood: {color}'
    return result


def _padded_concatenate(tensors, dim, pad_width, pad_value):
    tensors = list(tensors)
    device = tensors[0].device
    if pad_width != 0:
        pad_shape = list(tensors[0].shape)
        pad_shape[dim] = pad_width
        if isinstance(pad_value, list):
            pad_value = torch.Tensor(pad_value).to(device).type_as(tensors[0])
        pad_tensor = torch.ones(pad_shape).to(device) * pad_value
        [tensors.insert(i, pad_tensor) for i in range(len(tensors)-1, 0, -1)]
    return torch.cat(tensors, dim=dim)


class ImageGridVisualizer(ContainerVisualizer):
    def __init__(self, row_specs=('H', 'C', 'V'), column_specs=('W', 'D', 'T', 'B'),
                 pad_width=1, pad_value=.5, upsampling_factor=1, *super_args, **super_kwargs):
        super(ImageGridVisualizer, self).__init__(
            in_spec=None, out_spec=None,
            suppress_spec_adjustment=True,
            equalize_visualization_shapes=False,
            *super_args, **super_kwargs)
        assert all([d not in column_specs for d in row_specs]), 'every spec has to go either in rows or colums'

        # determine if the individual visualizers should be stacked as rows or columns
        if 'V' in row_specs:
            assert row_specs[-1] == 'V'
            row_specs = row_specs[:-1]
            self.visualizer_stacking = 'rows'
        elif 'V' in column_specs:
            assert column_specs[-1] == 'V'
            column_specs = column_specs[:-1]
            self.visualizer_stacking = 'columns'
        else:
            self.visualizer_stacking = 'rows'

        self.n_row_dims = len(row_specs)
        self.n_col_dims = len(column_specs)
        self.initial_spec = list(row_specs) + list(column_specs) + ['out_height', 'out_width', 'Color']

        self.pad_value = pad_value
        self.pad_width = pad_width

        self.upsampling_factor = upsampling_factor

    def get_pad_kwargs(self, spec):
        # helper function to manage padding widths and values
        result = dict()
        hw = ('H', 'W')
        if isinstance(self.pad_width, dict):
            result['pad_width'] = self.pad_width.get(spec, self.pad_width.get('rest', 0))
        else:
            result['pad_width'] = self.pad_width if spec not in hw else 0

        if isinstance(self.pad_value, dict):
            result['pad_value'] = self.pad_value.get(spec, self.pad_value.get('rest', .5))
        else:
            result['pad_value'] = self.pad_value if spec not in hw else 0
        result['pad_value'] = _to_rgba(result['pad_value'])

        return result

    def visualization_to_image(self, visualization, spec):
        # this function should not be overridden for regular container visualizers, but is here, as the specs have to be
        # known in the main visualization function. 'combine()' is never called, internal is used directly

        collapsing_rules = [(d, 'B') for d in spec if d not in self.initial_spec]  # everything unknown goes into batch
        visualization, spec = convert_dim(visualization, in_spec=spec, out_spec=self.initial_spec,
                                          collapsing_rules=collapsing_rules, return_spec=True)

        # collapse the rows in the 'out_width' dimension, it is at position -2
        for _ in range(self.n_row_dims):
            visualization = _padded_concatenate(visualization, dim=-3, **self.get_pad_kwargs(spec[0]))
            spec = spec[1:]

        # collapse the columns in the 'out_height' dimension, it is at position -3
        for _ in range(self.n_col_dims):
            visualization = _padded_concatenate(visualization, dim=-2, **self.get_pad_kwargs(spec[0]))
            spec = spec[1:]
        return visualization

    def internal(self, *args, return_spec=False, **states):
        images = []
        for name in self.visualizer_kwarg_names:
            images.append(self.visualization_to_image(*states[name]))

        if self.visualizer_stacking == 'rows':
            result = _padded_concatenate(images, dim=-3, **self.get_pad_kwargs('V'))
        else:
            result = _padded_concatenate(images, dim=-2, **self.get_pad_kwargs('V'))

        if self.upsampling_factor is not 1:
            result = F.upsample(
                result.permute(2, 0, 1)[None],
                scale_factor=self.upsampling_factor,
                mode='nearest')
            result = result[0].permute(1, 2, 0)

        if return_spec:
            return result, ['H', 'W', 'Color']
        else:
            return result


class RowVisualizer(ImageGridVisualizer):
    def __init__(self, *super_args, **super_kwargs):
        super(RowVisualizer, self).__init__(
            row_specs=('H', 'S', 'C', 'V'),
            column_specs=('W', 'D', 'T', 'B'),
            *super_args, **super_kwargs)


class ColumnVisualizer(ImageGridVisualizer):
    def __init__(self, *super_args, **super_kwargs):
        super(ColumnVisualizer, self).__init__(
            row_specs=('H', 'D', 'T', 'B'),
            column_specs=('W', 'S', 'C', 'V'),
            *super_args, **super_kwargs)


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


class RiffleVisualizer(ContainerVisualizer):
    def __init__(self, riffle_dim='C', *super_args, **super_kwargs):
        super(RiffleVisualizer, self).__init__(
            in_spec=[riffle_dim, 'B'],
            out_spec=[riffle_dim, 'B'],
            *super_args, **super_kwargs
        )

    def combine(self, *visualizations, **_):
        assert len(visualizations) > 0
        assert all(v.shape == visualizations[0].shape for v in visualizations[1:]), \
            f'Not all input visualizations have the same shape: {[v.shape for v in visualizations]}'
        result = torch.stack(visualizations, dim=1)
        result = result.contiguous().view(-1, visualizations[0].shape[1])
        return result


class StackVisualizer(ContainerVisualizer):
    def __init__(self, stack_dim='S', *super_args, **super_kwargs):
        super(StackVisualizer, self).__init__(
            in_spec=['B'],
            out_spec=[stack_dim, 'B'],
            *super_args, **super_kwargs
        )

    def combine(self, *visualizations, **_):
        assert len(visualizations) > 0
        assert all(v.shape == visualizations[0].shape for v in visualizations[1:]), \
            f'Not all input visualizations have the same shape: {[v.shape for v in visualizations]}'
        result = torch.stack(visualizations, dim=0)
        return result
