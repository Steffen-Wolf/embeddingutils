import embeddingutils.visualizers.base
from embeddingutils.visualizers.dimspec import SpecFunction, convert_dim

import colorsys
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import torch
import numpy as np


def hsv_to_rgb(h, s, v):  # TODO: remove colorsys dependency
    return np.array(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)


def get_distinct_colors(n, min_sat=.5, min_val=.5):
    huePartition = 1.0 / (n + 1)
    hues = np.arange(0, n) * huePartition
    saturations = np.random.rand(n) * (1-min_sat) + min_sat
    values = np.random.rand(n) * (1-min_val) + min_val
    return np.stack([hsv_to_rgb(h, s, v) for h, s, v in zip(hues, saturations, values)], axis=0)


def colorize_segmentation(seg, ignore_label=None, ignore_color=(0, 0, 0)):
    assert isinstance(seg, np.ndarray)
    assert seg.dtype.kind in ('u', 'i')
    if ignore_label is not None:
        ignore_ind = seg == ignore_label
    seg = seg - np.min(seg)
    colors = get_distinct_colors(np.max(seg) + 1)
    np.random.shuffle(colors)
    result = colors[seg]
    if ignore_label is not None:
        result[ignore_ind] = ignore_color
    return result


def _from_matplotlib_cmap(cmap):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    return scalarMap.to_rgba


class Colorize(SpecFunction):
    def __init__(self, cmap=None):
        super(Colorize, self).__init__(in_specs={'tensor': ['B', 'D', 'H', 'W', 'Color']},
                                       out_spec=['B', 'D', 'H', 'W', 'Color'])
        self.cmap = _from_matplotlib_cmap(cmap) if isinstance(cmap, str) else cmap

    def internal(self, tensor):
        # TODO: proper scaling, alpha

        tensor -= torch.min(tensor)
        tensor /= torch.max(tensor)

        # first, add color if none is there
        if tensor.shape[-1] == 1:  # no color yet
            if self.cmap is not None:  # apply colormap
                dtype = tensor.dtype
                tensor = self.cmap(tensor.numpy()[..., 0])[..., :3]
                tensor = torch.tensor(tensor, dtype=dtype)
            else:
                tensor = tensor.repeat(1, 1, 1, 1, 3)
        elif tensor.shape[-1] == 3:
            pass
        else:
            assert False, f'{tensor.shape}'

        # TODO: second, check if alpha channel is present and add it if it is not
        pass

        # TODO: lastly, scale the color channels to lie in [0, 1]
        pass

        return tensor


if __name__ == '__main__':
    tensor = torch.Tensor([0, 1, 2, 3, 4])
    colorize = Colorize(cmap='inferno')
    out, spec = colorize(tensor=(tensor, 'W'), out_spec=['W', 'Color'], return_spec=True)
    print(out)
    print(out.shape)
    print(spec)

