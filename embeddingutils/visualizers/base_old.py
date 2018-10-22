import numpy as np
import torch
from inferno.trainers.callbacks.base import Callback
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from embeddingutils.visualization import pca


def _colorize(img, cmap=None):
    def from_matplotlib_cmap(cmap):
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        cNorm = colors.Normalize(vmin=0, vmax=1)
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
        return scalarMap.to_rgba

    if cmap is not None:  # apply colormap
        cmap = from_matplotlib_cmap(cmap) if isinstance(cmap, str) else cmap
        dtype = img.dtype
        img = cmap(img.numpy())
        img = np.moveaxis(img, -1, 1)
        img = torch.tensor(img, dtype=dtype)
    return img


def auto_colorization(func):
    def inner(*args, cmap='inferno', suppress_colorization=False, **kwargs):
        result = func(*args, **kwargs)
        if suppress_colorization:  # if no colorization is wanted, return result directly
            return result

        return _colorize(img=result, cmap=cmap)
    return inner


class BaseVisualizer:

    @auto_colorization
    def visualize_pca(self, embedding):
        pass

    @auto_colorization
    def v_gt_seg(self, segmentation, **_):
        return segmentation

    def v_pca(self, embedding, mask=None, background_brightness=.8, **_):
        # embedding shape: (B, C, (D), H, W)
        # mask shape: (B, (1), (D), H, W)
        if mask is None:
            mask = torch.ones((embedding.shape[0],) + embedding.shape[2:])
        if len(mask.shape) == len(embedding.shape):
            assert mask.shape[1] == 1, f'{mask.shape}'
            mask = mask[:, 0]
        mask = mask.byte()
        masked = [embedding[i, :, m] for i, m in enumerate(mask)]
        masked = [pca(d[None], 3, center_data=True)[0] for d in masked]
        output_shape = list(embedding.shape)
        output_shape[1] = 3
        result = torch.ones(output_shape) * background_brightness
        for i, m in enumerate(mask):
            result[i, :, m] = masked[i]
        return result

    def overlay_visualization(self, v1, v2):
        pass

    def visualize(self, **states):
        pass


class VisualizationCallback(Callback):
    def __init__(self, name, visualizer):
        self.name = name
        self.visualizer = visualizer

    @property
    def logger(self):
        assert self.trainer is not None
        assert hasattr(self.trainer, 'logger')
        return self.trainer.logger

    def get_trainer_states(self):
        states = ['inputs', 'error', 'target', 'prediction', 'loss']
        # TODO: better way to determine in what phase trainer is?
        pre = 'training' if self.trainer.model.training else 'validation'
        result = {}
        for s in states:
            state = self.trainer.get_state(pre + '_' + s)
            if isinstance(state, torch.Tensor):
                state = state.detach()
            result[s] = state
        return result

    def do_logging(self, **_):
        image = self.visualizer.visualize(**self.get_trainer_states())
        self.logger.log_object(self.name, image)

    def end_of_training_iteration(self, **_):
        if not self.logger.log_images_now:
            self.do_logging()

    def end_of_validation_run(self, **_):
        self.do_logging()


if __name__ == '__main__':

    class TestVisualizer(BaseVisualizer):
        def visualize(self, **states):
            #return self.v_gt_seg(states['seg'])
            return self.v_pca(embedding=states['embedding'])

    visualizer = TestVisualizer()
    seg = torch.tensor(np.linspace(0, 1, 5))
    print(seg)
    embedding = torch.rand(2, 16, 11, 12, 13)
    mask = torch.rand(2, 1, 11, 12, 13) > 0
    states = {
        'seg': seg,
        'mask': mask,
        'embedding': embedding
    }

    print(visualizer.visualize(**states).shape)
