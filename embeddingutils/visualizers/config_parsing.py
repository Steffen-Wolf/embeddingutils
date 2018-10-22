from embeddingutils.visualizers.visualizers import \
    GridVisualizer, \
    PcaVisualizer, \
    MaskedPcaVisualizer, \
    SegmentationVisualizer, \
    InputVisualizer, \
    TargetVisualizer, \
    PredictionVisualizer
from embeddingutils.visualizers.base import ContainerVisualizer
from embeddingutils.visualizers.base import VisualizationCallback
from inferno.utils.io_utils import yaml2dict


def get_single_key_value_pair(d):
    assert isinstance(d, dict), f'{d}'
    assert len(d) == 1, f'{d}'
    return list(d.keys())[0], list(d.values())[0]


def get_visualizer_class(name):
    assert name in globals(), f"Transform {name} not found."
    return globals().get(name)


def get_visualizer(config):
    config = yaml2dict(config)
    name, args = get_single_key_value_pair(config)
    if name not in globals():
        return config
    visualizer = get_visualizer_class(name)
    if issubclass(visualizer, ContainerVisualizer):  # container visualizer: parse sub-visualizers first
        assert isinstance(args['visualizers'], list), f'{args["visualizers"]}, {type(args["visualizers"])}'
        args['visualizers'] = [get_visualizer(c) for c in args['visualizers']]
    return visualizer(**args)


def get_visualization_callback(config):
    # a visualization config is parsed like this:
    # 1. input formats and global slicing is read
    # 2. the visualizers are processed:\
    #       - check if ContainerVisualizer -> if yes costruct sub-visualizers
    #       - else, just pass arguments and construct visualizer
    config = yaml2dict(config)
    name, config = get_single_key_value_pair(config)
    visualizer = get_visualizer(config)
    callback = VisualizationCallback(name, visualizer)
    return callback


if __name__ == '__main__':
    import torch
    import numpy as np

    config = './example_configs/test_config.yml'
    config = yaml2dict(config)
    callback = get_visualization_callback(config)
    v = callback.visualizer

    tensor = torch.Tensor(np.random.randn(2, 32, 10, 8, 8))
    result = v(inputs=tensor,
               prediction=2 * tensor,
               target=tensor > 0)
    print(result)
    print(result.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(result)
    plt.show()

