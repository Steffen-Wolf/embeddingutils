import numpy as np
import torch
from inferno.trainers.callbacks.base import Callback
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from embeddingutils.visualizers.dimspec import SpecFunction, convert_dim, collapse_dim
from embeddingutils.visualizers.colorization import Colorize
from inferno.utils.io_utils import yaml2dict
from inferno.io.volumetric import volumetric_utils as vu
import torchvision.utils as vutils
from copy import copy


def get_single_key_value_pair(d):
    assert isinstance(d, dict), f'{d}'
    assert len(d) == 1, f'{d}'
    return list(d.keys())[0], list(d.values())[0]


def list_of_dicts_to_dict(list_of_dicts):
    result = dict()
    for d in list_of_dicts:
        key, value = get_single_key_value_pair(d)
        result[key] = value
    return result


def parse_named_slicing(slicing, spec):
    """Parse a slicing as a list of slice objects."""
    if slicing is None:
        return slicing
    elif isinstance(slicing, str):
        slices = vu.parse_data_slice(slicing)
        assert len(slices) <= len(spec)
        return list(slices)
    elif isinstance(slicing, list):
        slicing = list_of_dicts_to_dict(slicing)
    else:
        assert isinstance(slicing, dict)
    # Build slice objects
    slices = []
    for d in spec:
        if d not in slicing:
            slices.append(slice(None, None, None))
        else:
            dim_slice = str(slicing[d])
            # Get rid of whitespace
            dim_slice.replace(' ', '')
            indices = dim_slice.split(':')
            if len(indices) == 1:
                start, stop, step = indices[0], int(indices[0]) + 1, None
            elif len(indices) == 2:
                start, stop, step = indices[0], indices[1], None
            elif len(indices) == 3:
                start, stop, step = indices
            else:
                raise RuntimeError
            # Convert to ints
            start = int(start) if start != '' else None
            stop = int(stop) if stop != '' else None
            step = int(step) if step is not None and step != '' else None
            # Build slices
            slices.append(slice(start, stop, step))
    # Done.
    return slices


DEFAULT_SPECS = {
    3: list('BHW'),
    4: list('BCHW'),
    5: list('BCDHW'),
    6: list('BCTDHW')
}


def apply_slice_mapping(mapping, states, include_old_states=True):
    mapping = copy(mapping)
    # assumes states are tuples of (tensor, spec) if included in mapping
    assert isinstance(states, dict)
    if include_old_states:
        result = copy(states)
    else:
        result = dict()
    if mapping is None:
        return result

    global_slice_info = mapping.pop('global', {})
    if isinstance(global_slice_info, list):
        global_slice_info = list_of_dicts_to_dict(global_slice_info)

    for map_to in mapping:
        map_from_info = mapping[map_to]
        if isinstance(map_from_info, str):
            map_from_key = map_from_info
            map_from_info = {}
        elif isinstance(map_from_info, (list, dict)):
            if isinstance(map_from_info, list) and isinstance(map_from_info[0], str):
                map_from_key = map_from_info[0]
                map_from_info = map_from_info[1:]
            else:
                map_from_key = map_to
            if isinstance(map_from_info, list):
                map_from_info = list_of_dicts_to_dict(map_from_info)
        # apply the global slicing
        temp = copy(global_slice_info)
        temp.update(map_from_info)
        map_from_info = temp

        # figure out state
        state_info = states[map_from_key]  # either (state, spec) or state
        state = state_info[0] if isinstance(state_info, tuple) else state_info
        if not isinstance(state, (tuple, torch.Tensor)) and isinstance(state, list):
            #assert 'index' in map_from_info, \
            #    f'if you give a list to the visualizer, please provide an index to use. {map_from_info}'

            index = map_from_info.pop('index', 0)
            index = int(index)
            state = state[index]
            assert isinstance(state, torch.Tensor), f'{map_from_key}, {type(state)}'

        # figure out spec
        if 'spec' in map_from_info:
            spec = list(map_from_info.pop('spec'))
        else:
            if isinstance(state_info, tuple):
                spec = state_info[1]
            else:
                assert len(state.shape) in DEFAULT_SPECS, f'{map_from_key}, {len(state.shape)}'
                spec = DEFAULT_SPECS[len(state.shape)]
        # get the slices
        map_from_slices = parse_named_slicing(map_from_info, spec)
        # finally map the state
        assert len(state.shape) == len(spec), f'{state.shape}, {spec} ({map_from_key})'
        # print()
        # print(map_to)
        # print(map_from_slices)
        # print(state.shape)
        # print(spec)
        result[map_to] = (state[map_from_slices], spec)
    return result


def to_img_grid(tensor, spec, return_spec=False):
    collapsing_rules = [('D', 'B'), ('T', 'B')]
    out_spec = ['B', 'C', 'Color', 'H', 'W']
    tensor, spec = convert_dim(tensor, in_spec=spec, out_spec=out_spec, collapsing_rules=collapsing_rules,
                               return_spec=True)
    if return_spec:
        return tensor, spec
    else:
        return tensor


class BaseVisualizer(SpecFunction):

    def __init__(self, input_mapping=None, suppress_colorization=False, cmap=None, **super_kwargs):
        # input mapping is a dictionary. Its keys are argument names that are to be passed to visualize, and its values
        # specify where to fetch them in the state dict supplied to __call__.
        # The Syntax of this specification can be one of the following:
        #   - just the name of the entry in the state dict
        #   - a list, consisting of the name of the entry in the state dict and a slicing configuration
        super(BaseVisualizer, self).__init__(**super_kwargs)
        self.input_mapping = input_mapping
        self.suppress_colorization = suppress_colorization
        self.colorize = Colorize(cmap=cmap)

    def __call__(self, return_spec=False, **states):
        # map input keywords and apply slicing
        states = apply_slice_mapping(self.input_mapping, states)
        # apply visualize
        result, spec = super(BaseVisualizer, self).__call__(**states, return_spec=True)
        # color the result, if not suppressed
        result = result.float()
        if not self.suppress_colorization:
            out_spec = spec if 'Color' in spec else spec + ['Color']
            result, spec = self.colorize(tensor=(result, spec), out_spec=out_spec, return_spec=True)
        if return_spec:
            return result, spec
        else:
            return result

    def internal(self, *args, **kwargs):
        # essentially rename internal to visualize
        return self.visualize(*args, **kwargs)

    def visualize(self, **states):
        pass


class ContainerVisualizer(BaseVisualizer):

    def __init__(self, visualizers, input_mapping=None):
        super(ContainerVisualizer, self).__init__(input_mapping=input_mapping, in_specs={}, out_spec=[])
        self.visualizers = visualizers

    def __call__(self, **states):
        # map input keywords and apply slicing
        states = apply_slice_mapping(self.input_mapping, states)
        # apply visualizers
        image_grids = [v(**states, return_spec=True) for v in self.visualizers]
        image_grids = [to_img_grid(t[0], spec=t[1]) for t in image_grids]
        # shapes are now (B, C, Color, H, W)
        # put W into B
        result = self.combine(*image_grids)
        return result.float()

    def combine(self, *image_grids):
        pass


class VisualizationCallback(Callback):
    def __init__(self, name, visualizer):
        super(VisualizationCallback, self).__init__()
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
                state = state.cpu().detach()  #logging is done on the cpu
            result[s] = state
        return result

    def do_logging(self, **_):
        print(f'Logging now: {self.name}')
        assert isinstance(self.logger, TensorboardLogger)
        writer = self.logger.writer
        image = self.visualizer(**self.get_trainer_states()).permute(2, 0, 1)  # to [Color, Height, Width]
        pre = 'training' if self.trainer.model.training else 'validation'
        writer.add_image(tag=pre+'_'+self.name, img_tensor=image, global_step=self.trainer.iteration_count)
        # TODO: make Tensorboard logger accept rgb images
        #self.logger.log_object(self.name, image)

    def end_of_training_iteration(self, **_):
        # log_now = self.logger.log_images_now  # TODO: ask Nasim about this
        log_now = self.logger.log_images_every.match(
            iteration_count=self.trainer.iteration_count,
            epoch_count=self.trainer.epoch_count,
            persistent=False)
        if log_now:
            self.do_logging()

    def end_of_validation_run(self, **_):
        self.do_logging()


if __name__ == '__main__':
    class TestVisualizer(BaseVisualizer):
        pass

    config = yaml2dict('example_configs/example_config.yml')
    print(config['visualization_config']['stack_vertical'][2])

    slice_mapping = config['visualization_config']['stack_vertical'][2]['PointedToVisualizer']['input_mapping']
    print(parse_named_slicing(slice_mapping['vectorfield'][1:],
          spec=list('BFCHW')))

    test_tensor = torch.Tensor(np.stack(np.meshgrid(*[np.arange(3) for _ in range(3)]), axis=-1))
    print(test_tensor)
    print(test_tensor.shape)
    test_states = {'prediction': (test_tensor, 'BXC')}
    print([s[0].shape for s in apply_slice_mapping(slice_mapping, test_states).values()])
    #print(vu.parse_data_slice('0:1'))
    # visualizer = TestVisualizer()
    # seg = torch.tensor(np.linspace(0, 1, 5))
    # print(seg)
    # embedding = torch.rand(2, 16, 11, 12, 13)
    # mask = torch.rand(2, 1, 11, 12, 13) > 0
    # states = {
    #     'seg': seg,
    #     'mask': mask,
    #     'embedding': embedding
    # }

