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
import torch.nn.functional as F


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


def parse_pre_func(pre_info):
    if isinstance(pre_info, list):
        # parse as concatenation
        funcs = [parse_pre_func(info) for info in pre_info]

        def pre_func(x):
            for f in funcs:
                x = f(x)
            return x

        return pre_func
    elif isinstance(pre_info, dict):
        pre_name, arg_info = get_single_key_value_pair(pre_info)
    elif isinstance(pre_info, str):
        pre_name = pre_info
        arg_info = []
    else:
        assert False, f'{pre_info}'
    if isinstance(arg_info, dict):
        kwargs = arg_info
        args = []
    elif isinstance(arg_info, list):
        kwargs = {}
        args = arg_info
    pre_func_without_args = getattr(F, pre_name)
    pre_func = lambda x: pre_func_without_args(x, *args, **kwargs)
    return pre_func


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
            index = map_from_info.pop('index', None)
            if index is not None:  # allow for index to be left unspecified
                index = int(index)
                state = state[index]
                assert isinstance(state, torch.Tensor), f'{map_from_key}, {type(state)}'
        if 'pre' in map_from_info:
            pre_func = parse_pre_func(map_from_info.pop('pre'))
        else:
            pre_func = None
        # figure out spec
        if 'spec' in map_from_info:
            spec = list(map_from_info.pop('spec'))
        else:
            if isinstance(state_info, tuple):
                spec = state_info[1]
            else:
                dimensionality = len(state.shape) if isinstance(state, torch.Tensor) else len(state[0].shape)
                assert dimensionality in DEFAULT_SPECS, f'{map_from_key}, {dimensionality}'
                spec = DEFAULT_SPECS[dimensionality]
        # get the slices
        map_from_slices = parse_named_slicing(map_from_info, spec)
        # finally map the state
        if isinstance(state, torch.Tensor):
            assert len(state.shape) == len(spec), f'{state.shape}, {spec} ({map_from_key})'
            state = state[map_from_slices]
        elif isinstance(state, list):
            assert all(len(s.shape) == len(spec) for s in state), f'{[s.shape for s in state]}, {spec} ({map_from_key})'
            state = [s[map_from_slices] for s in state]
        else:
            assert False, f'state has to be list or tensor: {map_from_key}, {type(state)}'

        if pre_func is None:
            result[map_to] = (state, spec)
        else:
            result[map_to] = (pre_func(state), spec)
    return result


class BaseVisualizer(SpecFunction):

    def __init__(self, input_mapping=None, suppress_colorization=False,
                 cmap=None, background_label=None, background_color=None,
                 value_range=None,
                 **super_kwargs):
        # input mapping is a dictionary. Its keys are argument names that are to be passed to visualize, and its values
        # specify where to fetch them in the state dict supplied to __call__.
        # The Syntax of this specification can be one of the following:
        #   - just the name of the entry in the state dict
        #   - a list, consisting of the name of the entry in the state dict and a slicing configuration
        super(BaseVisualizer, self).__init__(**super_kwargs)
        self.input_mapping = input_mapping
        self.suppress_colorization = suppress_colorization
        self.colorize = Colorize(cmap=cmap, background_color=background_color, background_label=background_label,
                                 value_range=value_range)

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


def to_img_grid(tensor, spec, return_spec=False):
    collapsing_rules = [('D', 'B'), ('T', 'B')]
    out_spec = ['B', 'C', 'Color', 'H', 'W']
    tensor, spec = convert_dim(tensor, in_spec=spec, out_spec=out_spec, collapsing_rules=collapsing_rules,
                               return_spec=True)
    if return_spec:
        return tensor, spec
    else:
        return tensor


class ContainerVisualizer(BaseVisualizer):

    def __init__(self, visualizers, in_spec, out_spec, extra_in_specs=None, input_mapping=None,
                 suppress_colorization=True, **super_kwargs):
        # in_spec: spec the outputs of all visualizers will be converted to
        self.in_spec = in_spec
        self.visualizers = visualizers
        self.n_visualizers = len(visualizers)
        self.visualizer_kwarg_names = ['visualized_' + str(i) for i in range(self.n_visualizers)]
        in_specs = dict() if extra_in_specs is None else extra_in_specs
        in_specs.update({self.visualizer_kwarg_names[i]: in_spec for i in range(self.n_visualizers)})
        super(ContainerVisualizer, self).__init__(
            input_mapping=input_mapping,
            in_specs=in_specs,
            out_spec=out_spec,
            suppress_colorization=suppress_colorization
        )

    def __call__(self, return_spec=False, **states):
        states = copy(states)
        # map input keywords and apply slicing
        states = apply_slice_mapping(self.input_mapping, states)
        # apply visualizers and update state dict
        for i in range(self.n_visualizers):
            states[self.visualizer_kwarg_names[i]] = self.visualizers[i](**states, return_spec=True)

        return super(ContainerVisualizer, self).__call__(**states, return_spec=return_spec)

    def internal(self, **states):
        visualizations = []
        for name in self.visualizer_kwarg_names:
            visualizations.append(states[name])
        return self.combine(*visualizations, **states)

    def combine(self, *visualizations, **extra_states):
        raise NotImplementedError


def _remove_alpha(tensor, background_brightness=1):
    return torch.ones_like(tensor[..., :3]) * background_brightness * (1-tensor[..., 3:4]) + \
           tensor[..., :3] * tensor[..., 3:4]


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
        image = _remove_alpha(self.visualizer(**self.get_trainer_states())).permute(2, 0, 1)  # to [Color, Height, Width]
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

