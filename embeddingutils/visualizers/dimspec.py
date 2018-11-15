import numpy as np
import torch
from copy import copy
from collections import OrderedDict
import pdb


def join_specs(spec1, spec2):
    result = copy(spec1)
    for d in spec2:
        if d not in result:
            result.append(d)
    return result


def extend_dim(tensor, in_spec, out_spec, return_spec=False):
    assert all(d in out_spec for d in in_spec)
    i = 0
    for d in out_spec:
        if d not in in_spec:
            tensor = tensor.unsqueeze(i)
        i += 1
    if return_spec:
        new_spec = out_spec + [d for d in in_spec if d not in out_spec]
        return tensor, new_spec
    else:
        return tensor


def _moving_permutation(length, origin, goal):
    result = []
    for i in range(length):
        if i == goal:
            result.append(origin)
        elif (i < goal and i < origin) or (i > goal and i > origin):
            result.append(i)
        elif goal < i <= origin:
            result.append(i-1)
        elif origin <= i < goal:
            result.append(i+1)
        else:
            assert False
    return result


def collapse_dim(tensor, to_collapse, collapse_into=None, spec=None, return_spec=False):
    spec = list(range(len(tensor.shape))) if spec is None else spec
    assert to_collapse in spec, f'{to_collapse}, {spec}'
    i_from = spec.index(to_collapse)
    if collapse_into is None:
        i_delete = i_from
        assert tensor.shape[i_delete] == 1, f'{to_collapse}, {tensor.shape[i_delete]}'
        tensor = tensor.squeeze(i_delete)
    else:
        assert collapse_into in spec, f'{collapse_into}, {spec}'
        i_to = spec.index(collapse_into)
        i_to = i_to + 1 if i_from > i_to else i_to
        tensor = tensor.permute(_moving_permutation(len(spec), i_from, i_to))
        new_shape = tensor.shape[:i_to-1] + (tensor.shape[i_to-1] * tensor.shape[i_to],) + tensor.shape[i_to+1:]
        tensor = tensor.contiguous().view(new_shape)
    if return_spec:
        new_spec = [spec[i] for i in range(len(spec)) if i is not i_from]
        return tensor, new_spec
    else:
        return tensor


def convert_dim(tensor, in_spec, out_spec, collapsing_rules=None, uncollapsing_rules=None,
                return_spec=False, return_inverse_kwargs=False):
    assert len(tensor.shape) == len(in_spec), f'{tensor.shape}, {in_spec}'
    temp_spec = copy(in_spec)
    # uncollapse as specified
    if uncollapsing_rules is not None:
        for rule in uncollapsing_rules:
            if isinstance(rule, tuple):
                rule = {
                    'to_uncollapse': rule[0],
                    'uncollapsed_length': rule[1],
                    'uncollapse_into': rule[2]
                }
            tensor, temp_spec = uncollapse_dim(tensor, spec=temp_spec, **rule, return_spec=True)

    # bring tensor's spec in same order as out_spec, with dims not present in out_spec at the end
    joined_spec = join_specs(out_spec, in_spec)
    order = list(np.argsort([joined_spec.index(d) for d in temp_spec]))
    tensor = tensor.permute(order)
    temp_spec = [temp_spec[i] for i in order]
    # unsqueeze to match out_spec
    tensor = extend_dim(tensor, temp_spec, joined_spec)
    temp_spec = joined_spec
    # apply dimension collapsing rules
    inverse_uncollapsing_rules = []  # needed if inverse is requested
    if collapsing_rules is not None:
        for rule in collapsing_rules:
            if rule[0] in temp_spec:
                inverse_uncollapsing_rules.append({
                    'to_uncollapse': rule[1],
                    'uncollapsed_length': tensor.shape[temp_spec.index(rule[0])],
                    'uncollapse_into': rule[0]
                })
                # print(f'{tensor.shape}, {temp_spec}, {out_spec}')
                tensor, temp_spec = collapse_dim(tensor, spec=temp_spec, to_collapse=rule[0], collapse_into=rule[1],
                                                 return_spec=True)
    # drop trivial dims not in out_spec
    for d in reversed(temp_spec):
        if d not in out_spec:
            tensor, temp_spec = collapse_dim(tensor, to_collapse=d, spec=temp_spec, return_spec=True)

    assert all(d in out_spec for d in temp_spec), \
        f'{temp_spec}, {out_spec}: please provide appropriate collapsing rules'
    tensor = extend_dim(tensor, temp_spec, out_spec)

    result = [tensor]
    if return_spec:
        result.append(temp_spec)
    if return_inverse_kwargs:
        inverse_kwargs = {
            'in_spec': out_spec,
            'out_spec': in_spec,
            'uncollapsing_rules': inverse_uncollapsing_rules[::-1]
        }
        result.append(inverse_kwargs)
    if len(result) == 1:
        return result[0]
    else:
        return result


def uncollapse_dim(tensor, to_uncollapse, uncollapsed_length, uncollapse_into=None, spec=None, return_spec=False):
    # puts the new dimension directly behind the old one
    spec = list(range(len(tensor.shape))) if spec is None else spec
    assert to_uncollapse in spec, f'{to_uncollapse}, {spec}'
    assert uncollapse_into not in spec, f'{uncollapse_into}, {spec}'
    assert isinstance(tensor, torch.Tensor), f'unexpected type: {type(tensor)}'
    i_from = spec.index(to_uncollapse)
    assert tensor.shape[i_from] % uncollapsed_length == 0, f'{tensor.shape[i_from]}, {uncollapsed_length}'
    new_shape = tensor.shape[:i_from] + \
                (tensor.shape[i_from]//uncollapsed_length, uncollapsed_length) + \
                tensor.shape[i_from + 1:]
    tensor = tensor.contiguous().view(new_shape)
    if return_spec:
        assert uncollapse_into is not None
        new_spec = copy(spec)
        new_spec.insert(i_from + 1, uncollapse_into)
        return tensor, new_spec
    else:
        return tensor


def add_dim(tensor, length=1, new_dim=None, spec=None, return_spec=False):
    tensor = tensor[None].repeat([length] + [1] * len(tensor.shape))
    if return_spec:
        return tensor, [new_dim] + spec
    else:
        return tensor


class SpecFunction:
    def __init__(self, in_specs, out_spec):
        self.internal_in_specs = {key: list(in_specs[key]) for key in in_specs}
        self.internal_out_spec = list(out_spec)
        assert (all('B' in spec for spec in self.internal_in_specs.values())) or \
               (all('B' not in spec for spec in self.internal_in_specs.values())), \
            f'"B" has to be in all or none of the internal specs: {self.internal_in_specs}'
        if all('B' not in spec for spec in self.internal_in_specs.values()):
            self.parallel = False
            self.internal_in_specs_with_B = {key: ['B'] + self.internal_in_specs[key] for key in in_specs}
        else:
            self.parallel = True
            self.internal_in_specs_with_B = self.internal_in_specs
        #self.default_out_spec = list(default_out_spec) if default_out_spec is not None else self.internal_out_spec

    def __call__(self, *args, out_spec=None, return_spec=False, **kwargs):
        given_spec_kwargs = [kw for kw in self.internal_in_specs if kw in kwargs]

        # determine the extra specs in the input. they will be put in the 'B' spec.
        extra_given_in_specs = OrderedDict()
        for kw in given_spec_kwargs:  # loop over given argument names that support dynamic specs
            assert len(kwargs[kw]) == 2, f'{kwargs[kw]}'  # has to be a pair of (arg, spec)
            arg, spec = kwargs[kw]
            kwargs[kw] = (arg, list(spec))  # make spec list, in case it is given as string
            # assert all(d in spec for d in extra_given_in_specs), \
            #     f'if extra specs are given, all input args need to have them: {kw}, {extra_given_in_specs}, {spec}'
            extra_given_in_specs.update({d: arg.shape[spec.index(d)] for d in spec
                                         if (d not in self.internal_in_specs[kw] and d not in extra_given_in_specs)})

        # print('extra specs', extra_given_in_specs)

        # add and repeat extra dimensions not present in some of the inputs
        for kw in given_spec_kwargs:
            arg, spec = kwargs[kw]
            for d in extra_given_in_specs:
                if d not in spec:
                    length = extra_given_in_specs[d]
                    arg, spec = add_dim(arg, length=length, new_dim=d, spec=spec, return_spec=True)
            kwargs[kw] = arg, spec

        # remove specs from extra given specs that are present in internal_in_specs
        # TODO: right now, this is unnecessary. allow for partially missing dims in the input_specs!
        for d in extra_given_in_specs:
            if not all(d not in spec for spec in self.internal_in_specs.values()):
                extra_given_in_specs.pop(d)
                assert d not in self.internal_out_spec, \
                    f'spec {d} is an internal_out_spec, cannot be an extra given spec'

        collapsing_rules = [(d, 'B') for d in extra_given_in_specs]
        for kw in self.internal_in_specs:
            assert kw in kwargs, \
                f'Missing key {kw}. Provided keys were {kwargs.keys()} in SpecFunction of class {type(self)}'
            arg, spec = kwargs[kw]
            # make it so 'B' is present
            if 'B' not in spec:
                arg, spec = extend_dim(arg, spec, ['B'] + spec, return_spec=True)
            # collapse the extra dimensions of the input into B
            arg = convert_dim(arg, spec, self.internal_in_specs_with_B[kw], collapsing_rules)
            kwargs[kw] = arg  # finally update kwargs dictionary

        if self.parallel:
            result = self.internal(*args, **kwargs)
            spec = self.internal_out_spec
        else:
            n_batch = kwargs[list(self.internal_in_specs.keys())[0]].shape[0] if len(self.internal_in_specs) > 0 else 1
            result = torch.stack(
                [self.internal(*args, **{kw: kwargs[kw] if kw not in self.internal_in_specs else kwargs[kw][i]
                                         for kw in kwargs})
                 for i in range(n_batch)], dim=0)
            spec = ['B'] + self.internal_out_spec

        assert isinstance(result, torch.Tensor), f'unexpected type: {type(result)}'

        # uncollapse the previously collapsed dims
        if not (self.parallel and ('B' not in self.internal_out_spec)):  # skip if function 'consumes' parallel dimension
            for i in reversed(range(len(extra_given_in_specs))):
                d = list(extra_given_in_specs.keys())[i]
                length = extra_given_in_specs[d]
                result, spec = uncollapse_dim(result, to_uncollapse='B', uncollapsed_length=length, uncollapse_into=d,
                                              spec=spec, return_spec=True)

        # finally, convert to out_spec, if specified
        if out_spec is not None:
            out_spec = list(out_spec)
            result, spec = convert_dim(result, in_spec=spec, out_spec=out_spec, return_spec=True)
        if return_spec:
            return result, spec
        else:
            return result

    def internal(self, *args, **kwargs):
        pass


if __name__ == '__main__':

    # t = torch.Tensor(np.random.randn(2, 3, 4, 5))
    # spec = list('ABCD')
    # out_spec = list('DA')  # list('EDA')
    # collapsing_rules = ['BD', 'CA']
    # before = t.clone()
    #
    # t, inverse_kwargs = convert_dim(t, in_spec=spec, out_spec=out_spec,
    #                                 collapsing_rules=collapsing_rules,
    #                                 return_inverse_kwargs=True)
    # print(inverse_kwargs)
    #
    # t = convert_dim(t, **inverse_kwargs)
    # print(t.shape)
    # print(before.shape)
    # print((t == before).float().mean())


    # t = torch.Tensor([[[0, 1], [0, 2]], [[5, 10], [5, 20]]])
    # t = torch.Tensor([[1, 2, 3], [2, 3, 4]])
    # print(t)
    # print(t.shape)
    # t = convert_dim(t, in_spec=[0, 1], out_spec=[0,], collapsing_rules=[(1, 0)])
    # print(t)
    # print(t.shape)
    #
    # assert False

    class MaskArrays(SpecFunction):
        def __init__(self, **super_kwargs):
            super(MaskArrays, self).__init__(in_specs={'mask': 'BW', 'array': 'BWC'}, out_spec='BWC', **super_kwargs)

        def internal(self, mask, array, value=0.0):
            result = array.clone()
            result[mask == 0] = value
            return result

    class MultiScale(SpecFunction):
        def __init__(self, **super_kwargs):
            super(MultiScale, self).__init__(in_specs={'data': 'BX', 'scales': 'BS'}, out_spec='BXS', **super_kwargs)

        def internal(self, data, scales):
            return data[:, :, None] * scales[:, None, :]


    def rand_tensor(*shape):
        return torch.Tensor(np.random.randn(*shape))

    inputs = {
        'array': (rand_tensor(1, 3, 4, 5), 'XYFH'),
        'mask': (rand_tensor(1, 5) > 0, list('XH')),
        'value': 100.0,
        'out_spec': 'XYHWCF',
    }
    maskArrays = MaskArrays()
    result = maskArrays(**inputs)
    print(result[0, 0, :, 0, :, :])
    print(result.shape)

    inputs = {
        'data': (rand_tensor(2, 3), 'XY'),
        'scales': (torch.Tensor([0, 1, 100]).float(), 'S')
    }
    multiScale = MultiScale()
    print(multiScale(**inputs))

    #
    # spec1 = list('CHW')
    # spec2 = list('BWFDHC')
    #
    # t = rand_tensor(2, 6, 4)
    #
    # print(_moving_permutation(5, 0, 2))
    # print(_moving_permutation(5, 2, 0))
    # print(_moving_permutation(5, 1, 4))
    # print(_moving_permutation(5, 4, 1))
    #
    # #print(collapse_dim(t, 'H', 'C', spec1))
    # #print(collapse_dim(t, 'C', 'W', spec1))
    # #print(collapse_dim(t, 'W', 'C', spec1))
    #
    # t, spec1 = uncollapse_dim(t, spec=spec1, to_uncollapse='H', uncollapsed_length=3, uncollapse_into='X', return_spec=True)
    # print(t.shape)
    #
    # #print(convert_dim(t, spec1, spec2, collapsing_rules=['HW', 'WF', 'FB']).shape)
