import numpy as np
import torch
import torch.nn.functional as F


def label_equal_similarity(x, y, dim=0):
    assert x.shape[dim] == 1, 'label images should have one channel only'
    return ((x == y).squeeze(dim=dim)).float()


def ignore_label_mask_similarity(x, y, dim=0, ignore_label=0):
    # returns a mask that is 0 where x or y is to be ignored
    assert x.shape[dim] == 1, 'label images should have one channel only'
    return (torch.stack([x != ignore_label, y != ignore_label], dim=0)).min(dim=0)[0].squeeze(dim=dim).byte()


def euclidean_distance(x, y, dim=0):
    return torch.sqrt(((x - y)**2).sum(dim))


def squared_euclidean_distance(x, y, dim=0):
    return ((x - y) ** 2).sum(dim)


def logistic_similarity(x, y, dim=0, offset=None):
    if offset is None:
        sig = 1
    else:
        sig = np.linalg.norm(offset)
    return 2 / (1 + (1 / torch.exp(-squared_euclidean_distance(x, y, dim=dim)/(2 * sig**2)).clamp(min=1e-6)))


def normalized_cosine_similarity(x, y, dim=0):
    return 0.5 * (1 + F.cosine_similarity(x, y, dim=dim))


def offset_slice(offset, reverse=False, extra_dims=0):
    def shift(o):
        if o == 0:
            return slice(None)
        elif o > 0:
            return slice(o, None)
        else:
            return slice(0, o)
    if not reverse:
        return (slice(None),) * extra_dims + tuple(shift(int(o)) for o in offset)
    else:
        return (slice(None),) * extra_dims + tuple(shift(-int(o)) for o in offset)


def offset_padding(offset):
    result = []
    for o in reversed(offset):
        result.append(int(max(-o, 0)))
        result.append(int(max(o, 0)))
    return tuple(result)


def get_offsets(offsets):
    if isinstance(offsets, str):
        if offsets == 'default-3D':
            offsets = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                                [0, -9, 0], [0, 0, -9], [0, -9, -9], [0, 9, -9],
                                [0, -9, 4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
                                [-1, -1, 0], [-1, 0, -1],
                                [0, -27, 0], [0, 0, -27]], int)
        elif offsets == 'default-2D':
            offsets = np.array([[-1, 0], [0, -1],
                                [-9, 0], [0, -9],
                                [-9, -9], [9, -9],
                                [-9, -4], [-4, -9], [4, -9], [9, -4],
                                [-27, 0], [0, -27]], int)
        else:
            assert False, "Please provide a list of offsets or one of ['default-3D', 'default-2D']"
    return offsets if isinstance(offsets, np.ndarray) else np.array(offsets, int)


def embedding_to_affinities(emb, offsets='default-3D', affinity_measure=euclidean_distance, pass_offset=False):
    # if len(emb.shape) = n + 1 + len(offsets[0]):
    # function is parallel over first n dimensions
    # the (n+1)th dimension is assumed to be embedding dimenstion
    # rest are going to be shifted by offsets

    offsets = get_offsets(offsets)

    result = []
    emb_axis = len(emb.shape) - len(offsets[0]) - 1
    extra_dims = len(emb.shape) - len(offsets[0])
    for off in offsets:
        if all(abs(o) < s for o, s in zip(off, emb.shape[-len(off):])):
            s1 = offset_slice(off, reverse=True, extra_dims=extra_dims)
            s2 = offset_slice(off, extra_dims=extra_dims)
            if not pass_offset:
                aff = affinity_measure(emb[s1], emb[s2], dim=emb_axis)
            else:
                aff = affinity_measure(emb[s1], emb[s2], dim=emb_axis, offset=off)
            aff = F.pad(aff, offset_padding(off), value=1.)
        else:
            print('warning: offset bigger than image')
            aff = torch.zeros(emb.shape[:emb_axis] + emb.shape[emb_axis+1:]).to(emb.device)
        result.append(aff)

    return torch.stack(result, dim=emb_axis)


class EmbeddingToAffinities(torch.nn.Module):
    def __init__(self, offsets='default-3D', affinity_measure=logistic_similarity, pass_offset=True):
        super(EmbeddingToAffinities, self).__init__()
        self.offsets = get_offsets(offsets)
        self.affinity_measure = affinity_measure
        self.pass_offset = pass_offset
        # TODO: compute offset slices and paddings in constructor

    def forward(self, emb):
        if self.pass_offset:
            return embedding_to_affinities(emb,
                                           offsets=self.offsets,
                                           affinity_measure=self.affinity_measure,
                                           pass_offset=self.pass_offset)
        else:
            return embedding_to_affinities(emb,
                                           offsets=self.offsets,
                                           affinity_measure=self.affinity_measure)


if __name__ == '__main__':
    t = torch.arange(25).view(5, 5)
    print(t)
    off = (2, 0)
    pad = offset_padding(off)
    print(pad)
    s = offset_slice(off)
    print(s)
    print(t[s])
    print(F.pad(t[s], pad))
    print('-'*100)
    emb = torch.eye(10).view(1, 10, 10)
    print(emb)
    offsets = ((1, -5), (1, -1))
    print(embedding_to_affinities(emb, offsets))
