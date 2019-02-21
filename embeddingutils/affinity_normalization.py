from .affinities import offset_slice, get_offsets
import numpy as np
from scipy.interpolate import interp1d


def _get_probability_mapping(pred_aff, gt_aff, N=None):
    N = 100 if N is None else N
    f, a_star = pred_aff.flatten(), gt_aff.flatten()
    order = np.argsort(f)
    f, a_star = f[order], a_star[order]
    b = np.array([np.floor(i * len(f) / N) for i in range(N + 1)]).astype(np.int32)
    intervals = [slice(b[i], b[i + 1]) for i in range(N)]
    p = np.array([np.mean(a_star[interval]) for interval in intervals])
    b[-1] -= 1
    c = f[b]
    d = [-np.inf] + [c[i // 2] if i % 2 == 0 else (c[(i - 1) // 2] + c[(i + 1) // 2]) / 2
                     for i in range(2 * N + 1)] + [np.inf]
    p = np.concatenate([[0], p, [1]])  # this assumes affinities to bo small when repulsive

    e = [0, ] + [p[i // 2] if i % 2 == 0 else (p[(i - 1) // 2] + p[(i + 1) // 2]) / 2
                 for i in range(1, 2 * N + 2)] + [1, ]
    return interp1d(d, e)


def get_per_channel_mapping(pred_aff, gt_aff, offsets, return_list_of_mappings=False, **mapping_kwargs):
    """
    given lists of predicted and ground-truth affinities (both in (B, Offsets, (D), H, W) format), returns a mapping
    that can be applied on predicted affinities (in the same format) to make them somehow closer to edge probabilities.
    """
    offsets = get_offsets(offsets)
    assert len(offsets) == pred_aff[0].shape[0]
    assert pred_aff.shape == gt_aff.shape
    mappings = []
    for i, offset in enumerate(offsets):
        print(f'Computing mapping for offset {offset}')
        flat_pred, flat_gt = np.concatenate(
            [np.stack([pred[i], gt[i]])[(slice(None),) + offset_slice(-offset)].reshape(2, -1)
             for pred, gt in zip(pred_aff, gt_aff)])
        mappings.append(_get_probability_mapping(flat_pred, flat_gt, **mapping_kwargs))

    def mapping(aff):
        assert len(aff.shape) == len(offsets[0]) + 2, f'must be shape (Batch, Offsets, (D), H, W)'
        result = np.zeros_like(aff)
        for i, offset in enumerate(offsets):
            result[:, i] = 0
            result[(slice(None), slice(i, i + 1)) + offset_slice(-offset)] = mappings[i](
                aff[(slice(None), slice(i, i + 1)) + offset_slice(-offset)])
        return result

    if return_list_of_mappings:
        return mapping, mappings
    else:
        return mapping
