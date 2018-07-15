import torch
from torch.nn.functional import cosine_similarity
from skimage.measure import label
import constrained_mst as cmst
import numpy as np
import hdbscan
import collections
from embeddingutils.affinities import embedding_to_affinities, get_offsets, logistic_similarity


def mws_segmentation(embedding, offsets='default-3D', affinity_measure=logistic_similarity,
                     ATT_C=3, repulsive_strides=None, percentile=5 ,return_affinities=False):

    if offsets == 'default-2D':
        ATT_C = 2
    offsets = get_offsets(offsets)

    emb_shape = embedding.shape
    img_shape = embedding.shape[-len(offsets[0]):]
    n_img_dims = len(img_shape)

    if repulsive_strides is None:
        repulsive_strides = (1,) * (n_img_dims - 2) + (8, 8)

    # actually not needed, huge offsets are fine
    #for off in offsets:
    #    assert all(abs(o) < s for o, s in zip(off, emb.shape[-len(off):])), \
    #        f'offset {off} is to big for image of shape {img_shape}'

    affinities = embedding_to_affinities(embedding, offsets=offsets, affinity_measure=affinity_measure)
    affinities = affinities.contiguous().view((-1, len(offsets)) + emb_shape[-n_img_dims:])

    if percentile is not None:
        affinities -= np.percentile(affinities, percentile)
        affinities[:, :ATT_C] *= -1
    else:
        affinities[:, :ATT_C] *= -1
        affinities[:, :ATT_C] += 1


    result = []
    for aff in affinities:
        dws = cmst.ConstrainedWatershed(np.array(img_shape),
                                        offsets,
                                        ATT_C,
                                        np.array(repulsive_strides, int))
        sorted_edges = np.argsort(aff, axis=None)
        dws.repulsive_ucc_mst_cut(sorted_edges, 0)
        seg = label(dws.get_flat_label_image().reshape(img_shape))
        seg = np.random.permutation(seg.max()+1)[seg]
        result.append(seg)

    result = np.stack(result, axis=-(n_img_dims+1)).reshape(emb_shape[:-n_img_dims-1] + emb_shape[-n_img_dims:])

    if return_affinities:
        return torch.from_numpy(result), affinities
    else:
        return torch.from_numpy(result)


def hdbscan_segmentation(embedding, n_img_dims=None, coord_scales=None,
                         metric='euclidean', min_cluster_size=50, **hdbscan_kwargs):
    assert metric in hdbscan.dist_metrics.METRIC_MAPPING
    if n_img_dims is None:
        # default: assume one embedding image is being passed
        n_img_dims = len(embedding.shape) - 1
    emb_shape = embedding.shape
    img_shape = emb_shape[-n_img_dims:]

    # compute #pixels per image
    n_pixels = 1
    for s in img_shape:
        n_pixels *= s

    # reshape embedding for clustering
    embedding = embedding.contiguous().view(-1, emb_shape[-n_img_dims-1], n_pixels).permute(0, 2, 1)

    # append image coordinates as features if requested
    if coord_scales is not None:
        if not isinstance(coord_scales, collections.Iterable):
            coord_scales = n_img_dims * (coord_scales,)
        assert len(coord_scales) == n_img_dims, f'{coord_scales}, {n_img_dims}'
        coord_axes = []
        for i, scale in enumerate(coord_scales):
            coord_axes.append(np.linspace(0, (img_shape[i]-1) * scale, img_shape[i], dtype=np.float32))
        coord_mesh = np.stack(np.meshgrid(*coord_axes), axis=-1).reshape(n_pixels, -1)[None].repeat(embedding.shape[0], 0)
        embedding = torch.cat([torch.from_numpy(coord_mesh).type(embedding.dtype), embedding], dim=-1)

    # init HDBSCAN clusterer
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, **hdbscan_kwargs)

    # iterate over images in batch
    result = []
    for emb in embedding:
        labels = clusterer.fit_predict(emb).reshape(img_shape)
        result.append(labels)

    result = np.stack(result, axis=0).reshape(emb_shape[:-n_img_dims - 1] + emb_shape[-n_img_dims:])
    return torch.from_numpy(result)


if __name__ == '__main__':
    from embeddingutils.affinities import normalized_cosine_similarity, logistic_similarity
    import matplotlib.pyplot as plt
    emb = torch.ones((2, 11, 11)).float()
    emb[0] = -1
    emb[0, :5, :3] = 1
    emb[1, :5, :3] = -1
    emb[1, 7:9, 4:] = -2
    emb[0, 4:, 7:9] = 2
    emb = emb + torch.randn(emb.shape)/5
    emb = emb[:, None]
    print('embedding shape:', emb.shape)

    for e in emb:
        plt.imshow(e[0])
        plt.show()

    #for aff in embedding_to_affinities(emb, offsets='default-2D', affinity_measure=euclidean_similarity):
    #    print(aff.shape)
    #    print(torch.max(aff), torch.min(aff))

    offsets = ((0, 1), (1, 0), (0, 1), (1, 0))
    segs = (mws_segmentation(emb, offsets=offsets, affinity_measure=logistic_similarity,
                               ATT_C=2, repulsive_strides=(1, 1)))
    for s in segs:
        plt.imshow(s)
        plt.show()

    segs = hdbscan_segmentation(emb, metric='euclidean', min_cluster_size=8, n_img_dims=2, coord_scales=(.1, .1))
    for s in segs:
        plt.imshow(s)
        plt.show()