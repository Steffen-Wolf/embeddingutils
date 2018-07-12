import torch
from torch.nn.functional import cosine_similarity
from skimage.measure import label
import constrained_mst as cmst
import numpy as np
import hdbscan
from embeddingutils.affinities import embedding_to_affinities, get_offsets, euclidean_similarity


def mws_segmentation(embedding, offsets='default-3D', affinity_measure=euclidean_similarity,
                     ATT_C=3, repulsive_strides=None, return_affinities=False):

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
    affinities[:, :ATT_C] *= -1
    affinities[:, :ATT_C] += 1.


    result = []
    print(affinities.shape)
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
        return torch.from_numpy(result), torch.from_numpy(affinities)
    else:
        return torch.from_numpy(result)


def hdbscan_segmentation(embedding, metric='euclidean', n_img_dims=None, min_cluster_size=50):
    assert metric in hdbscan.dist_metrics.METRIC_MAPPING
    if n_img_dims is None:
        n_img_dims = len(embedding.shape) - 1
    emb_shape = embedding.shape
    img_shape = emb_shape[-n_img_dims:]

    n_pixels = 1
    for s in img_shape:
        n_pixels *= s
    print(n_pixels)
    embedding = embedding.contiguous().view(-1, emb_shape[-n_img_dims-1], n_pixels).permute(0, 2, 1)
    print(embedding.shape)

    result = []
    for emb in embedding:
        #np.concatenate([tags[i], xx[None], yy[None]] TODO
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
        labels = clusterer.fit_predict(emb).reshape(img_shape)
        result.append(labels)
        print(np.max(labels))

    result = np.stack(result, axis=0).reshape(emb_shape[:-n_img_dims - 1] + emb_shape[-n_img_dims:])
    return torch.from_numpy(result)


if __name__ == '__main__':
    from embeddingutils.affinities import normalized_cosine_similarity, euclidean_similarity
    import matplotlib.pyplot as plt
    emb = torch.ones((2, 11, 11)).float()
    emb[0] = -1
    emb[0, :5, :3] = 1
    emb[1, :5, :3] = -1
    emb[1, 7:9, 4:] = -2
    emb[0, 4:, 7:9] = 2
    emb = emb + torch.randn(emb.shape)/5
    emb = emb[:, None]
    print(emb.shape)

    for e in emb:
        plt.imshow(e[0])
        plt.show()

    #for aff in embedding_to_affinities(emb, offsets='default-2D', affinity_measure=euclidean_similarity):
    #    print(aff.shape)
    #    print(torch.max(aff), torch.min(aff))

    offsets = ((0, 1), (1, 0), (0, 1), (1, 0))
    segs = (mws_segmentation(emb, offsets=offsets, affinity_measure=euclidean_similarity,
                               ATT_C=2, repulsive_strides=(1, 1)))
    for s in segs:
        plt.imshow(s)
        plt.show()

    segs = hdbscan_segmentation(emb, metric='euclidean', min_cluster_size=8, n_img_dims=2)
    for s in segs:
        plt.imshow(s)
        plt.show()