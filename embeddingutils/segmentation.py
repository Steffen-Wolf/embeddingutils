from torch.nn.functional import cosine_similarity
from skimage.measure import label
import constrained_mst as cmst
import numpy as np

def mws_segmentation(embedding, distance_measure=cosine_similarity):
    ATT_C = 3
    # offsets = np.array([[0, -1, 0], [0, 0, -1], [0, -9, 0], [0, 0, -9], [0, -9, -9], [0, 9, -9],\
    # [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4], [0, -27, 0], [0, 0, -27]], np.int) 
    offsets = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1], 
                        [0, -9, 0], [0, 0, -9], [0, -9, -9], [0, 9, -9], 
                        [0, -9, 4], [0, -4, -9], [0, 4, -9], [0, 9, -4], 
                        [-1, -1, 0], [-1, 0, -1],
                        [0, -27, 0], [0, 0, -27]], int)

    affinities = []
    for o in offsets:
        padding = []
        # reverse offset order, because pytorch padding work backwards!
        for s in reversed(o):
            padding.append(int(-s))
            padding.append(int(s))
        comp_embedding = torch.nn.functional.pad(embedding[None], tuple(padding))[0]
        affinities.append(distance_measure(embedding, comp_embedding))

    affinities = np.stack(affinities)
    affinities[:ATT_C] *= -1
    affinities[:ATT_C] += 1.
    img_shape = affinities.shape[1:]

    # load image and restore original image statistics
    dws =cmst.ConstrainedWatershed(np.array(img_shape),
                            offsets,
                            ATT_C,
                            np.array([1, 8, 8]))
    sorted_edges = np.argsort(affinities, axis=None)
    dws.repulsive_ucc_mst_cut(sorted_edges, 0)

    seg = label(dws.get_flat_label_image().reshape(img_shape))
    seg = np.random.permutation(seg.max()+1)[seg]

    return torch.from_numpy(seg), torch.from_numpy(affinities)
    