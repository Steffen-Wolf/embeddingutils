import torch
from inferno.io.transform import Transform
import torch
from embeddingutils.affinities import embedding_to_affinities, label_equal_similarity_with_mask_le

class Segmentation2AffinitiesWithPadding(Transform):

    def __init__(self, offsets,
                 segmentation_to_binary=True,
                 ignore_label=-1,
                 retain_segmentation=True,
                 **super_kwargs):
        self.offsets = offsets
        self.retain_segmentation = retain_segmentation
        self.segmentation_to_binary = segmentation_to_binary

        super().__init__(**super_kwargs)

    def tensor_function(self, tensor):

        tensor = torch.from_numpy(tensor[None])

        out = embedding_to_affinities(tensor,
                                      offsets=self.offsets,
                                      affinity_measure=label_equal_similarity_with_mask_le)

        if self.segmentation_to_binary:
            out = torch.cat(((tensor > 0).float(), out))

        if self.retain_segmentation:
            out = torch.cat((tensor.float(), out))

        return out.numpy()