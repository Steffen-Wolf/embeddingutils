import torch
import numpy as np
from sklearn.decomposition import PCA

def pca(embedding, output_dimensions=3):
    _pca = PCA(n_components=output_dimensions)
    # reshape embedding
    output_shape = list(embedding.shape)
    output_shape[1] = output_dimensions
    flat_embedding = embedding.cpu().numpy().reshape(embedding.shape[0], embedding.shape[1], -1)
    flat_embedding = flat_embedding.transpose((0, 2, 1))

    pca_output = []
    for flat_image in flat_embedding:
        # fit PCA to array of shape (n_samples, n_features) and apply to input data
        pca_output.append(_pca.fit_transform(flat_image))

    return torch.stack([torch.from_numpy(x.T) for x in pca_output]).reshape(output_shape)

if __name__ == '__main__':
    print(show_pca(torch.rand(2, 64, 20, 100, 100)).shape)