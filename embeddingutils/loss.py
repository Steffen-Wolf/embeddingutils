from torch.nn.functional import cosine_similarity
import inferno.utils.torch_utils as thu
import torch
import numpy as np
from embeddingutils.affinities import get_offsets, offset_slice, EmbeddingToAffinities
from embeddingutils.affinities import logistic_similarity, squared_euclidean_distance, normalized_cosine_similarity, label_equal_similarity
from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss


class WeightedLoss(torch.nn.Module):

    def __init__(self, loss_weights, trainer=None, loss_names=None,
                 split_by_stages=False, enable_logging=True):
        super(WeightedLoss, self).__init__()
        self.loss_weights = loss_weights
        self.trainer = trainer
        if loss_names is None:
            loss_names = [str(i) for i in range(len(loss_weights))]
        self.loss_names = loss_names
        self.n_losses = len(loss_weights)
        self.logging_enabled = False
        self.enable_logging = enable_logging
        assert not split_by_stages
        self.split_by_stages = split_by_stages

    def forward(self, preds, labels):
        losses = self.get_losses(preds, labels)
        loss = 0
        for i, current in enumerate(losses):
            loss = loss + self.loss_weights[i] * current
        self.save_losses(losses)
        return loss.mean()

    def save_losses(self, losses):
        if self.trainer is None:
            return
        if not self.logging_enabled:
            if self.enable_logging:
                self.register_logger(self.trainer.logger)
            else:
                return
        losses = [loss.detach().mean() for loss in losses]
        for i, current in enumerate(losses):
            self.trainer.update_state(self.get_loss_name(
                i), thu.unwrap(current))

    def register_logger(self, logger):
        for i in range(self.n_losses):
            logger.observe_state(self.get_loss_name(
                i, training=True), 'training')
            logger.observe_state(self.get_loss_name(
                i, training=False), 'validation')

        self.logging_enabled = True

    def get_loss_name(self, i, training=None):
        if training is None:
            assert self.trainer is not None
            assert self.trainer.model_is_defined
            training = self.trainer.model.training
        if training:
            return 'training_' + self.loss_names[i]
        else:
            return 'validation_' + self.loss_names[i]

    def __getstate__(self):  # TODO make this nicer
        """Return state values to be pickled."""
        # mydict = dict(self.__dict__)
        # mydict['trainer'] = None
        return {}


class LossSegmentwiseFreeTags(WeightedLoss):  # TODO: requires_grad = False on centroids
    def __init__(self, margin=0.25, loss_weights=(1, 0.1), ignore_label=None,
                 use_cosine_distance=False, **super_kwargs):
        super(LossSegmentwiseFreeTags, self).__init__(
            loss_weights=loss_weights,
            loss_names=['push-loss', 'pull-loss'],
            **super_kwargs)
        self.margin = margin
        self.relu = torch.nn.ReLU()
        self.ignore_label = ignore_label
        self.use_cosine_distance = use_cosine_distance
        assert self.ignore_label in [
            None, 0], 'Ignore label other that 0 not implemented'

    def distance_measure(self, x1, x2, dim=1):
        if self.use_cosine_distance:
            try:
                return 0.5 * (1 - cosine_similarity(x1, x2, dim=dim))
            except:
                import pdb; pdb.set_trace()
        return torch.abs(x1 - x2).mean(dim=1)

    def push_loss(self, centroids):
        # return zero if there is only one centroid
        if len(centroids.shape) < 3:
            print("skipping because number of centroids is too low")
            return 0.
        # shape: n_stack * tag_dim * n_segments
        n_stack, tag_dim, n_segments = centroids.shape
        # calculate the distance of all cluster combinations
        distance_matrix = self.distance_measure(centroids[:, :, :, None].repeat(1, 1, 1, n_segments),
                                                centroids[:, :, None, :].repeat(1, 1, n_segments, 1))

        # select vectorized upper triangle of distance matrix
        upper_tri_index = torch.arange(1, n_segments*n_segments+1)\
                            .view(n_segments, n_segments)\
                            .triu(diagonal=1).nonzero().transpose(0, 1)
        cluster_distances = distance_matrix[:, upper_tri_index[0], upper_tri_index[1]]

        return (self.relu(self.margin - cluster_distances)**2).mean()

    def pull_loss(self, preds, masks):  # TODO weigh with segment size?
        return (self.distance_measure(preds, masks)**2).mean()

    def get_losses(self, preds, labels):

        if torch.is_tensor(labels):
            gt_segs = labels
        else:
            gt_segs = labels[0]

        if len(preds.shape) == len(labels.shape):  # no intermediate predictions
            preds = preds[:, None]

        pulls = []
        pushes = []
        for gt_seg, tags in zip(gt_segs, preds):  # iterate over minibatch
            n_segments = torch.max(gt_seg).int().item() + 1
            gt_seg = gt_seg.long()
            assert gt_seg.shape[
                0] == 1, 'segmentation should have one channel only'
            gt_seg = gt_seg[0]

            centroids = []
            for seg_id in range(n_segments):
                if self.ignore_label == seg_id:
                    centroids.append(tags.new(np.zeros((tags.shape[0], tags.shape[1]))))
                    continue
                centroids.append(tags[:, :, gt_seg == seg_id].mean(-1))

            centroids = torch.stack(centroids, dim=0)
            # move fist axis to the last position in n-dim
            centroids = centroids.permute(*(list(range(1, len(centroids.shape))) + [0]))

            if self.ignore_label == 0:
                push = self.push_loss(centroids[..., 1:])
            elif self.ignore_label is None:
                push = self.push_loss(centroids)
            else:
                raise NotImplemented("ignore_label other than 0 not implemented")

            pixelwise_centroids = centroids[:, :, gt_seg].contiguous()

            if self.ignore_label is not None:
                pull = self.pull_loss(tags[:, :, gt_seg.ne(self.ignore_label)],
                                      pixelwise_centroids[:, :, gt_seg.ne(self.ignore_label)])
            else:
                pull = self.pull_loss(tags, pixelwise_centroids)
            pushes.append(push)
            pulls.append(pull)
        # FIXME should be averaged in minibatch and split in output channels
        return torch.stack(pushes), torch.stack(pulls)


class LossAffinitiesFromEmbedding(WeightedLoss):
    def __init__(self, offsets='default-3D', loss_weights=None, ignore_label=None,
                 use_cosine_distance=False, **super_kwargs):
        self.offsets = get_offsets(offsets)
        if loss_weights is None:
            loss_weights = (1,) * len(self.offsets)
        assert len(loss_weights) == len(self.offsets)
        loss_names = ["offset_" + '|'.join(str(o) for o in off) for off in self.offsets]
        print(loss_names)
        super(LossAffinitiesFromEmbedding, self).__init__(
            loss_weights=loss_weights,
            loss_names=loss_names,
            **super_kwargs)
        self.ignore_label = ignore_label
        self.use_cosine_distance = use_cosine_distance
        assert not self.use_cosine_distance
        assert self.ignore_label in [
            None,], 'Ignore label other that 0 not implemented'

        self.seg_to_aff = EmbeddingToAffinities(offsets=offsets,
                                                affinity_measure=label_equal_similarity,
                                                pass_offset=False)
        self.emb_to_aff = EmbeddingToAffinities(offsets=offsets,
                                                affinity_measure=self.affinity_measure,
                                                pass_offset=True)

        self.aff_loss = SorensenDiceLoss(channelwise=False)

    def affinity_measure(self, x, y, dim, offset):
        return logistic_similarity(x, y, dim=dim, offset=offset/100)

    def get_losses(self, preds, labels):
        if torch.is_tensor(labels):
            gt_segs = labels
        else:
            gt_segs = labels[0]

        if len(preds.shape) == len(labels.shape):  # no intermediate predictions
            preds = preds[:, None]

        assert not torch.isnan(preds).any()
        print(torch.max(preds).item(), torch.min(preds).item())
        gt_aff = self.seg_to_aff(gt_segs)
        pred_aff = self.emb_to_aff(preds)

        assert not torch.isnan(pred_aff).any()

        # FIXME: ignore label

        losses_per_offset = []
        for j, offset in enumerate(self.offsets):  # iterate over offsets
            loss_this_offset = []
            for i in range(pred_aff.shape[1]):  # iterate over intermediate outputs
                if self.ignore_label is None:
                    s = offset_slice(offset, reverse=True, extra_dims=1)
                    loss = self.aff_loss(1 - pred_aff[:, i, j][s], 1 - gt_aff[:, j][s])
                    loss_this_offset.append(loss)
                else:
                    assert False
            losses_per_offset.append(torch.stack(loss_this_offset, dim=0))\

        return losses_per_offset
