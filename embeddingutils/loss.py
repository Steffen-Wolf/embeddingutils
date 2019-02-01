from torch.nn.functional import cosine_similarity
import inferno.utils.torch_utils as thu
import torch
import numpy as np
from embeddingutils.affinities import *
from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
import inferno.utils.train_utils as tu
import collections


class ScalarLoggingMixin:
    def __init__(self):
        super(ScalarLoggingMixin, self).__init__()
        self.validation_averages = dict()
        self.current_validation_iteration = None
        self.registered_states = set()

    def save_scalar(self, name, value, trainer=None):
        if trainer is None:
            assert hasattr(self, 'trainer')
            trainer = self.trainer
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().clone()
        # add prefix ('training' or 'validation') to name
        name = self.add_prefix(name, trainer)
        if name not in self.registered_states:
            self.observe_state(name, trainer)
        if trainer.model.training:  # training
            trainer.update_state(name, value)
        else:  # validation
            # check if it is a new validation run
            if self.current_validation_iteration != trainer._last_validated_at_iteration:
                self.current_validation_iteration = trainer._last_validated_at_iteration
                self.validation_averages = dict()

            # check if average meter for name has already been initialized this run
            if name not in self.validation_averages:
                self.validation_averages[name] = tu.AverageMeter()

            self.validation_averages[name].update(value)
            trainer.update_state(name, self.validation_averages[name].avg)

    def observe_state(self, name, trainer):
        assert hasattr(trainer, 'logger')
        logger = trainer.logger
        assert hasattr(logger, 'observe_state')
        time = 'training' if trainer.model.training else 'validation'
        logger.observe_state(name, time)

    def add_prefix(self, name, trainer):
        if trainer.model.training:
            return 'training_' + name
        else:
            return 'validation_' + name


class WeightedLoss(ScalarLoggingMixin, torch.nn.Module):

    def __init__(self, loss_weights=None, trainer=None, loss_names=None):
        super(WeightedLoss, self).__init__()
        self.loss_weights = loss_weights
        if isinstance(loss_weights, collections.Sized) and not isinstance(loss_weights, str):
            self.n_losses = len(loss_weights)
            self.enable_logging = True
        if loss_names is None and loss_weights is not None:
            loss_names = [str(i) for i in range(len(loss_weights))]
        self.loss_names = loss_names
        self.logging_enabled = False
        self.trainer = trainer
        self.validation_averages = None  # Used to keep track of averages during validation

    def forward(self, preds, labels):
        losses = self.get_losses(preds, labels)
        loss = 0
        for i, current in enumerate(losses):
            if self.loss_weights is not None and not isinstance(self.loss_weights, str):
                weight = self.loss_weights[i]
            elif self.loss_weights == 'average':
                weight = 1/len(losses)
            else:
                weight = 1
            loss = loss + weight * current

        self.save_losses(losses)
        return loss.mean()

    def save_losses(self, losses):
        if self.trainer is None:
            return
        losses = [loss.detach().mean() for loss in losses]
        for name, value in zip(self.loss_names, losses):
            self.save_scalar(name, value)

    def __getstate__(self):  # TODO make this nicer
        """Return state values to be pickled."""
        # mydict = dict(self.__dict__)
        # mydict['trainer'] = None
        return {}


class SumLoss(WeightedLoss):
    GRAD_PREFIX = {
        'norm': 'grad-norm_',
        'max': 'grad-max_',
        'mean': 'grad-mean_',
    }

    def __init__(self, losses, grad_stats=None, **super_kwargs):
        super(SumLoss, self).__init__(**super_kwargs)
        assert isinstance(losses, collections.Iterable)
        self.losses = losses
        self.grad_stats = grad_stats
        assert grad_stats is None or all(stat in self.GRAD_PREFIX for stat in grad_stats), \
            f'Supported stats: {list(self.GRAD_PREFIX.keys())}. Got {grad_stats}'
        if self.grad_stats is not None:
            assert self.trainer is not None
        self.hook_handle = None

    def save_grad_stats(self, stat_name, values):
        # divide by loss weights to get unweighted grad norms that are comparable for different weights
        values = [value / w for value, w in zip(values, self.loss_weights)]
        for name, value in zip(self.loss_names, values):
            self.save_scalar(self.GRAD_PREFIX[stat_name] + name, value)

    def hook(self, grad):
        grad = grad.detach()
        if 'norm' in self.grad_stats:
            # calculate mini-batch average of L2 norms of gradients on model prediction
            grad = grad.view(grad.size(0), grad.size(1), -1)
            grad_norms = torch.norm(grad, dim=2).mean(dim=1)
            # divide by loss weights to get unweighted grad norms that are comparable for different weights
            self.save_grad_stats('norm', grad_norms)
        if 'max' in self.grad_stats:
            # calculate the maximum gradient applied on a single pixel.
            grad_max = grad.view(grad.size(0), -1).abs().max(dim=1)[0]
            self.save_grad_stats('max', grad_max)
        if 'mean' in self.grad_stats:
            # calculate the mean gradients.
            grad_mean = grad.view(grad.size(0), -1).mean(dim=1)
            self.save_grad_stats('mean', grad_mean)
        self.hook_handle.remove()

    def get_losses(self, preds, labels):
        result = []
        if self.grad_stats is None or not self.trainer.model.training:
            for loss in self.losses:
                result.append(loss(preds, labels))
        else:
            pred_per_loss = preds[None].repeat(self.n_losses, *((1,) * len(preds.shape)))
            self.hook_handle = pred_per_loss.register_hook(self.hook)
            for pred, loss in zip(pred_per_loss, self.losses):
                result.append(loss(pred, labels))
        return result


class LossSegmentwiseFreeTags(WeightedLoss):
    def __init__(self, margin=0.25, loss_weights=(1, 0.1), ignore_label=None,
                 use_cosine_distance=False, weigh_push_with_size=True, **super_kwargs):
        super_kwargs = dict(loss_weights=loss_weights, loss_names=['push-loss', 'pull-loss'], **super_kwargs)
        super(LossSegmentwiseFreeTags, self).__init__(**super_kwargs)
        self.margin = margin
        self.relu = torch.nn.ReLU()
        self.ignore_label = ignore_label
        self.use_cosine_distance = use_cosine_distance
        self.weigh_push_with_size = weigh_push_with_size
        assert self.ignore_label in [None, 0], \
            'Ignore label other that 0 not implemented'

    def distance_measure(self, x1, x2, dim=1):
        try:
            if self.use_cosine_distance:
                return 0.5 * (1 - cosine_similarity(x1, x2, dim=dim))
            else:
                return torch.abs(x1 - x2).mean(dim=1)  # FIXME: having L1 norm here and squaring it later is awkward
        except:
            import pdb; pdb.set_trace()

    def push_loss(self, centroids, weights=None):
        # return zero if there is only one centroid
        if len(centroids.shape) < 3:
            print("skipping because number of centroids is too low")
            return torch.zeros(1).float().mean().to(centroids.device)
        if centroids.shape[2] <= 1:
            print(f'skipping push: only {centroids.shape[2]} segment')
            return torch.zeros(1).float().mean().to(centroids.device)
        # shape: n_stack * tag_dim * n_segments
        n_stack, tag_dim, n_segments = centroids.shape
        # calculate the distance of all cluster combinations
        distance_matrix = self.distance_measure(centroids[:, :, :, None].repeat(1, 1, 1, n_segments),
                                                centroids[:, :, None, :].repeat(1, 1, n_segments, 1))

        # select vectorized upper triangle of distance matrix
        distance_weights = 1
        if distance_matrix.shape[0] > 1:
            upper_tri_index = torch.arange(1, n_segments*n_segments+1)\
                                .view(n_segments, n_segments)\
                                .triu(diagonal=1).nonzero().transpose(0, 1)
            cluster_distances = distance_matrix[:, upper_tri_index[0], upper_tri_index[1]]
            if weights is not None:
                weight_matrix = weights[None] * weights[:, None]
                distance_weights = weight_matrix[None, upper_tri_index[0], upper_tri_index[1]]
                distance_weights = distance_weights.expand_as(cluster_distances)
        else:
            cluster_distances = distance_matrix[0].triu(diagonal=1)[None]
            if weights is not None:
                distance_weights = weights[None] * weights[:, None]

        return (distance_weights * self.relu(self.margin - cluster_distances)**2).mean()

    def pull_loss(self, preds, masks):
        if preds.shape[0] == 0:
            print('skipping pull because everything is ignored')
            return torch.zeros(1).float().mean().cuda()
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
            assert gt_seg.shape[0] == 1, 'segmentation should have one channel only'

            # get rid of extra spatial dimensions
            gt_seg = gt_seg[0].flatten()
            tags = tags.view(*tags.shape[:2], -1)

            # mask away ignore regions
            mask_ratio = 1
            if self.ignore_label is not None:
                n_segments -= 1
                mask = gt_seg.ne(self.ignore_label)
                mask_ratio = mask.float().mean()
                gt_seg = gt_seg[mask] - 1
                tags = tags[:, :, mask]

            # calculate centroids and segment sizes of the individual segments
            centroids = []
            segment_sizes = []
            for seg_id in range(n_segments):
                seg_mask = gt_seg == seg_id
                segment_sizes.append(seg_mask.float().sum())
                centroids.append(tags[:, :, seg_mask].mean(-1))
            centroids = torch.stack(centroids, dim=-1)
            segment_sizes = torch.stack(segment_sizes)

            # push loss
            push_weights = segment_sizes / tags[0].nelement()**0.5 if self.weigh_push_with_size else None
            push = self.push_loss(centroids, weights=push_weights)
            pushes.append(push)

            # pull loss, scale with mask ratio to avoid gradient spikes
            pixel_wise_centroids = centroids.detach()[:, :, gt_seg]
            pull = mask_ratio * self.pull_loss(tags, pixel_wise_centroids)
            pulls.append(pull)

        return torch.stack(pushes), torch.stack(pulls)


class LossAffinitiesFromEmbedding(WeightedLoss):
    def __init__(self, offsets='default-3D', loss_weights=None, ignore_label=None, margin=0,
                 use_cosine_distance=False, pull_weight=0, push_weight=0, affinity_weight=1,
                 affinities_direct=False, **super_kwargs):
        if callable(offsets):
            self.offset_sampler = offsets
            self.dynamic_offsets = True
            offsets = self.offset_sampler()
            loss_names = None
            loss_weights = 'average'
        else:
            self.offsets = get_offsets(offsets)
            self.dynamic_offsets = False
            if loss_weights is None:
                loss_weights = (1/len(self.offsets),) * len(self.offsets)
            assert len(loss_weights) == len(self.offsets)
            loss_names = ["offset_" + '_'.join(str(o) for o in off) for off in self.offsets]
            print(loss_names)

        super(LossAffinitiesFromEmbedding, self).__init__(
            loss_weights=loss_weights,
            loss_names=loss_names,
            enable_logging=not self.dynamic_offsets,
            **super_kwargs)
        self.ignore_label = ignore_label
        self.use_cosine_distance = use_cosine_distance
        self.ignore_label = ignore_label
        self.margin = margin
        self.push_weight = push_weight
        self.pull_weight = pull_weight
        self.affinity_weight = affinity_weight

        # initialize distance/affinity generating functions
        self.seg_to_aff = EmbeddingToAffinities(offsets=offsets,
                                                affinity_measure=label_equal_similarity,
                                                pass_offset=False)

        if self.affinity_weight is not 0:
            self.emb_to_aff = EmbeddingToAffinities(offsets=offsets,
                                                    affinity_measure=self.affinity_measure,
                                                    pass_offset=True)
            self.aff_loss = SorensenDiceLoss(channelwise=False)

        if self.ignore_label is not None:
            self.seg_to_mask = EmbeddingToAffinities(offsets=offsets,
                                                     affinity_measure=ignore_label_mask_similarity,
                                                     pass_offset=False)

        if self.push_weight != 0 or self.pull_weight != 0:
            self.emb_to_dist = EmbeddingToAffinities(offsets=offsets,
                                                     affinity_measure=self.distance_measure,
                                                     pass_offset=False)

        self.affinities_direct = affinities_direct
        if affinities_direct:
            assert self.pull_weight == self.push_weight == 0 and self.affinity_weight != 0
        self.relu = torch.nn.ReLU()

    def set_offsets(self, offsets):
        self.offsets = offsets
        if self.affinity_weight != 0:
            self.seg_to_aff.offsets = offsets
            self.emb_to_aff.offsets = offsets
        if self.pull_weight != 0 or self.push_weight != 0:
            self.emb_to_dist.offsets = offsets
        if self.ignore_label is not None:
            self.seg_to_mask.offsets = offsets

    def push_loss(self, dists):
        #return (self.relu(self.margin - dists)).mean()
        return (self.relu(self.margin - dists)**2).mean()

    def pull_loss(self, dists):
        #return (dists).mean()
        return (dists**2).mean()

    def affinity_measure(self, x, y, dim, offset):
        if self.use_cosine_distance:
            return self.relu(normalized_cosine_similarity(x, y, dim=dim) * 2 - 1)
        else:
            return logistic_similarity(x, y, dim=dim)  # =offset/100)#

    def distance_measure(self, x, y, dim):
        if self.use_cosine_distance:
            return normalized_cosine_similarity(x, y)
        else:
            return euclidean_distance(x, y, dim)

    def get_losses(self, preds, labels):
        # check if random offsets are used and if yes, sample them
        if self.dynamic_offsets:
            offsets = self.offset_sampler()
            self.set_offsets(offsets)
        else:
            offsets = self.offsets

        if torch.is_tensor(labels):
            gt_segs = labels
        else:
            gt_segs = labels[0]

        if len(preds.shape) == len(labels.shape):  # no intermediate predictions
            preds = preds[:, None]

        with torch.no_grad():
            gt_aff = self.seg_to_aff(gt_segs)

        if self.affinity_weight != 0:
            if not self.affinities_direct:
                pred_aff = self.emb_to_aff(preds)
            else:
                pred_aff = preds

        if self.push_weight != 0 or self.pull_weight != 0:
            pred_dist = self.emb_to_dist(preds)

        if self.ignore_label is not None:
            masks = self.seg_to_mask(gt_segs)
        else:
            masks = torch.ones_like(gt_segs).byte()

        losses_per_offset = []
        for j, offset in enumerate(offsets):  # iterate over offsets
            loss_this_offset = []
            for i in range(preds.shape[1]):  # iterate over intermediate outputs
                current_loss = torch.tensor(0).float().to(preds.device)
                if self.affinity_weight != 0:
                    ind = masks[:, j]
                    if ind.any():
                        m = ind.float().mean()
                        current_loss += -(1-m) + m * self.affinity_weight * \
                                        self.aff_loss(1 - pred_aff[:, i, j][ind], 1 - gt_aff[:, j][ind])
                if self.push_weight != 0:
                    ind = masks[:, j] * (gt_aff[:, j]).byte() != 0
                    if ind.any():
                        m = ind.float().mean()
                        current_loss += m * self.push_weight * \
                                        self.push_loss(pred_dist[:, i, j][ind])
                if self.pull_weight != 0:
                    ind = masks[:, j] * (1 - gt_aff[:, j]).byte() != 0
                    if ind.any():
                        m = ind.float().mean()
                        current_loss += m * self.pull_weight * \
                                        self.pull_loss(pred_dist[:, i, j][ind])
                loss_this_offset.append(current_loss)
            losses_per_offset.append(torch.stack(loss_this_offset, dim=0))

        return losses_per_offset
