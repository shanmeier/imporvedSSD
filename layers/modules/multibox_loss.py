import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from .box_utils import match_gious, bbox_overlaps_giou, decode
import numpy as np

GPU = False
if torch.cuda.is_available():
    GPU = True


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap,
                 encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        # print(self.gamma, self.alpha)

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        class_mask = inputs.data.new(N, C).fill_(0)
        # class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class GiouLoss(nn.Module):
    def __init__(self, pred_mode='Center', size_sum=True, variances=None):
        super(GiouLoss, self).__init__()
        self.size_sum = size_sum
        self.pred_mode = pred_mode
        self.variances = variances

    def forward(self, loc_p, loc_t, prior_data):
        num = loc_p.shape[0]

        if self.pred_mode == 'Center':
            decoded_boxes = decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p
        # loss = torch.tensor([1.0])
        gious = 1.0 - bbox_overlaps_giou(decoded_boxes, loc_t)

        loss = torch.sum(gious)

        if self.size_sum:
            loss = loss
        else:
            loss = loss / num
        return 5 * loss


def balanced_l1_loss(input, target, beta=1.0, alpha=0.5, gamma=1.5):
    assert beta > 0
    assert input.size() == target.size() and target.numel() > 0

    diff = torch.abs(input - target)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    return loss.sum()


class MultiBoxLoss_Focal_Giou(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, cfg, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, loss_c="CrossEntropy", loss_r='SmoothL1'):
        super(MultiBoxLoss_Focal_Giou, self).__init__()
        self.use_gpu = use_gpu

        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.focalloss = FocalLoss(self.num_classes, gamma=2, size_average=False)
        self.gious = GiouLoss(pred_mode='Center', size_sum=True, variances=self.variance)
        self.loss_c = loss_c
        self.loss_r = loss_r
        if self.loss_r != 'SmoothL1' or self.loss_r != 'Giou' or self.loss_r !='BanlancedL1':
            assert Exception("THe loss_r is Error, loss name must be SmoothL1 or Giou or BanlancedL1")
        elif self.loss_c != 'CrossEntropy' or self.loss_c != 'FocalLoss':
            assert Exception("THe loss_c is Error, loss name must be CrossEntropy or FocalLoss")
        # elif self.loss_r == 'Giou':
        #     match_gious(self.threshold, truths, defaults, self.variance, labels,
        #                 loc_t, conf_t, idx)

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions
        priors = priors
        num = loc_data.size(0)

        priors = priors[:loc_data.size(1), :]

        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)

        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            if self.loss_r == 'SmoothL1' or self.loss_r == 'BanlancedL1':
                match(self.threshold, truths, defaults, self.variance, labels,
                      loc_t, conf_t, idx)
            elif self.loss_r == 'Giou':
                match_gious(self.threshold, truths, defaults, self.variance, labels,
                            loc_t, conf_t, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        # loc_t = Variable(loc_t, requires_grad=True)
        # conf_t = Variable(conf_t, requires_grad=True)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        if self.loss_r == 'SmoothL1':
            loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        elif self.loss_r == 'BanlancedL1':
            loss_l = balanced_l1_loss(loc_p, loc_t)
        elif self.loss_r == 'Giou':
            giou_priors = priors.data.unsqueeze(0).expand_as(loc_data)
            loss_l = self.gious(loc_p, loc_t, giou_priors[pos_idx].view(-1, 4))
        # Compute max conf across batch for hard negative mining
        if self.loss_c == "CrossEntropy":

            batch_conf = conf_data.view(-1, self.num_classes)
            loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

            # Hard Negative Mining
            loss_c = loss_c.view(num, -1)
            loss_c[pos] = 0
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)
            num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
            neg = idx_rank < num_neg.expand_as(idx_rank)

            # Confidence Loss Including Positive and Negative Examples
            pos_idx = pos.unsqueeze(2).expand_as(conf_data)
            neg_idx = neg.unsqueeze(2).expand_as(conf_data)
            conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            targets_weighted = conf_t[(pos + neg).gt(0)]
            loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        elif self.loss_c == "FocalLoss":
            batch_conf = conf_data.view(-1, self.num_classes)
            loss_c = self.focalloss(batch_conf, conf_t)

        N = num_pos.data.sum().double()
        loss_l = loss_l.double()
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
