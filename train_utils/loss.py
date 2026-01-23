from torch.nn import functional as F

import torch
import torch.nn as nn
import numpy as np


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):

    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    dice_target = dice_target.permute(0, 3, 1, 2)

    return dice_target


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):

    d = 0.0
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:

            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        intersection = torch.dot(x_i, t_i)
        union = torch.sum(x_i) + torch.sum(t_i)
        if union == 0:
            union = 2 * intersection

        d += (2 * intersection + epsilon) / (union + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):

    dice = 0.
    num_classes = x.shape[1]
    for channel in range(num_classes):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):

    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)


def segment_criterion(outputs, target, loss_weight=None,
              num_classes: int = 2, dice: bool = True, ignore_index: int = -100):

    if num_classes == 2:

        losses = {}
        for name, x in outputs.items():

            if name == 'cls' or name == 'out_three' or name == 'aux_three':
                continue
            else:
                ce_loss = F.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
                if dice is True:
                    dice_target = build_target(target, num_classes, ignore_index)
                    dice_aux_loss = dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
                    losses[name] = ce_loss + dice_aux_loss

                losses[name] = ce_loss

        return losses['out_two'] + 0.5 * losses['aux_two']

    elif num_classes == 3:
        losses = {}
        for name, x in outputs.items():

            if name == 'cls' or name == 'out_two' or name == 'aux_two':
                continue
            else:
                ce_loss = F.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
                if dice is True:
                    dice_target = build_target(target, num_classes, ignore_index)
                    dice_aux_loss = dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
                    losses[name] = ce_loss + dice_aux_loss

                losses[name] = ce_loss

        if len(losses) == 1:
            return losses['out']

        return losses['out'] + 0.5 * losses['aux']


def classify_criterion(pred, label, loss_weight):
    loss_function = torch.nn.CrossEntropyLoss(weight=loss_weight)
    loss = loss_function(pred, label)
    return loss


def consisteny_loss(x, preds, labels, device):
    batch_size = x.shape[0]
    total_loss = 0

    for b in range(batch_size):
        x_0 = x[b].unsqueeze(0)
        pred = preds[b].reshape(1, -1)
        label = labels[b].reshape(1)

        x_softmax = torch.softmax(x_0, dim=1)
        ob_tensor = x_softmax[..., 1, :, :].to(device)
        seg_as_cls = torch.max(ob_tensor).item()

        unhealthy_seg_tensor = torch.tensor(seg_as_cls).reshape(1).to(device)
        healthy_seg_tensor = torch.tensor(1 - seg_as_cls).reshape(1).to(device)
        cls_tensor = torch.concat([healthy_seg_tensor, unhealthy_seg_tensor]).reshape(1, -1).to(device)

        nllloss = nn.NLLLoss()
        cls_tensor = torch.log(cls_tensor)
        seg_loss = nllloss(cls_tensor, label)
        total_loss += seg_loss

    return total_loss / batch_size


def consisteny_loss_three(x, preds, labels, device):
    batch_size = x.shape[0]
    total_loss = 0

    for b in range(batch_size):
        x_0 = x[b].unsqueeze(0)
        pred = preds[b].reshape(1, -1)
        label = labels[b].reshape(1)

        x_softmax = torch.softmax(x_0, dim=1)
        leaf_tensor = x_softmax[..., 1, :, :].to(device)
        seg_as_cls = torch.max(leaf_tensor).item()

        unhealthy_seg_tensor = torch.tensor(seg_as_cls).reshape(1).to(device)
        healthy_seg_tensor = torch.tensor(1 - seg_as_cls).reshape(1).to(device)
        cls_tensor = torch.concat([healthy_seg_tensor, unhealthy_seg_tensor]).reshape(1, -1).to(device)

        nllloss = nn.NLLLoss()
        cls_tensor = torch.log(cls_tensor)
        seg_loss = nllloss(cls_tensor, label)
        total_loss += seg_loss

        leaf_tensor = x_softmax[..., 2, :, :].to(device)
        seg_as_cls = torch.max(leaf_tensor).item()

        unhealthy_seg_tensor = torch.tensor(seg_as_cls).reshape(1).to(device)
        healthy_seg_tensor = torch.tensor(1 - seg_as_cls).reshape(1).to(device)
        cls_tensor = torch.concat([healthy_seg_tensor, unhealthy_seg_tensor]).reshape(1, -1).to(device)

        nllloss = nn.NLLLoss()
        cls_tensor = torch.log(cls_tensor)
        seg_loss = nllloss(cls_tensor, label)
        total_loss += seg_loss

    return total_loss / batch_size