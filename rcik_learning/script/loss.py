import torch
import torch.nn as nn


class RankingLoss(nn.Module):
    def __init__(self, margin):
        super(RankingLoss, self).__init__()
        self.margin = margin

        if self.margin is None:
            self.margin_loss = nn.SoftMarginLoss()
        else:
            self.margin_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, preds, y):
        assert len(preds) % 2 == 0, 'the batch size is not even.'

        preds_i = preds[:preds.size(0) // 2]
        preds_j = preds[preds.size(0) // 2:]
        y_i = y[:y.size(0) // 2]
        y_j = y[y.size(0) // 2:]
        labels = torch.sign(y_i - y_j)

        if self.margin is None:
            return self.margin_loss(preds_i-preds_j, labels)
        else:
            return self.margin_loss(preds_i, preds_j, labels)


class DenseRankingLoss(nn.Module):
    def __init__(self, margin = None):
        super(DenseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, preds, y):
        n = preds.size(0)
        gt_diff_mat = y.expand(n, n) - y.expand(n, n).t()
        gt_small_mat = (gt_diff_mat < 0).float()  # each row -> 1: smaller than row_idx, 0: same or bigger than row_idx
        diff_mat = preds.expand(n, n) - preds.expand(n, n).t()
        big_mat = torch.clamp(diff_mat, min=0)
        error = big_mat * gt_small_mat

        if self.margin is not None:
            error -= self.margin
            error = nn.functional.relu(error)

        return error.sum()/n


class RankingAccuracyLoss(nn.Module):
    def __init__(self):
        super(RankingAccuracyLoss, self).__init__()

    def forward(self, preds, y, size_average=True):
        n = preds.size(0)
        gt_diff_mat = y.expand(n, n) - y.expand(n, n).t()
        gt_comparison = torch.sign(gt_diff_mat)
        pred_diff_mat = preds.expand(n, n) - preds.expand(n, n).t()
        pred_comparison = torch.sign(pred_diff_mat)
        acc_mat = (gt_comparison == pred_comparison) + (gt_comparison == 0)
        m = (gt_comparison == 0).sum().float()
        acc_mat = torch.sign(acc_mat).float()
        acc = (acc_mat.sum() - m) / (n * n - m)

        return 1.0 - acc
