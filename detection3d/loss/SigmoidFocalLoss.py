import torch
from torch import nn
import torch.nn.functional as F


class SigmoidFocalLossMultiChannel(nn.Module):
    """
    Multi-channel sigmoid focal loss for ternary masks {-1,0,1}.

    Each channel is an independent binary task:
      target = 1 -> positive
      target = 0 -> negative
      target = -1 -> ignore (no loss computed)

    input  : logits [B,C,D,H,W] OR [N,C]
    target : same shape as input (ternary mask)
    """

    def __init__(self, class_num, alpha=0.25, gamma=2.0, size_average=True, eps=1e-6):
        super().__init__()
        self.class_num = class_num
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.size_average = size_average
        self.eps = eps

    def forward(self, input, target):
        assert input.shape == target.shape, f"Shape mismatch: {input.shape} vs {target.shape}"
        assert input.dim() in (2, 4, 5), f"Expected input dim 2/4/5, got {input.dim()}"

        # --- reshape to [N, C] like your old focal loss ---
        if input.dim() == 4:
            # [B,C,H,W] -> [N,C]
            input = input.permute(0, 2, 3, 1).contiguous().view(-1, self.class_num)
            target = target.permute(0, 2, 3, 1).contiguous().view(-1, self.class_num)

        elif input.dim() == 5:
            # [B,C,D,H,W] -> [N,C]
            input = input.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.class_num)
            target = target.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.class_num)

        # input, target now [N,C]
        target = target.float()

        # --- external ignore mask ---
        valid_mask = (target != -1).float()         # [N,C]
        target_bin = (target == 1).float()          # [N,C]

        # --- sigmoid focal loss per element ---
        p = torch.sigmoid(input).clamp(self.eps, 1.0 - self.eps)

        # BCE with logits (stable)
        ce_loss = F.binary_cross_entropy_with_logits(input, target_bin, reduction="none")  # [N,C]

        # p_t
        p_t = target_bin * p + (1 - target_bin) * (1 - p)

        # alpha_t
        alpha_t = target_bin * self.alpha + (1 - target_bin) * (1 - self.alpha)

        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce_loss  # [N,C]

        # apply ignore
        loss = loss * valid_mask

        # normalize by valid elements only
        denom = valid_mask.sum().clamp_min(1.0)

        if self.size_average:
            return loss.sum() / denom
        else:
            return loss.sum()
