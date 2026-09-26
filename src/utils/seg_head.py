import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentSegmentationHead3D(nn.Module):

    def __init__(self, latent_channels=1, hidden_channels=64, num_classes=1, upsample_factor=4):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, num_classes, 1),
        )

    def forward(self, z_0_pred):
        seg = self.decoder(z_0_pred)
        if self.upsample_factor > 1:
            seg = F.interpolate(seg, scale_factor=self.upsample_factor, mode="trilinear", align_corners=False)
        return seg


class SegmentationLoss(nn.Module):

    def __init__(self, loss_weight=0.25, noise_threshold=0.5, smooth=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.noise_threshold = noise_threshold
        self.smooth = smooth

    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        p = pred.flatten(1)
        t = target.flatten(1)
        inter = (p * t).sum(1)
        return 1 - (2 * inter + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)

    def forward(self, seg_pred, seg_target, t, T):
        if seg_target.ndim == seg_pred.ndim - 1:
            seg_target = seg_target.unsqueeze(1)
        if seg_pred.shape[2:] != seg_target.shape[2:]:
            seg_target = F.interpolate(seg_target.float(), size=seg_pred.shape[2:], mode="nearest")

        ratio = t.float() / T
        w = ((ratio < self.noise_threshold).float() * (1 - ratio / self.noise_threshold).clamp(0, 1))
        if w.sum() < 1e-6:
            return torch.tensor(0.0, device=seg_pred.device, requires_grad=True)

        bce = F.binary_cross_entropy_with_logits(seg_pred, seg_target.float(), reduction="none").mean(dim=list(range(1, seg_pred.ndim)))
        dice = self.dice_loss(seg_pred, seg_target)
        loss = ((bce + dice) * w).sum() / w.sum().clamp(min=1)
        return self.loss_weight * loss
