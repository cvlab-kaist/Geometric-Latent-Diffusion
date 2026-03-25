from functools import cache

import torch
from einops import reduce
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor


@torch.no_grad()
def compute_psnr(
    ground_truth: Tensor,  # (batch, ...)
    predicted: Tensor,     # (batch, ...)
) -> Tensor:              # (batch,)
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b ... -> b", "mean")
    return -10 * mse.log10()


@torch.no_grad()
def compute_mse(
    ground_truth: Tensor,  # (batch, ...)
    predicted: Tensor,     # (batch, ...)
) -> Tensor:              # (batch,)
    """Compute per-sample MSE for latent-space evaluation."""
    return reduce((ground_truth - predicted) ** 2, "b ... -> b", "mean")


@torch.no_grad()
def compute_cosine_similarity(
    ground_truth: Tensor,  # (batch, ...)
    predicted: Tensor,     # (batch, ...)
) -> Tensor:              # (batch,)
    """
    Compute per-sample cosine similarity for latent-space evaluation.
    Returns values in [-1, 1], where 1 means perfect alignment.
    """
    # Flatten all dimensions except batch
    gt_flat = ground_truth.flatten(start_dim=1)  # (batch, features)
    pred_flat = predicted.flatten(start_dim=1)   # (batch, features)
    
    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(gt_flat, pred_flat, dim=1)
    return cos_sim


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Tensor,  # (batch, channel, height, width)
    predicted: Tensor,     # (batch, channel, height, width)
) -> Tensor:              # (batch,)
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Tensor,  # (batch, channel, height, width)
    predicted: Tensor,     # (batch, channel, height, width)
) -> Tensor:              # (batch,)
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


# ==============================================================================
# Depth Evaluation Metrics
# ==============================================================================

@torch.no_grad()
def compute_abs_rel(
    predicted: Tensor,     # (batch, 1, height, width) or (batch, height, width)
    ground_truth: Tensor,  # (batch, 1, height, width) or (batch, height, width)
    valid_mask: Tensor = None,  # Optional mask for valid pixels
) -> Tensor:              # (batch,)
    """
    Compute Absolute Relative Error (AbsRel) for depth evaluation.
    AbsRel = mean(|pred - gt| / gt) over valid pixels.
    
    Args:
        predicted: Predicted depth map
        ground_truth: Ground truth depth map (pseudo-GT from DA3)
        valid_mask: Optional mask, if None uses gt > 0 as valid
    """
    # Ensure same shape
    if predicted.dim() == 4 and predicted.shape[1] == 1:
        predicted = predicted.squeeze(1)
    if ground_truth.dim() == 4 and ground_truth.shape[1] == 1:
        ground_truth = ground_truth.squeeze(1)
    
    if valid_mask is None:
        valid_mask = ground_truth > 0
    
    abs_rel_per_sample = []
    for pred, gt, mask in zip(predicted, ground_truth, valid_mask):
        if mask.sum() == 0:
            abs_rel_per_sample.append(torch.tensor(0.0, device=pred.device))
            continue
        abs_rel = (torch.abs(pred[mask] - gt[mask]) / gt[mask]).mean()
        abs_rel_per_sample.append(abs_rel)
    
    return torch.stack(abs_rel_per_sample)


@torch.no_grad()
def compute_depth_rmse(
    predicted: Tensor,     # (batch, 1, height, width) or (batch, height, width)
    ground_truth: Tensor,  # (batch, 1, height, width) or (batch, height, width)
    valid_mask: Tensor = None,
) -> Tensor:              # (batch,)
    """
    Compute Root Mean Squared Error (RMSE) for depth evaluation.
    RMSE = sqrt(mean((pred - gt)^2)) over valid pixels.
    """
    if predicted.dim() == 4 and predicted.shape[1] == 1:
        predicted = predicted.squeeze(1)
    if ground_truth.dim() == 4 and ground_truth.shape[1] == 1:
        ground_truth = ground_truth.squeeze(1)
    
    if valid_mask is None:
        valid_mask = ground_truth > 0
    
    rmse_per_sample = []
    for pred, gt, mask in zip(predicted, ground_truth, valid_mask):
        if mask.sum() == 0:
            rmse_per_sample.append(torch.tensor(0.0, device=pred.device))
            continue
        rmse = torch.sqrt(((pred[mask] - gt[mask]) ** 2).mean())
        rmse_per_sample.append(rmse)
    
    return torch.stack(rmse_per_sample)


@torch.no_grad()
def compute_delta(
    predicted: Tensor,     # (batch, 1, height, width) or (batch, height, width)
    ground_truth: Tensor,  # (batch, 1, height, width) or (batch, height, width)
    threshold: float = 1.25,
    valid_mask: Tensor = None,
) -> Tensor:              # (batch,)
    """
    Compute δ accuracy for depth evaluation.
    δ < threshold: percentage of pixels where max(pred/gt, gt/pred) < threshold.
    
    Common thresholds: 1.25 (δ₁), 1.25² (δ₂), 1.25³ (δ₃)
    """
    if predicted.dim() == 4 and predicted.shape[1] == 1:
        predicted = predicted.squeeze(1)
    if ground_truth.dim() == 4 and ground_truth.shape[1] == 1:
        ground_truth = ground_truth.squeeze(1)
    
    if valid_mask is None:
        valid_mask = ground_truth > 0
    
    delta_per_sample = []
    for pred, gt, mask in zip(predicted, ground_truth, valid_mask):
        if mask.sum() == 0:
            delta_per_sample.append(torch.tensor(1.0, device=pred.device))
            continue
        pred_valid = pred[mask]
        gt_valid = gt[mask]
        
        # Compute max(pred/gt, gt/pred)
        ratio = torch.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
        delta = (ratio < threshold).float().mean()
        delta_per_sample.append(delta)
    
    return torch.stack(delta_per_sample)


# def compute_geodesic_distance_from_two_matrices(m1, m2):
#     batch = m1.shape[0]
#     m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

#     cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
#     cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)))
#     cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) * -1)

#     theta = torch.acos(cos)

#     # theta = torch.min(theta, 2*np.pi - theta)

#     return theta


# def angle_error_mat(R1, R2):
#     cos = (torch.trace(torch.mm(R1.T, R2)) - 1) / 2
#     cos = torch.clamp(cos, -1.0, 1.0)  # numerical errors can make it out of bounds
#     return torch.rad2deg(torch.abs(torch.acos(cos)))


# def angle_error_vec(v1, v2):
#     n = torch.norm(v1) * torch.norm(v2)
#     cos_theta = torch.dot(v1, v2) / n
#     cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # numerical errors can make it out of bounds
#     return torch.rad2deg(torch.acos(cos_theta))


# def compute_translation_error(t1, t2):
#     return torch.norm(t1 - t2)


# @torch.no_grad()
# def compute_pose_error(pose_gt, pose_pred):
#     R_gt = pose_gt[:3, :3]
#     t_gt = pose_gt[:3, 3]

#     R = pose_pred[:3, :3]
#     t = pose_pred[:3, 3]

#     error_t = angle_error_vec(t, t_gt)
#     error_t = torch.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
#     error_t_scale = compute_translation_error(t, t_gt)
#     error_R = angle_error_mat(R, R_gt)
#     return error_t, error_t_scale, error_R
