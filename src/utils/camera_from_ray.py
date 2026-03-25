"""Recover full camera pose from DualDPT ray head and evaluate with standard metrics.

DualDPT ray head output (6 channels, BVHWC):
ray[:3] ≈ H_v @ [u, v, 1]  where H_v = R_rel(v) @ K⁻¹
ray[3:] ≈ relative displacement vector (spatially near-constant per view)

Metrics implemented:
1) DA3 style:  AUC@3°, AUC@30° — max(rot_err, trans_err) per pair
2) DUSt3R/VGGT style: RRA@τ, RTA@τ, mAA@30
3) Trajectory style: ATE, RPE_t, RPE_r (evo library, Sim3 aligned)
"""

import numpy as np


# ═══════════════════════════ Low-level helpers ═══════════════════════════

def _fit_homography(ray_view, subsample=4, conf=None):
    """Fit 3x3 H such that ray[:3](u,v) ≈ H @ [u,v,1]."""
    H_ray, W_ray = ray_view.shape[:2]
    direction = ray_view[..., :3].astype(np.float64)

    ys = np.arange(0, H_ray, subsample)
    xs = np.arange(0, W_ray, subsample)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    yy, xx = yy.ravel(), xx.ravel()

    P = np.stack([xx.astype(np.float64), yy.astype(np.float64),
                np.ones_like(xx, dtype=np.float64)], axis=1)
    D = direction[yy, xx]

    if conf is not None:
        w = conf[yy, xx].astype(np.float64)[:, None]
        Pw, Dw = P * w, D * w
    else:
        Pw, Dw = P, D

    Ht = np.linalg.lstsq(Pw, Dw, rcond=None)[0]
    return Ht.T


def _rotation_error_deg(R1, R2):
    """Geodesic rotation error in degrees."""
    R_err = R1 @ R2.T
    cos_a = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def _translation_direction_error_deg(t1, t2):
    """Angular error between two translation vectors (scale-free)."""
    n1, n2 = np.linalg.norm(t1), np.linalg.norm(t2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 180.0
    cos_a = np.clip(np.dot(t1, t2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


# ═══════════════════════════ Pose recovery ═══════════════════════════

def recover_intrinsics(ray, ray_conf=None, ref_view=0, subsample=4, input_size=None):
    """Recover K from ref view's ray[:3] ≈ K⁻¹ @ [u,v,1]."""
    if ray.ndim == 5:
        ray = ray[0]
    if ray_conf is not None and ray_conf.ndim == 4:
        ray_conf = ray_conf[0]

    V, H_ray, W_ray, _ = ray.shape
    conf = ray_conf[ref_view] if ray_conf is not None else None
    H_ref = _fit_homography(ray[ref_view], subsample=subsample, conf=conf)

    try:
        K = np.linalg.inv(H_ref)
    except np.linalg.LinAlgError:
        return np.eye(3)

    if abs(K[2, 2]) > 1e-12:
        K = K / K[2, 2]

    if input_size is not None:
        H_in, W_in = input_size
        K = np.diag([W_in / W_ray, H_in / H_ray, 1.0]) @ K

    return K


def recover_poses(ray, ray_conf=None, ref_view=0, subsample=4, input_size=None):
    """Recover per-view c2w from DualDPT ray head.

    Returns:
        c2w_list: list of V (4,4) ndarrays — relative poses (ref = identity).
        K: (3,3) recovered intrinsics.
    """
    if ray.ndim == 5:
        ray = ray[0]
    if ray_conf is not None and ray_conf.ndim == 4:
        ray_conf = ray_conf[0]

    V, H_ray, W_ray, _ = ray.shape

    H_list = []
    for v in range(V):
        conf_v = ray_conf[v] if ray_conf is not None else None
        H_list.append(_fit_homography(ray[v], subsample=subsample, conf=conf_v))

    H_ref = H_list[ref_view]
    try:
        H_ref_inv = np.linalg.inv(H_ref)
    except np.linalg.LinAlgError:
        H_ref_inv = np.eye(3)

    try:
        K = np.linalg.inv(H_ref)
        if abs(K[2, 2]) > 1e-12:
            K = K / K[2, 2]
    except np.linalg.LinAlgError:
        K = np.eye(3)

    if input_size is not None:
        H_in, W_in = input_size
        K = np.diag([W_in / W_ray, H_in / H_ray, 1.0]) @ K

    c2w_list = []
    for v in range(V):
        R_raw = H_list[v] @ H_ref_inv
        U, _, Vt = np.linalg.svd(R_raw)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt

        disp = ray[v, ..., 3:].astype(np.float64)
        if ray_conf is not None:
            c = ray_conf[v].astype(np.float64)
            t = (disp * c[..., None]).sum(axis=(0, 1)) / (c.sum() + 1e-12)
        else:
            t = disp.mean(axis=(0, 1))

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
        c2w_list.append(c2w)

    return c2w_list, K


# ═══════════════════════════ Pairwise errors ═══════════════════════════

def _get_pairwise_errors(pred_c2w_list, gt_c2w_list):
    """Compute per-pair rotation and translation direction errors.

    Returns:
        rot_errors: list of floats (degrees)
        trans_errors: list of floats (degrees)
    """
    V = len(pred_c2w_list)
    rot_errors, trans_errors = [], []

    for i in range(V):
        for j in range(i + 1, V):
            R_rel_pred = pred_c2w_list[i][:3, :3] @ pred_c2w_list[j][:3, :3].T
            R_rel_gt = gt_c2w_list[i][:3, :3] @ gt_c2w_list[j][:3, :3].T
            rot_errors.append(_rotation_error_deg(R_rel_pred, R_rel_gt))

            t_rel_pred = pred_c2w_list[j][:3, 3] - pred_c2w_list[i][:3, 3]
            t_rel_gt = gt_c2w_list[j][:3, 3] - gt_c2w_list[i][:3, 3]
            trans_errors.append(_translation_direction_error_deg(t_rel_pred, t_rel_gt))

    return rot_errors, trans_errors


# ═══════════════════════ DA3 style: AUC@θ ═══════════════════════════

def compute_auc(pred_c2w_list, gt_c2w_list, max_threshold=30.0, num_bins=1000):
    """Compute AUC@θ (DA3 paper style).

    For each pair: error = max(rot_err, trans_dir_err).
    AUC@θ = (1/θ) * ∫₀ᶿ accuracy(τ) dτ  where accuracy(τ) = fraction with error < τ.

    Returns:
        auc: float in [0, 1] (1 = perfect).
    """
    rot_errors, trans_errors = _get_pairwise_errors(pred_c2w_list, gt_c2w_list)
    if not rot_errors:
        return 0.0

    # Per-pair max error
    max_errors = np.maximum(np.array(rot_errors), np.array(trans_errors))

    # Integrate accuracy curve
    thresholds = np.linspace(0, max_threshold, num_bins + 1)
    accuracies = np.array([np.mean(max_errors < t) for t in thresholds])
    auc = float(np.trapz(accuracies, thresholds) / max_threshold)
    return auc


# ═══════════════════ DUSt3R/VGGT style: RRA/RTA/mAA ═══════════════════

def compute_rra_rta(pred_c2w_list, gt_c2w_list, threshold=15.0):
    """Compute RRA@τ and RTA@τ."""
    rot_errors, trans_errors = _get_pairwise_errors(pred_c2w_list, gt_c2w_list)
    if not rot_errors:
        return 0.0, 0.0, [], []

    rot_arr = np.array(rot_errors)
    trans_arr = np.array(trans_errors)
    rra = float(np.mean(rot_arr < threshold) * 100)
    rta = float(np.mean(trans_arr < threshold) * 100)
    return rra, rta, rot_errors, trans_errors


def compute_maa(pred_c2w_list, gt_c2w_list, max_threshold=30):
    """Compute mAA@T (PoseDiffusion/VGGT standard).

    Per pair: error = max(rot_err, trans_dir_err).
    mAA = mean of CDF evaluated at integer-degree bins [1..T].
    Equivalent to mean(cumsum(histogram / num_pairs)).
    """
    rot_errors, trans_errors = _get_pairwise_errors(pred_c2w_list, gt_c2w_list)
    if not rot_errors:
        return 0.0

    max_errors = np.maximum(np.array(rot_errors), np.array(trans_errors))
    # CDF at integer bins [1, 2, ..., max_threshold]
    bins = np.arange(1, max_threshold + 1, dtype=np.float64)
    cdf = np.array([np.mean(max_errors < t) for t in bins])
    return float(np.mean(cdf)) * 100  # percentage


# ═══════════════════ Trajectory style: ATE/RPE (evo) ═══════════════════

def compute_ate_rpe(pred_c2w_list, gt_c2w_list):
    """Compute ATE and RPE using evo library (Sim3 alignment).

    Returns:
        ate: float (RMSE, meters after scale alignment)
        rpe_trans: float (RMSE, meters)
        rpe_rot: float (RMSE, degrees)
    """
    if len(pred_c2w_list) < 2:
        return None, None, None

    try:
        from utils.evo_utils import eval_metrics, get_tum_poses
        import tempfile, os

        pred_traj = get_tum_poses(pred_c2w_list)
        gt_traj = get_tum_poses(gt_c2w_list)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            tmpfile = f.name
        try:
            ate, rpe_trans, rpe_rot = eval_metrics(
                pred_traj, gt_traj, seq="ray_head", filename=tmpfile)
        finally:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)
        return float(ate), float(rpe_trans), float(rpe_rot)
    except Exception as e:
        print(f"[ATE/RPE] Error: {e}")
        return None, None, None


# ═══════════════════════════ Main entry point ═══════════════════════════

def _to_ref_centric(c2w_list, ref_idx=0):
    """Convert c2w poses to ref-centric frame (ref camera = identity)."""
    T_ref_inv = np.linalg.inv(c2w_list[ref_idx])
    return [T_ref_inv @ c2w_list[v] for v in range(len(c2w_list))]


def _scale_align(pred_pts, gt_pts):
    """Least-squares scale: s = (pred . gt) / (pred . pred)."""
    num = np.sum(pred_pts * gt_pts)
    den = np.sum(pred_pts * pred_pts)
    return num / den if den > 1e-12 else 1.0


def compute_camera_metrics(ray, ray_conf, gt_K, gt_c2w, cond_num,
                        subsample=4, input_size=None):
    """Compute ALL camera metrics from ray head output.

    Key: pred poses are in ref-centric frame, so GT is converted to
    ref-centric frame before comparison. ATE/RPE use evo library
    (same as eval_cam.sh: Sim3 Umeyama, all_pairs, RMSE).

    Returns dict with:
        - focal_rel_err: relative focal length error
        - auc3, auc30: DA3-style AUC (max(rot,trans) per pair, scale-free)
        - rra15, rta15, maa30: DUSt3R/VGGT pairwise metrics (scale-free)
        - mean_rot_err, mean_trans_err: average pairwise errors (scale-free)
        - ate, rpe_trans, rpe_rot: trajectory metrics (evo Sim3 Umeyama, no pre-scaling)
        - scale: LS scale factor (for visualization only)
        - recovered_K, pred_c2w: recovered camera parameters
        - gt_c2w_ref: GT in ref-centric frame
        - pred_c2w_scaled: pred with LS scale-aligned translations (visualization only)
    """
    if ray.ndim == 5:
        ray = ray[0]
    if ray_conf is not None and ray_conf.ndim == 4:
        ray_conf = ray_conf[0]

    V = ray.shape[0]

    # Recover poses (already in ref-centric frame: ref=identity)
    pred_c2w_list, K_pred = recover_poses(
        ray[None], ray_conf[None] if ray_conf is not None else None,
        ref_view=0, subsample=subsample, input_size=input_size,
    )

    # Convert GT to ref-centric frame
    gt_c2w_world = [gt_c2w[v].astype(np.float64) for v in range(V)]
    gt_c2w_ref = _to_ref_centric(gt_c2w_world, ref_idx=0)

    # Scale alignment (LS)
    pred_pts = np.array([p[:3, 3] for p in pred_c2w_list])
    gt_pts_ref = np.array([g[:3, 3] for g in gt_c2w_ref])
    scale = _scale_align(pred_pts, gt_pts_ref)

    # Scale-aligned pred (for visualization ONLY — metrics use un-scaled pred)
    pred_scaled = []
    for p in pred_c2w_list:
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = p[:3, :3]
        c2w[:3, 3] = p[:3, 3] * scale
        pred_scaled.append(c2w)

    # Focal length error
    gt_fx, gt_fy = gt_K[0, 0, 0], gt_K[0, 1, 1]
    pred_fx, pred_fy = abs(K_pred[0, 0]), abs(K_pred[1, 1])
    focal_rel_err = (abs(pred_fx - gt_fx) / (gt_fx + 1e-8)
                    + abs(pred_fy - gt_fy) / (gt_fy + 1e-8)) / 2.0

    # Target views only (ref-centric frame, no pre-scaling)
    pred_tgt = [pred_c2w_list[v] for v in range(cond_num, V)]
    gt_tgt = [gt_c2w_ref[v] for v in range(cond_num, V)]

    # Defaults
    auc3, auc30 = 0.0, 0.0
    rra15, rta15, maa30 = 0.0, 0.0, 0.0
    mean_rot_err, mean_trans_err = None, None
    ate, rpe_trans, rpe_rot = None, None, None

    if len(pred_tgt) >= 2:
        # Pairwise metrics (scale-free: rotation=geodesic, translation=angular)
        auc3 = compute_auc(pred_tgt, gt_tgt, max_threshold=3.0)
        auc30 = compute_auc(pred_tgt, gt_tgt, max_threshold=30.0)

        rra15, rta15, rot_errs, trans_errs = compute_rra_rta(pred_tgt, gt_tgt, 15.0)
        maa30 = compute_maa(pred_tgt, gt_tgt, 30)
        mean_rot_err = float(np.mean(rot_errs))
        mean_trans_err = float(np.mean(trans_errs))

        # Trajectory metrics via evo (Sim3 Umeyama alignment inside evo)
        # Pass un-scaled pred — evo handles align=True, correct_scale=True
        ate, rpe_trans, rpe_rot = compute_ate_rpe(pred_tgt, gt_tgt)

    return {
        'focal_rel_err': float(focal_rel_err),
        'auc3': auc3, 'auc30': auc30,
        'rra15': rra15, 'rta15': rta15, 'maa30': maa30,
        'mean_rot_err': mean_rot_err, 'mean_trans_err': mean_trans_err,
        'ate': ate, 'rpe_trans': rpe_trans, 'rpe_rot': rpe_rot,
        'scale': float(scale),
        'recovered_K': K_pred,
        'pred_c2w': pred_c2w_list,
        'gt_c2w_ref': gt_c2w_ref,
        'pred_c2w_scaled': pred_scaled,
    }
