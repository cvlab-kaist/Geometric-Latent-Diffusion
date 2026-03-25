#!/usr/bin/env python3

"""Reconstruct 3D scenes from DA3RAE raw NPZ exports.

Supports:
- input path as a single `sample_xxxx_raw.npz` or a directory containing them
- camera source selection (`gt` or `pred`)
- optional Sim(3) alignment of predicted poses to GT poses
- export to GLB and/or COLMAP using Depth-Anything-3 utilities
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals


REPO_ROOT = Path(__file__).resolve().parents[1]
# depth-anything-3 is installed via pip; this fallback is for local dev only
DA3_SRC = REPO_ROOT / "third_party" / "Depth-Anything-3" / "src"

import sys

if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export.colmap import export_to_colmap
from depth_anything_3.utils.export.glb import export_to_glb
from depth_anything_3.utils.export.glb import (
    _add_cameras_to_scene,
    _compute_alignment_transform_first_cam_glTF_center_by_points,
    _depths_to_world_points_with_colors,
    _estimate_scene_scale,
    _filter_and_downsample,
    get_conf_thresh,
)
from depth_anything_3.utils.pose_align import align_poses_umeyama


def _list_npz_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix != ".npz":
            raise ValueError(f"Expected .npz file, got: {input_path}")
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("sample_*_raw.npz"))
        if not files:
            raise FileNotFoundError(f"No sample_*_raw.npz under: {input_path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _ensure_4x4(pose: np.ndarray) -> np.ndarray:
    if pose.shape[-2:] == (4, 4):
        return pose
    if pose.shape[-2:] == (3, 4):
        out = np.zeros((*pose.shape[:-2], 4, 4), dtype=pose.dtype)
        out[..., :3, :4] = pose
        out[..., 3, 3] = 1.0
        return out
    raise ValueError(f"Pose must be (...,4,4) or (...,3,4), got: {pose.shape}")


def _fxfycxcy_to_K(fxfycxcy: np.ndarray, height: int, width: int) -> np.ndarray:
    fx, fy, cx, cy = map(float, fxfycxcy.tolist())
    if fx < 10.0 and fy < 10.0:
        fx *= width
        fy *= height
        cx *= width
        cy *= height
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def _extract_depth(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    depth = np.asarray(npz["pred_depth"], dtype=np.float64)
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    if depth.ndim != 3:
        raise ValueError(f"pred_depth must be (V,H,W) or (V,1,H,W), got: {depth.shape}")
    return depth


def _extract_rgb_u8(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    rgb = np.asarray(npz["pred_rgb"], dtype=np.float64)
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"pred_rgb must be (V,3,H,W), got: {rgb.shape}")
    rgb = np.transpose(rgb, (0, 2, 3, 1))
    rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb_u8


def _extract_pred_ray_conf(npz: np.lib.npyio.NpzFile, views: int) -> np.ndarray:
    if "pred_ray_conf" not in npz:
        raise KeyError("Pred confidence missing: expected `pred_ray_conf` in npz")
    conf = np.asarray(npz["pred_ray_conf"], dtype=np.float64)
    if conf.ndim == 4 and conf.shape[0] == 1:
        conf = conf[0]
    if conf.ndim == 4 and conf.shape[1] == 1:
        conf = conf[:, 0]
    if conf.ndim != 3:
        raise ValueError(f"pred_ray_conf must be (V,h,w) or (1,V,h,w), got: {conf.shape}")
    if conf.shape[0] != views:
        raise ValueError(f"pred_ray_conf views mismatch: {conf.shape[0]} vs {views}")
    return conf


def _resize_conf_to_depth_resolution(conf: np.ndarray, height: int, width: int) -> np.ndarray:
    if conf.shape[1] == height and conf.shape[2] == width:
        return conf.astype(np.float32)
    out = np.empty((conf.shape[0], height, width), dtype=np.float32)
    for i in range(conf.shape[0]):
        im = Image.fromarray(conf[i].astype(np.float32))
        im = im.resize((width, height), Image.BILINEAR)
        out[i] = np.asarray(im, dtype=np.float32)
    return out


def _extract_gt_intrinsics(npz: np.lib.npyio.NpzFile, height: int, width: int, views: int) -> np.ndarray:
    if "intrinsics" in npz:
        intr = np.asarray(npz["intrinsics"], dtype=np.float64)
    elif "fxfycxcy" in npz:
        intr = np.asarray(npz["fxfycxcy"], dtype=np.float64)
    else:
        raise KeyError("GT intrinsics missing: expected `intrinsics` or `fxfycxcy` in npz")

    if intr.ndim == 4 and intr.shape[0] == 1:
        intr = intr[0]

    if intr.ndim == 3 and intr.shape[-2:] == (3, 3):
        if intr.shape[0] != views:
            raise ValueError(f"GT intrinsics views mismatch: {intr.shape[0]} vs {views}")
        return intr

    if intr.ndim == 2 and intr.shape[1] == 4:
        if intr.shape[0] != views:
            raise ValueError(f"GT fxfycxcy views mismatch: {intr.shape[0]} vs {views}")
        return np.stack([_fxfycxcy_to_K(intr[i], height, width) for i in range(views)], axis=0)

    raise ValueError(f"Unsupported GT intrinsics shape: {intr.shape}")


def _extract_gt_c2w(npz: np.lib.npyio.NpzFile, views: int) -> np.ndarray:
    if "extrinsics" not in npz:
        raise KeyError("GT extrinsics missing: expected `extrinsics` in npz")
    c2w = np.asarray(npz["extrinsics"], dtype=np.float64)
    if c2w.ndim == 4 and c2w.shape[0] == 1:
        c2w = c2w[0]
    c2w = _ensure_4x4(c2w)
    if c2w.shape[0] != views:
        raise ValueError(f"GT c2w views mismatch: {c2w.shape[0]} vs {views}")
    return c2w


def _extract_pred_intrinsics(npz: np.lib.npyio.NpzFile, views: int) -> np.ndarray:
    if "pred_K" not in npz:
        raise KeyError("Pred intrinsics missing: expected `pred_K` in npz")
    pred_K = np.asarray(npz["pred_K"], dtype=np.float64)
    if pred_K.shape == (3, 3):
        return np.repeat(pred_K[None, ...], views, axis=0)
    if pred_K.ndim == 3 and pred_K.shape[-2:] == (3, 3) and pred_K.shape[0] == views:
        return pred_K
    raise ValueError(f"Unsupported pred_K shape: {pred_K.shape}")


def _extract_pred_c2w(npz: np.lib.npyio.NpzFile, views: int) -> np.ndarray:
    if "pred_c2w" not in npz:
        raise KeyError("Pred poses missing: expected `pred_c2w` in npz")
    c2w = np.asarray(npz["pred_c2w"], dtype=np.float64)
    c2w = _ensure_4x4(c2w)
    if c2w.shape[0] != views:
        raise ValueError(f"Pred c2w views mismatch: {c2w.shape[0]} vs {views}")
    return c2w


def _cond_num(npz: np.lib.npyio.NpzFile) -> int:
    if "cond_num" not in npz:
        return 0
    return int(np.asarray(npz["cond_num"]).item())


def _align_pred_c2w_to_gt_sim3(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    ransac: bool,
    ransac_max_iters: int,
) -> tuple[np.ndarray, float]:
    pred_w2c = np.linalg.inv(pred_c2w)
    gt_w2c = np.linalg.inv(gt_c2w)
    _, _, scale, pred_w2c_aligned = align_poses_umeyama(
        ext_ref=gt_w2c,
        ext_est=pred_w2c,
        return_aligned=True,
        ransac=ransac,
        ransac_max_iters=ransac_max_iters,
    )
    pred_c2w_aligned = np.linalg.inv(pred_w2c_aligned)
    return pred_c2w_aligned, float(scale)


def _view_indices(total_views: int, cond_num: int, view_set: str) -> np.ndarray:
    if view_set == "all":
        return np.arange(total_views)
    if view_set == "target":
        if cond_num >= total_views:
            raise ValueError(f"cond_num={cond_num} >= total_views={total_views}, target set is empty")
        return np.arange(cond_num, total_views)
    if view_set == "reference":
        if cond_num <= 0:
            raise ValueError(f"cond_num={cond_num}, reference set is empty")
        return np.arange(min(cond_num, total_views))
    raise ValueError(f"Unsupported view_set: {view_set}")


def _export_colmap_bundle(
    prediction: Prediction,
    rgb_u8: np.ndarray,
    out_dir: Path,
    conf_thresh_percentile: float,
) -> None:
    images_dir = out_dir / "images"
    sparse_dir = out_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[str] = []
    for i in range(rgb_u8.shape[0]):
        image_path = images_dir / f"{i:04d}.png"
        Image.fromarray(rgb_u8[i]).save(image_path)
        image_paths.append(str(image_path))

    export_to_colmap(
        prediction=prediction,
        export_dir=str(sparse_dir),
        image_paths=image_paths,
        conf_thresh_percentile=conf_thresh_percentile,
    )


def _write_meta(meta_path: Path, meta: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)


def _resize_rgb_u8(image_u8: np.ndarray, max_edge: int) -> np.ndarray:
    if max_edge <= 0:
        return image_u8
    h, w = image_u8.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_edge:
        return image_u8
    scale = float(max_edge) / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return np.asarray(Image.fromarray(image_u8).resize((new_w, new_h), Image.BILINEAR))


def _make_view_plane_mesh(
    K: np.ndarray,
    ext_w2c: np.ndarray,
    image_u8: np.ndarray,
    plane_depth: float,
    alignment: np.ndarray,
) -> trimesh.Trimesh:
    h, w = image_u8.shape[:2]
    corners_px = np.array(
        [
            [0.0, 0.0, 1.0],
            [w - 1.0, 0.0, 1.0],
            [w - 1.0, h - 1.0, 1.0],
            [0.0, h - 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    K_inv = np.linalg.inv(K)
    c2w = np.linalg.inv(_ensure_4x4(ext_w2c))

    rays = (K_inv @ corners_px.T).T
    z = rays[:, 2:3]
    z[np.abs(z) < 1e-8] = 1.0
    plane_cam = (rays / z) * plane_depth
    plane_cam_h = np.concatenate([plane_cam, np.ones((4, 1), dtype=np.float64)], axis=1)
    plane_world = (c2w @ plane_cam_h.T).T[:, :3]
    plane_aligned = trimesh.transform_points(plane_world, alignment)

    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uv = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    material = PBRMaterial(
        baseColorTexture=Image.fromarray(image_u8),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
        alphaMode="OPAQUE",
    )
    visual = TextureVisuals(uv=uv, material=material)
    mesh = trimesh.Trimesh(vertices=plane_aligned, faces=faces, visual=visual, process=False)
    return mesh


def _export_glb_with_camera_views(
    prediction: Prediction,
    out_file: Path,
    num_max_points: int,
    conf_thresh: float,
    conf_thresh_percentile: float,
    ensure_thresh_percentile: float,
    show_cameras: bool,
    camera_size: float,
    view_plane_interval: int,
    view_plane_max_edge: int,
    view_plane_depth_factor: float,
) -> None:
    assert prediction.processed_images is not None
    assert prediction.depth is not None
    assert prediction.intrinsics is not None
    assert prediction.extrinsics is not None
    assert prediction.conf is not None

    conf_thr = get_conf_thresh(
        prediction,
        sky_mask=getattr(prediction, "sky_mask", None),
        conf_thresh=conf_thresh,
        conf_thresh_percentile=conf_thresh_percentile,
        ensure_thresh_percentile=ensure_thresh_percentile,
    )
    points_world, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,
        prediction.processed_images,
        prediction.conf,
        conf_thr,
    )
    alignment = _compute_alignment_transform_first_cam_glTF_center_by_points(
        prediction.extrinsics[0], points_world
    )
    if len(points_world) > 0:
        points_world = trimesh.transform_points(points_world, alignment)
    points_world, colors = _filter_and_downsample(points_world, colors, num_max_points)

    scene = trimesh.Scene()
    if scene.metadata is None:
        scene.metadata = {}
    scene.metadata["hf_alignment"] = alignment
    if len(points_world) > 0:
        scene.add_geometry(trimesh.points.PointCloud(vertices=points_world, colors=colors))

    scene_scale = _estimate_scene_scale(points_world, fallback=1.0)
    frame_h, frame_w = prediction.depth.shape[1:]
    if show_cameras:
        _add_cameras_to_scene(
            scene=scene,
            K=prediction.intrinsics,
            ext_w2c=prediction.extrinsics,
            image_sizes=[(frame_h, frame_w)] * prediction.depth.shape[0],
            scale=scene_scale * camera_size,
        )

    frustum_depth = scene_scale * camera_size
    plane_depth = frustum_depth * view_plane_depth_factor
    step = max(1, int(view_plane_interval))
    for idx in range(0, prediction.depth.shape[0], step):
        image_u8 = _resize_rgb_u8(prediction.processed_images[idx], view_plane_max_edge)
        mesh = _make_view_plane_mesh(
            K=prediction.intrinsics[idx],
            ext_w2c=prediction.extrinsics[idx],
            image_u8=image_u8,
            plane_depth=plane_depth,
            alignment=alignment,
        )
        scene.add_geometry(mesh, node_name=f"view_plane_{idx:04d}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_file))


def process_npz(npz_path: Path, args: argparse.Namespace) -> tuple[bool, str]:
    conf: np.ndarray | None = None
    with np.load(npz_path, allow_pickle=True) as data:
        depth = _extract_depth(data)
        rgb_u8 = _extract_rgb_u8(data)
        cond_num = _cond_num(data)
        views, height, width = depth.shape
        if args.use_pred_ray_conf:
            if "pred_ray_conf" in data:
                ray_conf = _extract_pred_ray_conf(data, views)
                conf = _resize_conf_to_depth_resolution(ray_conf, height, width)
            elif args.require_pred_ray_conf:
                raise KeyError(
                    f"{npz_path}: `pred_ray_conf` is missing while --require-pred-ray-conf is set"
                )
            else:
                print(f"[Recon] WARN: {npz_path.name} has no pred_ray_conf; fallback to uniform conf")

        if args.camera_source == "gt":
            intrinsics = _extract_gt_intrinsics(data, height, width, views)
            c2w = _extract_gt_c2w(data, views)
            sim3_scale = 1.0
            sim3_applied = False
        else:
            intrinsics = _extract_pred_intrinsics(data, views)
            c2w = _extract_pred_c2w(data, views)
            sim3_applied = False
            sim3_scale = 1.0
            if args.sim3:
                gt_c2w = _extract_gt_c2w(data, views)
                c2w, sim3_scale = _align_pred_c2w_to_gt_sim3(
                    pred_c2w=c2w,
                    gt_c2w=gt_c2w,
                    ransac=args.sim3_ransac,
                    ransac_max_iters=args.sim3_ransac_max_iters,
                )
                sim3_applied = True
                if args.scale_depth_with_sim3:
                    depth = depth * sim3_scale

        keep = _view_indices(views, cond_num, args.view_set)
        depth = depth[keep]
        rgb_u8 = rgb_u8[keep]
        intrinsics = intrinsics[keep]
        c2w = c2w[keep]
        if conf is not None:
            conf = conf[keep]
        w2c = np.linalg.inv(c2w)

    if conf is None:
        conf = np.ones_like(depth, dtype=np.float32)
        conf_source = "uniform_ones"
    else:
        conf = np.clip(conf.astype(np.float32), 1e-8, None)
        conf_source = "pred_ray_conf_resized"

    prediction = Prediction(
        depth=depth.astype(np.float32),
        is_metric=1 if (args.camera_source == "gt" or sim3_applied) else 0,
        conf=conf,
        extrinsics=w2c.astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        processed_images=rgb_u8,
        scale_factor=sim3_scale if sim3_applied else None,
    )

    sample_name = npz_path.stem.replace("_raw", "")
    suffix_parts = [args.camera_source]
    if sim3_applied:
        suffix_parts.append("sim3")
    suffix_parts.append(f"{args.view_set}_views")
    run_name = "_".join(suffix_parts)
    scene_out = args.output_root / sample_name / run_name
    scene_out.mkdir(parents=True, exist_ok=True)

    if args.export in ("glb", "both"):
        glb_dir = scene_out / "glb"
        glb_dir.mkdir(parents=True, exist_ok=True)
        export_to_glb(
            prediction=prediction,
            export_dir=str(glb_dir),
            num_max_points=args.num_max_points,
            conf_thresh=args.conf_thresh,
            conf_thresh_percentile=args.conf_thresh_percentile,
            ensure_thresh_percentile=args.ensure_thresh_percentile,
            show_cameras=not args.hide_cameras,
            export_depth_vis=args.export_depth_vis,
        )
        if args.export_view_glb:
            _export_glb_with_camera_views(
                prediction=prediction,
                out_file=glb_dir / "scene_with_views.glb",
                num_max_points=args.num_max_points,
                conf_thresh=args.conf_thresh,
                conf_thresh_percentile=args.conf_thresh_percentile,
                ensure_thresh_percentile=args.ensure_thresh_percentile,
                show_cameras=not args.hide_cameras,
                camera_size=args.camera_size,
                view_plane_interval=args.view_plane_interval,
                view_plane_max_edge=args.view_plane_max_edge,
                view_plane_depth_factor=args.view_plane_depth_factor,
            )

    if args.export in ("colmap", "both"):
        _export_colmap_bundle(
            prediction=prediction,
            rgb_u8=rgb_u8,
            out_dir=scene_out / "colmap",
            conf_thresh_percentile=args.conf_thresh_percentile,
        )

    meta = {
        "input_npz": str(npz_path),
        "output_dir": str(scene_out),
        "camera_source": args.camera_source,
        "sim3_applied": sim3_applied,
        "sim3_scale": sim3_scale,
        "scale_depth_with_sim3": bool(args.scale_depth_with_sim3),
        "view_set": args.view_set,
        "cond_num": int(cond_num),
        "kept_view_indices": keep.tolist(),
        "num_views_in_output": int(len(keep)),
        "export": args.export,
        "conf_source": conf_source,
        "use_pred_ray_conf": bool(args.use_pred_ray_conf),
        "require_pred_ray_conf": bool(args.require_pred_ray_conf),
        "conf_thresh": float(args.conf_thresh),
        "conf_thresh_percentile": float(args.conf_thresh_percentile),
        "ensure_thresh_percentile": float(args.ensure_thresh_percentile),
        "conf_stats": {
            "min": float(np.min(conf)),
            "p10": float(np.percentile(conf, 10.0)),
            "p50": float(np.percentile(conf, 50.0)),
            "p90": float(np.percentile(conf, 90.0)),
            "max": float(np.max(conf)),
        },
        "export_view_glb": bool(args.export_view_glb),
        "camera_size": float(args.camera_size),
        "view_plane_interval": int(args.view_plane_interval),
        "view_plane_max_edge": int(args.view_plane_max_edge),
        "view_plane_depth_factor": float(args.view_plane_depth_factor),
    }
    _write_meta(scene_out / "recon_meta.json", meta)
    return True, str(scene_out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D reconstruction from DA3RAE raw npz")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to sample_xxxx_raw.npz or directory containing sample_*_raw.npz",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "tmp" / "npz_reconstruction",
        help="Root output directory",
    )
    parser.add_argument(
        "--camera-source",
        choices=["gt", "pred"],
        default="pred",
        help="Which camera to use for reconstruction",
    )
    parser.add_argument(
        "--sim3",
        action="store_true",
        help="Apply Sim(3) alignment (pred camera only) against GT c2w from npz",
    )
    parser.add_argument(
        "--sim3-ransac",
        action="store_true",
        help="Use RANSAC-enabled Sim(3) alignment",
    )
    parser.add_argument(
        "--sim3-ransac-max-iters",
        type=int,
        default=10,
        help="RANSAC iterations for Sim(3) alignment",
    )
    parser.add_argument(
        "--no-scale-depth-with-sim3",
        dest="scale_depth_with_sim3",
        action="store_false",
        help="Do not multiply depth by Sim(3) scale",
    )
    parser.set_defaults(scale_depth_with_sim3=True)
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Deprecated alias for --view-set target",
    )
    parser.add_argument(
        "--view-set",
        choices=["all", "reference", "target"],
        default="all",
        help="Which view subset to reconstruct",
    )
    parser.add_argument(
        "--export",
        choices=["glb", "colmap", "both"],
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of npz files processed",
    )
    parser.add_argument("--num-max-points", type=int, default=500_000, help="GLB max points")
    parser.add_argument(
        "--use-pred-ray-conf",
        action="store_true",
        help="Use `pred_ray_conf` from npz (resized to depth resolution) as confidence for filtering",
    )
    parser.add_argument(
        "--require-pred-ray-conf",
        action="store_true",
        help="Fail if `pred_ray_conf` is unavailable while --use-pred-ray-conf is enabled",
    )
    parser.add_argument("--conf-thresh", type=float, default=1.05, help="Base confidence threshold")
    parser.add_argument(
        "--conf-thresh-percentile",
        type=float,
        default=0.0,
        help="Confidence percentile lower bound for point filtering",
    )
    parser.add_argument(
        "--ensure-thresh-percentile",
        type=float,
        default=100.0,
        help="Confidence percentile upper clamp",
    )
    parser.add_argument("--hide-cameras", action="store_true", help="Do not render cameras in GLB")
    parser.add_argument(
        "--camera-size",
        type=float,
        default=0.03,
        help="Camera frustum size ratio relative to scene scale (GLB)",
    )
    parser.add_argument(
        "--export-depth-vis",
        action="store_true",
        help="Also export depth visualizations alongside GLB",
    )
    parser.add_argument(
        "--export-view-glb",
        action="store_true",
        help="Also export scene_with_views.glb with textured camera view planes",
    )
    parser.add_argument(
        "--view-plane-interval",
        type=int,
        default=1,
        help="Render one view plane every N cameras (1 = all)",
    )
    parser.add_argument(
        "--view-plane-max-edge",
        type=int,
        default=320,
        help="Max edge length for each camera image texture (pixels)",
    )
    parser.add_argument(
        "--view-plane-depth-factor",
        type=float,
        default=1.0,
        help="Depth of image plane relative to frustum base depth",
    )
    return parser.parse_args()


def _iter_with_limit(paths: list[Path], max_samples: int | None) -> Iterable[Path]:
    if max_samples is None:
        return paths
    return paths[: max(0, max_samples)]


def main() -> None:
    args = _parse_args()
    if args.require_pred_ray_conf and not args.use_pred_ray_conf:
        raise ValueError("--require-pred-ray-conf needs --use-pred-ray-conf")
    if args.target_only and args.view_set == "all":
        args.view_set = "target"
    npz_files = _list_npz_files(args.input)
    args.output_root.mkdir(parents=True, exist_ok=True)

    selected = list(_iter_with_limit(npz_files, args.max_samples))
    print(f"[Recon] Found {len(npz_files)} files, processing {len(selected)}")

    success = 0
    failures: list[tuple[str, str]] = []
    for path in selected:
        print(f"[Recon] Processing: {path}")
        try:
            ok, out_dir = process_npz(path, args)
            if ok:
                success += 1
                print(f"[Recon] Done: {out_dir}")
        except Exception as exc:
            failures.append((str(path), str(exc)))
            print(f"[Recon] Failed: {path}")
            print(f"         {exc}")
            traceback.print_exc()

    print(f"[Recon] Summary: success={success}, failed={len(failures)}")
    if failures:
        for p, e in failures:
            print(f"  - {p}: {e}")


if __name__ == "__main__":
    main()
