#!/usr/bin/env python
"""Batch-isolated calibrated renderer for sparse 6--10 camera rigs.

Each physical camera is one network batch item, so DA3 global attention never
mixes unrelated sparse views.  The independent monocular depths are anchored
to one calibrated focus point and lifted into one shared, normalised rig world.
At display time only the two adjacent Gaussian anchors are rasterised; the
second anchor may fill narrow, depth-verified alpha holes but never overwrites
an already covered primary foreground.

One input frame per camera is a fast streaming 3D refresh.  Two or more frames
per camera retain NoPo4D's temporal Gaussian field and permit time playback.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
from dataclasses import dataclass, fields
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference import (  # noqa: E402
    IMAGE_EXTENSIONS,
    _camera_id,
    _natural_key,
    build_pose_inputs,
    estimate_camera_focus,
    load_calibrations,
    load_images,
)
from src.model.nopo4d import NoPo4D  # noqa: E402
from src.model.types import Gaussians  # noqa: E402
from src.utils.geometry import mat_to_quat, quat_to_mat, qmul  # noqa: E402


@dataclass
class SparseAnchorBank:
    model: NoPo4D
    gaussians: Gaussians                 # [camera, gaussian, ...]
    images: torch.Tensor                 # [camera, time, 3, H, W]
    c2w: torch.Tensor                    # [camera, time, 4, 4], shared rig world
    intrinsics: torch.Tensor             # [camera, time, 3, 3]
    timestamps: torch.Tensor | None       # [camera, time]
    camera_ids: list[int]
    height: int
    width: int
    axis_gaps_deg: torch.Tensor           # [camera-1]
    forward_seconds: float
    depth_scales: torch.Tensor
    local_gaussians: bool = False
    source_valid_mask: torch.Tensor | None = None
    background_gaussians: Gaussians | None = None

    @property
    def cameras(self) -> int:
        return len(self.camera_ids)

    @property
    def frames(self) -> int:
        return self.images.shape[1]


def _parse_ids(value: str) -> list[int]:
    ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("camera IDs must contain at least two unique integers")
    return ids


def _discover_camera_ids(image_dir: Path, cameras: int) -> list[int]:
    paths = sorted(
        (path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=_natural_key,
    )
    if len(paths) % cameras:
        raise ValueError(f"{len(paths)} images are not divisible by {cameras} cameras")
    frames = len(paths) // cameras
    return [_camera_id(paths[index * frames]) for index in range(cameras)]


def _normalise_complete_rig(
    flat_w2c: torch.Tensor,
    cameras: int,
    frames: int,
    focus_override: tuple[float, float, float] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Normalise the complete rig once, before splitting cameras along B."""
    w2c = flat_w2c.reshape(cameras, frames, 4, 4).double()
    c2w = torch.linalg.inv(w2c)
    physical_anchors = c2w[:, 0]
    focus_physical = (
        physical_anchors.new_tensor(focus_override)
        if focus_override is not None
        else estimate_camera_focus(physical_anchors)
    )
    radius = torch.linalg.vector_norm(
        physical_anchors[:, :3, 3] - focus_physical, dim=-1
    ).median().clamp_min(1e-4)

    normalised_c2w = c2w.clone()
    normalised_c2w[..., :3, 3] = (
        c2w[..., :3, 3] - focus_physical.view(1, 1, 3)
    ) / radius
    normalised_w2c = torch.linalg.inv(normalised_c2w).float()
    focus_normalised = torch.zeros(3, dtype=torch.float32)

    focus_cam = torch.einsum(
        "cij,j->ci",
        normalised_w2c[:, 0],
        torch.cat((focus_normalised, torch.ones(1))),
    )
    if (focus_cam[:, 2] <= 0).any():
        bad = torch.where(focus_cam[:, 2] <= 0)[0].tolist()
        raise ValueError(f"calibrated focus lies behind cameras {bad}; check w2c convention")
    return normalised_w2c, normalised_c2w.float(), focus_physical.float(), float(radius)


def _gaussian_batches(gaussians: Gaussians, indices: list[int]) -> Gaussians:
    index = torch.tensor(indices, device=gaussians.means.device, dtype=torch.long)
    values = {}
    for field in fields(gaussians):
        value = getattr(gaussians, field.name)
        values[field.name] = None if value is None else value.index_select(0, index)
    return Gaussians(**values)


def triangulated_depth_anchors(images, w2c, intrinsics, feature_images=None, workspace_only=False):
    """SIFT matches verified against known calibration, not a fitted pose.

    Returning metric camera-Z samples lets the encoder align actual observed
    surfaces instead of assuming the centre pixel lies at the optical focus.
    """
    feature_images = images if feature_images is None else feature_images
    arrays = (feature_images[:, 0].detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    sx = images.shape[-1] / feature_images.shape[-1]
    sy = images.shape[-2] / feature_images.shape[-2]
    ext, ks = w2c[:, 0].cpu().numpy(), intrinsics[:, 0].cpu().numpy()
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.02)
    features = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None) for im in arrays]
    samples = [[] for _ in arrays]
    reports = []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            ki, di = features[i]
            kj, dj = features[j]
            if di is None or dj is None or min(len(di), len(dj)) < 2:
                continue
            matches = [a for a, b in matcher.knnMatch(di, dj, k=2) if a.distance < 0.72 * b.distance]
            if len(matches) < 8:
                continue
            pi = np.float32([ki[m.queryIdx].pt for m in matches])
            pj = np.float32([kj[m.trainIdx].pt for m in matches])
            pi *= np.float32([sx, sy])
            pj *= np.float32([sx, sy])
            proj_i, proj_j = ks[i] @ ext[i, :3], ks[j] @ ext[j, :3]
            homogeneous = cv2.triangulatePoints(proj_i, proj_j, pi.T, pj.T).T
            good_w = np.abs(homogeneous[:, 3]) > 1e-8
            xyz = homogeneous[:, :3] / np.where(good_w, homogeneous[:, 3], 1)[:, None]
            xyz_h = np.c_[xyz, np.ones(len(xyz))]
            ci, cj = xyz_h @ ext[i].T, xyz_h @ ext[j].T
            qi, qj = xyz_h @ proj_i.T, xyz_h @ proj_j.T
            ui = qi[:, :2] / np.maximum(qi[:, 2:], 1e-8)
            uj = qj[:, :2] / np.maximum(qj[:, 2:], 1e-8)
            error = np.maximum(np.linalg.norm(ui - pi, axis=1), np.linalg.norm(uj - pj, axis=1))
            # Reject points far outside the capture volume and behind cameras.
            valid = good_w & np.isfinite(xyz).all(1) & (ci[:, 2] > 0.15) & (cj[:, 2] > 0.15)
            valid &= (ci[:, 2] < 3) & (cj[:, 2] < 3) & (error < 0.8)
            if workspace_only:
                # The room contains repeated light panels. Prefer real points
                # in the calibrated work volume over far-wall correspondences
                # when aligning people, hands and tabletop geometry.
                valid &= np.linalg.norm(xyz, axis=1) < .65
            for k, uv, camera_xyz in ((i, pi, ci), (j, pj, cj)):
                samples[k].extend(np.c_[uv[valid], camera_xyz[valid, 2]].tolist())
            reports.append({"pair": [i, j], "ratio_matches": len(matches), "calibration_inliers": int(valid.sum())})
    maximum = max(1, max(map(len, samples)))
    padded = torch.full((len(arrays), maximum, 3), float("nan"), device=images.device)
    for i, points in enumerate(samples):
        if points:
            padded[i, :len(points)] = torch.tensor(points, device=images.device)
    print("[MATCH] calibrated inlier counts per camera:", list(map(len, samples)))
    return padded, reports


def _merge_gaussian_batches(gaussians: Gaussians) -> Gaussians:
    """Merge already world-aligned batches only for a controlled diagnostic."""
    values = {}
    for field in fields(gaussians):
        value = getattr(gaussians, field.name)
        values[field.name] = (
            None if value is None else value.reshape(1, -1, *value.shape[2:])
        )
    return Gaussians(**values)


def build_static_background(bank: SparseAnchorBank, workspace_radius: float) -> Gaussians:
    """Merge only far environment outside a conservative work-area cylinder.

    Central people/table geometry is excluded from every helper background.
    The environment uses the first synchronized time only and DC appearance.
    """
    if not bank.local_gaussians:
        raise ValueError("background shell requires source-camera Gaussian frames")
    up = F.normalize(-bank.c2w[:, 0, :3, 1].mean(0), dim=0)
    radius = bank.c2w[:, 0, :3, 3].norm(dim=-1).median()
    parts = []
    pixels = bank.height * bank.width
    for i in range(bank.cameras):
        pose = bank.c2w[i, 0]
        local = bank.gaussians.means[i, :pixels]
        world = local @ pose[:3, :3].T + pose[:3, 3]
        horizontal = world - (world @ up)[:, None] * up
        keep = (horizontal.norm(dim=-1) > workspace_radius * radius)
        keep &= (world.norm(dim=-1) < radius * 4)
        keep &= bank.gaussians.opacities[i, :pixels] > .05
        rotation = mat_to_quat(pose[:3, :3])[[3,0,1,2]]
        # Splat quaternions and qmul use scalar-first, unlike pose helpers.
        rotations = qmul(rotation.expand(local.shape[0], -1), bank.gaussians.rotations[i, :pixels])
        parts.append(Gaussians(
            means=world[keep][None],
            harmonics=bank.gaussians.harmonics[i, :pixels][keep][None, ..., :1],
            opacities=bank.gaussians.opacities[i, :pixels][keep][None],
            scales=bank.gaussians.scales[i, :pixels][keep][None],
            rotations=rotations[keep][None],
        ))
    merged = {}
    for field in fields(parts[0]):
        values = [getattr(part, field.name) for part in parts]
        merged[field.name] = None if values[0] is None else torch.cat(values, dim=1)
    background = Gaussians(**merged)
    print(f"[BACKGROUND] {background.means.shape[1]:,} static environment Gaussians; "
          f"central workspace radius excluded={workspace_radius * float(radius):.3f}")
    return background


def _interpolate_pose(bank: SparseAnchorBank, position: float):
    position = float(np.clip(position, 0.0, bank.cameras - 1.0))
    left = min(int(np.floor(position)), bank.cameras - 2)
    amount = position - left
    a, b = bank.c2w[left, 0], bank.c2w[left + 1, 0]
    qa, qb = mat_to_quat(a[:3, :3]), mat_to_quat(b[:3, :3])
    if torch.dot(qa, qb) < 0:
        qb = -qb
    dot = torch.dot(qa, qb).abs().clamp(max=1.0)
    if dot > 0.9995:
        quat = F.normalize((1 - amount) * qa + amount * qb, dim=0)
    else:
        theta = torch.acos(dot)
        quat = (
            torch.sin((1 - amount) * theta) / torch.sin(theta) * qa
            + torch.sin(amount * theta) / torch.sin(theta) * qb
        )
    pose = torch.eye(4, device=a.device, dtype=a.dtype)
    pose[:3, :3] = quat_to_mat(quat)
    center_a, center_b = a[:3, 3], b[:3, 3]
    if bank.local_gaussians:
        # The shared rig is centred on the calibrated focus. A chord between
        # two cameras moves inward by ~5% at a 36-degree gap, creating periodic
        # zoom even when the physical cameras all have the same focal length.
        radius_a, radius_b = center_a.norm(), center_b.norm()
        direction_a, direction_b = center_a / radius_a, center_b / radius_b
        angle = torch.acos(torch.dot(direction_a, direction_b).clamp(-1, 1))
        if float(angle) > 1e-4:
            direction = (torch.sin((1-amount)*angle)*direction_a
                         + torch.sin(amount*angle)*direction_b) / torch.sin(angle)
        else:
            direction = F.normalize((1-amount)*direction_a + amount*direction_b, dim=0)
        pose[:3, 3] = direction * ((1-amount)*radius_a + amount*radius_b)
    else:
        pose[:3, 3] = (1 - amount) * center_a + amount * center_b
    ka, kb = bank.intrinsics[left, 0], bank.intrinsics[left + 1, 0]
    intrinsic = (1 - amount) * ka + amount * kb
    return pose, intrinsic, left, float(amount)


def _erode(tensor: torch.Tensor, kernel: int) -> torch.Tensor:
    pad = kernel // 2
    return -F.max_pool2d(-tensor[None, None], kernel, stride=1, padding=pad)[0, 0]


def _depth_guarded_fill(
    primary: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    helper: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fill only narrow primary holes with helper surfaces behind the edge."""
    color_p, alpha_p, depth_p = primary
    color_h, alpha_h, depth_h = helper
    eps = 1e-4
    z_primary = depth_p / alpha_p.clamp_min(eps)
    z_helper = depth_h / alpha_h.clamp_min(eps)
    valid = (alpha_p > args.primary_valid_alpha).float()
    kernel = args.depth_reference_kernel
    sum_depth = F.avg_pool2d(
        (z_primary * valid)[None, None], kernel, stride=1, padding=kernel // 2
    )[0, 0]
    count = F.avg_pool2d(
        valid[None, None], kernel, stride=1, padding=kernel // 2
    )[0, 0]
    z_reference = sum_depth / count.clamp_min(eps)
    helper_solid = _erode(alpha_h, args.helper_erode_kernel) > args.helper_alpha
    behind = z_helper > z_reference * (1 + args.depth_margin)
    gate = helper_solid & (count > 0) & behind & torch.isfinite(z_helper)

    amount = (
        (args.fill_alpha_high - alpha_p)
        / max(args.fill_alpha_high - args.fill_alpha_low, 1e-6)
    ).clamp(0, 1)
    amount = amount.square() * (3 - 2 * amount)
    amount = amount * gate.float()
    color = color_p * (1 - amount).unsqueeze(0) + color_h * amount.unsqueeze(0)
    alpha = alpha_p * (1 - amount) + alpha_h * amount
    fill = amount > 0.01
    return color, alpha, fill


def _spatial_wipe(
    old: tuple[torch.Tensor, torch.Tensor],
    new: tuple[torch.Tensor, torch.Tensor],
    progress: float,
    feather: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = old[0].shape[-1]
    x = torch.linspace(0, 1, width, device=old[0].device, dtype=old[0].dtype)
    feather = max(float(feather), 1e-4)
    threshold = float(progress) * (1 + 2 * feather) - feather
    amount = ((threshold - x) / feather + 0.5).clamp(0, 1)
    amount = amount.square() * (3 - 2 * amount)
    mask = amount.view(1, 1, width)
    color = old[0] * (1 - mask) + new[0] * mask
    alpha_mask = mask[0]
    alpha = old[1] * (1 - alpha_mask) + new[1] * alpha_mask
    return color, alpha


def _crop_resize(color: torch.Tensor, alpha: torch.Tensor, fraction: float):
    if fraction <= 0:
        return color, alpha
    height, width = alpha.shape
    dy, dx = round(height * fraction), round(width * fraction)
    color_crop = color[:, dy:height - dy, dx:width - dx]
    alpha_crop = alpha[dy:height - dy, dx:width - dx]
    color = F.interpolate(
        color_crop[None], size=(height, width), mode="bilinear", align_corners=False
    )[0]
    alpha = F.interpolate(
        alpha_crop[None, None], size=(height, width), mode="bilinear", align_corners=False
    )[0, 0]
    return color, alpha


def _display_region(color, alpha, args):
    color, alpha = _crop_resize(color, alpha, args.crop)
    if args.display_roi is not None:
        x0, y0, x1, y1 = args.display_roi
        height, width = alpha.shape
        # Even output sizes keep H.264 encoding deterministic, with no padding.
        left, top = round(x0 * width), round(y0 * height)
        right, bottom = round(x1 * width), round(y1 * height)
        right = left + ((right-left)//2)*2
        bottom = top + ((bottom-top)//2)*2
        color = color[:, top:bottom, left:right]
        alpha = alpha[top:bottom, left:right]
    return color, alpha


@torch.no_grad()
def render_position(bank: SparseAnchorBank, position: float, timestamp: float, args):
    target_c2w, target_k, left, amount = _interpolate_pose(bank, position)
    indices = [left, left + 1]
    selected = _gaussian_batches(bank.gaussians, indices)
    if args.composition_mode in {"pair_fused", "all_fused"}:
        fused = _merge_gaussian_batches(
            selected if args.composition_mode == "pair_fused" else bank.gaussians
        )
        target_time = None
        if fused.is_4d():
            target_time = target_c2w.new_full((1, 1), float(timestamp))
        output = bank.model.render(
            gaussians=fused,
            extrinsics=target_c2w.view(1, 1, 4, 4),
            intrinsics=target_k.view(1, 1, 3, 3),
            image_shape=(bank.height, bank.width),
            timestamps=target_time,
        )
        color, alpha = output.color[0, 0], output.alpha[0, 0]
        color, alpha = _display_region(color, alpha, args)
        display = color + float(args.background) * (1 - alpha).unsqueeze(0)
        primary = left if amount < 0.5 else left + 1
        return display.clamp(0, 1), alpha.clamp(0, 1), {
            "primary": primary,
            "helper": left + 1 if primary == left else left,
            "transition": False,
            "alpha_holes": float((alpha < 0.5).float().mean().item()),
            "left_fill": 0.0,
            "right_fill": 0.0,
        }
    target_ext = target_c2w.view(1, 1, 4, 4).expand(2, 1, 4, 4)
    if bank.local_gaussians:
        target_ext = torch.linalg.inv(bank.c2w[indices, :1]) @ target_ext
    target_intr = target_k.view(1, 1, 3, 3).expand(2, 1, 3, 3)
    target_time = None
    if selected.is_4d():
        target_time = target_c2w.new_full((2, 1), float(timestamp))
    output = bank.model.render(
        gaussians=selected,
        extrinsics=target_ext,
        intrinsics=target_intr,
        image_shape=(bank.height, bank.width),
        timestamps=target_time,
    )
    raw = [
        (output.color[i, 0], output.alpha[i, 0], output.depth[i, 0])
        for i in range(2)
    ]
    left_fill = _depth_guarded_fill(raw[0], raw[1], args)
    right_fill = _depth_guarded_fill(raw[1], raw[0], args)
    if bank.background_gaussians is not None:
        background = bank.model.render(
            bank.background_gaussians, target_c2w[None, None], target_k[None, None],
            (bank.height, bank.width),
        )
        bc, ba = background.color[0, 0], background.alpha[0, 0]
        def over_background(front):
            color, alpha, fill = front
            return (color + bc * (1-alpha)[None],
                    alpha + ba * (1-alpha), fill)
        left_fill, right_fill = over_background(left_fill), over_background(right_fill)

    gap = max(float(bank.axis_gaps_deg[left].item()), 1e-3)
    half_band = min(0.49, args.transition_degrees / (2 * gap))
    if args.composition_mode == "alpha_union":
        # Winner-takes-most coverage diagnostic.  The nearest source keeps all
        # covered foreground; its neighbour is admitted only where coverage is
        # weak.  This is less conservative than the depth gate and therefore
        # useful for deciding whether a sparse pair has enough real overlap.
        if amount < 0.5:
            base, helper, primary = left_fill, right_fill, left
        else:
            base, helper, primary = right_fill, left_fill, left + 1
        helper_weight = (
            (args.fill_alpha_high - base[1])
            / max(args.fill_alpha_high - args.fill_alpha_low, 1e-6)
        ).clamp(0, 1)
        helper_weight = helper_weight.square() * (3 - 2 * helper_weight)
        helper_weight *= (helper[1] > base[1] + 0.05).float()
        color = base[0] * (1 - helper_weight).unsqueeze(0) + helper[0] * helper_weight.unsqueeze(0)
        alpha = base[1] * (1 - helper_weight) + helper[1] * helper_weight
        transition = abs(amount - 0.5) <= half_band
    elif abs(amount - 0.5) <= half_band:
        progress = (amount - (0.5 - half_band)) / max(2 * half_band, 1e-6)
        progress = float(np.clip(progress, 0, 1))
        progress = progress * progress * (3 - 2 * progress)
        color, alpha = _spatial_wipe(
            (left_fill[0], left_fill[1]),
            (right_fill[0], right_fill[1]),
            progress,
            args.wipe_feather,
        )
        primary = left if progress < 0.5 else left + 1
        transition = True
    elif amount < 0.5:
        color, alpha, primary = left_fill[0], left_fill[1], left
        transition = False
    else:
        color, alpha, primary = right_fill[0], right_fill[1], left + 1
        transition = False

    color, alpha = _display_region(color, alpha, args)
    display = color + float(args.background) * (1 - alpha).unsqueeze(0)
    metadata = {
        "primary": primary,
        "helper": left + 1 if primary == left else left,
        "transition": transition,
        "alpha_holes": float((alpha < 0.5).float().mean().item()),
        "left_fill": float(left_fill[2].float().mean().item()),
        "right_fill": float(right_fill[2].float().mean().item()),
    }
    return display.clamp(0, 1), alpha.clamp(0, 1), metadata


def _to_u8(color: torch.Tensor) -> np.ndarray:
    return color.permute(1, 2, 0).mul(255).round().byte().cpu().numpy()


@torch.no_grad()
def run_reconstruction_check(bank: SparseAnchorBank, output_dir: Path) -> dict:
    times = bank.timestamps if bank.gaussians.is_4d() else None
    output = bank.model.render(
        gaussians=bank.gaussians,
        extrinsics=(torch.eye(4, device=bank.c2w.device).view(1,1,4,4).expand_as(bank.c2w)
                    if bank.local_gaussians else bank.c2w),
        intrinsics=bank.intrinsics,
        image_shape=(bank.height, bank.width),
        timestamps=times,
    )
    rendered = output.color + (1 - output.alpha).unsqueeze(2)
    difference = rendered.clamp(0, 1) - bank.images
    per_camera = -10 * torch.log10(
        difference.square().flatten(1).mean(1).clamp_min(1e-10)
    )
    overall = -10 * torch.log10(difference.square().mean().clamp_min(1e-10))
    valid_metrics = {}
    if bank.source_valid_mask is not None:
        valid = bank.source_valid_mask[:, :, None]
        squared_error = difference.square() * valid
        valid_mse = squared_error.sum() / (valid.sum() * 3).clamp_min(1)
        valid_psnr = -10 * valid_mse.clamp_min(1e-10).log10()
        per_camera_valid = -10 * (
            squared_error.flatten(1).sum(1)
            / (valid.flatten(1).sum(1) * 3).clamp_min(1)
        ).clamp_min(1e-10).log10()
        valid_metrics = {
            "valid_region_psnr": float(valid_psnr),
            "per_camera_valid_region_psnr": per_camera_valid.cpu().tolist(),
            "excluded_source_fraction": float(1 - valid.mean()),
        }
        print(f"[RECON] valid source region {float(valid_psnr):.2f} dB "
              f"(black source borders excluded; full-image metric below includes their removal)")
    holes = (output.alpha < 0.5).float().flatten(1).mean(1)
    print(f"[RECON] overall {overall.item():.2f} dB")
    for camera_id, psnr, hole in zip(bank.camera_ids, per_camera, holes):
        print(f"[RECON] cam{camera_id}: {psnr.item():.2f} dB, alpha holes {hole.item()*100:.2f}%")

    rows = []
    frame = bank.frames // 2
    for camera in range(bank.cameras):
        gt = bank.images[camera, frame]
        pred = rendered[camera, frame].clamp(0, 1)
        error = (gt - pred).abs().mul(4).clamp(0, 1)
        rows.append(np.concatenate([_to_u8(gt), _to_u8(pred), _to_u8(error)], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(output_dir / "recon_gt_render_error.png")
    return {
        **valid_metrics,
        "overall_psnr": float(overall.item()),
        "per_camera_psnr": {
            str(camera_id): float(value.item())
            for camera_id, value in zip(bank.camera_ids, per_camera)
        },
        "per_camera_alpha_holes": {
            str(camera_id): float(value.item())
            for camera_id, value in zip(bank.camera_ids, holes)
        },
    }


def _add_hud(
    frame: np.ndarray,
    bank: SparseAnchorBank,
    position: float,
    metadata: dict,
    ms: float,
    metric_label: str = "two-pass",
):
    primary_id = bank.camera_ids[metadata["primary"]]
    helper_id = bank.camera_ids[metadata["helper"]]
    mode = "WIPE" if metadata["transition"] else "ANCHOR"
    lines = [
        f"SPARSE CALIBRATED GS | B={bank.cameras}, V={bank.frames} | {mode}",
        f"path {position:.3f}/{bank.cameras-1} | primary cam{primary_id} | helper cam{helper_id}",
        f"{metric_label} {ms:.2f} ms ({1000/max(ms,1e-6):.1f} FPS) | alpha holes {metadata['alpha_holes']*100:.1f}%",
    ]
    for row, line in enumerate(lines):
        y = 24 + row * 22
        cv2.putText(frame, line, (11, y), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (11, y), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (245, 245, 245), 1, cv2.LINE_AA)


def run_recording(bank: SparseAnchorBank, args, output_dir: Path) -> dict:
    video_path = output_dir / "sparse_calibrated_two_pass.mp4"
    writer = imageio.get_writer(
        str(video_path), fps=args.fps, codec="libx264", quality=8, macro_block_size=2
    )
    render_ms, frames, frame_meta, frame_differences = [], [], [], []
    previous_color = None
    try:
        for index in range(args.render_frames):
            phase = index / max(args.render_frames - 1, 1)
            position = args.path_margin + phase * (
                bank.cameras - 1 - 2 * args.path_margin
            )
            timestamp = args.render_time
            if args.time_mode == "sweep" and bank.frames > 1:
                timestamp = phase
            torch.cuda.synchronize()
            started = time.perf_counter()
            color, _alpha, metadata = render_position(bank, position, timestamp, args)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            render_ms.append(elapsed_ms)
            if previous_color is not None:
                frame_differences.append(float((color - previous_color).abs().mean()))
            previous_color = color
            frame = np.ascontiguousarray(_to_u8(color))
            if not args.no_hud:
                _add_hud(frame, bank, position, metadata, elapsed_ms)
            writer.append_data(frame)
            frame_meta.append(metadata)
            if len(frames) < 12 and index >= round(len(frames) * (args.render_frames - 1) / 11):
                frames.append(frame.copy())
    finally:
        writer.close()

    if frames:
        columns = 4
        while len(frames) % columns:
            frames.append(np.zeros_like(frames[0]))
        sheet = np.concatenate([
            np.concatenate(frames[row:row + columns], axis=1)
            for row in range(0, len(frames), columns)
        ], axis=0)
        Image.fromarray(sheet).save(output_dir / "contact_sheet.png")
    values = np.asarray(render_ms)
    stats = {
        "two_pass_ms_median": float(np.median(values)),
        "two_pass_ms_p95": float(np.quantile(values, 0.95)),
        "two_pass_fps_median": float(1000 / np.median(values)),
        "mean_alpha_holes": float(np.mean([item["alpha_holes"] for item in frame_meta])),
        "frame_difference_p95": (float(np.quantile(frame_differences, .95))
                                 if frame_differences else 0.),
    }
    print(
        f"[RENDER] median {stats['two_pass_ms_median']:.2f} ms "
        f"({stats['two_pass_fps_median']:.1f} FPS), p95 {stats['two_pass_ms_p95']:.2f} ms"
    )
    print(f"[DONE] {video_path}")
    return stats


def _viewer_html(bank: SparseAnchorBank) -> bytes:
    ids = json.dumps(bank.camera_ids)
    temporal = "true" if bank.frames > 1 else "false"
    return f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Sparse calibrated live GS</title>
<style>body{{margin:0;background:#111;color:#eee;font:14px system-ui}}main{{max-width:1120px;margin:auto;padding:12px}}
#view{{width:100%;aspect-ratio:4/3;object-fit:contain;background:#222;cursor:grab}}.bar{{display:flex;gap:12px;align-items:center;margin:9px 0}}
input{{width:420px}}code{{color:#8fdaff}}</style></head><body><main><h2>稀疏标定相机 · 实时 Gaussian 双通道渲染</h2>
<div id=status>连接中…</div><img id=view draggable=false><div class=bar>视角 <input id=path type=range min=0 max=1 step=.0005 value=0><span id=pv>0</span></div>
<div class=bar>时间 <input id=time type=range min=0 max=1 step=.001 value=.5><span id=tv>.500</span></div>
<p>左键横拖为连续标定相机轨迹；画面由服务器当前驻留的 Gaussian 即时 rasterize，并非预渲染图片网格。相机：<code>{ids}</code></p>
<script>const im=document.querySelector('#view'),pa=document.querySelector('#path'),ti=document.querySelector('#time'),st=document.querySelector('#status'),pv=document.querySelector('#pv'),tv=document.querySelector('#tv');
const temporal={temporal};if(!temporal)ti.disabled=true;let busy=false,dirty=true,drag=false,last=0,url=null;
pa.oninput=()=>{{dirty=true;pv.textContent=(+pa.value).toFixed(3)}};ti.oninput=()=>{{dirty=true;tv.textContent=(+ti.value).toFixed(3)}};
im.onmousedown=e=>{{drag=true;last=e.clientX}};window.onmouseup=()=>drag=false;window.onmousemove=e=>{{if(!drag)return;pa.value=Math.max(0,Math.min(1,+pa.value+(e.clientX-last)/700));last=e.clientX;pa.oninput()}};
async function tick(){{if(dirty&&!busy){{dirty=false;busy=true;try{{let r=await fetch(`/frame?view=${{pa.value}}&time=${{ti.value}}`,{{cache:'no-store'}});if(!r.ok)throw Error(await r.text());let b=await r.blob(),u=URL.createObjectURL(b);im.onload=()=>{{if(url)URL.revokeObjectURL(url);url=u}};im.src=u;st.textContent=`GPU ${{r.headers.get('X-Render-Ms')}} ms | ${{r.headers.get('X-Render-Fps')}} FPS | holes ${{r.headers.get('X-Alpha-Holes')}}%`}}catch(e){{st.textContent=e}}finally{{busy=false}}}}requestAnimationFrame(tick)}}tick();</script>
</main></body></html>""".encode("utf-8")


def run_server(bank: SparseAnchorBank, args):
    html = _viewer_html(bank)
    lock = threading.Lock()
    for _ in range(2):
        render_position(bank, args.path_margin, args.render_time, args)
    torch.cuda.synchronize()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            if args.http_log:
                super().log_message(*_)

        def send_payload(self, payload, content_type, status=HTTPStatus.OK, headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                return self.send_payload(html, "text/html; charset=utf-8")
            if parsed.path == "/health":
                return self.send_payload(b"ok", "text/plain")
            if parsed.path != "/frame":
                return self.send_payload(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
            try:
                query = parse_qs(parsed.query)
                view = float(np.clip(float(query.get("view", [0])[0]), 0, 1))
                timestamp = float(np.clip(float(query.get("time", [args.render_time])[0]), 0, 1))
                position = args.path_margin + view * (bank.cameras - 1 - 2 * args.path_margin)
                with lock:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    color, _alpha, metadata = render_position(bank, position, timestamp, args)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                frame = cv2.cvtColor(_to_u8(color), cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                return self.send_payload(encoded.tobytes(), "image/jpeg", headers={
                    "X-Render-Ms": f"{elapsed*1000:.2f}",
                    "X-Render-Fps": f"{1/elapsed:.1f}",
                    "X-Alpha-Holes": f"{metadata['alpha_holes']*100:.1f}",
                })
            except Exception as error:
                return self.send_payload(
                    f"{type(error).__name__}: {error}".encode(),
                    "text/plain; charset=utf-8", HTTPStatus.BAD_REQUEST,
                )

    server = HTTPServer((args.host, args.port), Handler)
    print(f"[SERVER] http://{args.host}:{args.port}")
    if args.host in ("127.0.0.1", "localhost"):
        print(f"[SERVER] ssh -L {args.port}:127.0.0.1:{args.port} yang@lyg0168")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] stopped")
    finally:
        server.server_close()


def configure_model(args, device: torch.device) -> NoPo4D:
    model = NoPo4D.from_pretrained("bralani01/nopo4d").to(device).eval()
    model.encoder.gaussian_head.voxelizer = None
    model.encoder.gaussian_head.use_multiview_filter = False
    model.encoder.gaussian_head.use_bfloat16_dpt = args.bfloat16_gaussian_head
    model.decoder.background_color.zero_()
    model.decoder.max_gaussian_scale = args.max_gaussian_scale
    model.decoder.temporal_opacity_mode = "gaussian"
    model.decoder.max_sh_degree = args.sh_degree
    return model


@torch.no_grad()
def build_anchor_bank(
    args,
    device: torch.device,
    model: NoPo4D,
    depth_scale_override: torch.Tensor | None = None,
) -> tuple[SparseAnchorBank, dict]:
    camera_ids = args.camera_ids
    discovered = _discover_camera_ids(args.image_dir, len(camera_ids))
    if set(discovered) != set(camera_ids):
        raise ValueError(
            f"directory cameras are {discovered}, but --camera_ids is {camera_ids}"
        )
    # Physical path order need not match natural filename sorting (cam4 may
    # follow cam50 physically). Reorder images AND calibration together.
    camera_indices = [discovered.index(camera) for camera in camera_ids]
    calibrations = load_calibrations(args.calibration_json)
    images_flat, calibration = load_images(
        args.image_dir, device, len(camera_ids), args.max_long_edge,
        args.input_frames, camera_indices, args.frame_start, args.frame_stride, 0, calibrations,
    )
    _, views, _, height, width = images_flat.shape
    frames = views // len(camera_ids)
    w2c_flat, intr_flat = build_pose_inputs(
        calibration, height, width, frames, "w2c", False, "none", "pixel"
    )
    focus_override = tuple(args.focus_xyz) if args.focus_xyz is not None else None
    shared_w2c, shared_c2w, physical_focus, radius = _normalise_complete_rig(
        w2c_flat, len(camera_ids), frames, focus_override
    )
    images = images_flat.reshape(1, len(camera_ids), frames, 3, height, width)[0]
    intrinsics = intr_flat.reshape(len(camera_ids), frames, 3, 3).to(device)
    shared_w2c = shared_w2c.to(device)
    shared_c2w = shared_c2w.to(device)
    timestamps = None
    if frames > 1:
        timestamps = torch.linspace(0, 1, frames, device=device).repeat(len(camera_ids), 1)

    anchors, match_reports = None, []
    anchor_started = time.perf_counter()
    if args.depth_alignment == "triangulated" and depth_scale_override is None and args.encoding_mode == "isolated":
        features_flat, _ = load_images(
            args.image_dir, torch.device("cpu"), len(camera_ids),
            max(args.max_long_edge, 854), 1, camera_indices,
            args.frame_start, args.frame_stride, 0, calibrations,
        )
        feature_images = features_flat.reshape(len(camera_ids), 1, 3, *features_flat.shape[-2:])
        anchors, match_reports = triangulated_depth_anchors(
            images, shared_w2c, intrinsics, feature_images,
            workspace_only=args.anchor_region == "workspace",
        )
    anchor_initialisation_seconds = time.perf_counter() - anchor_started

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    if args.encoding_mode == "joint":
        encoded = model.forward(
            images.reshape(1, views, 3, height, width),
            timestamps=None if timestamps is None else timestamps.reshape(1, views),
            num_cameras=len(camera_ids),
            input_extrinsics=shared_w2c.reshape(1, views, 4, 4),
            input_intrinsics=intrinsics.reshape(1, views, 3, 3),
            sparse_local_gaussians=args.local_gaussians,
            sparse_inverse_depth_fit=args.inverse_depth_fit,
        )
        # Keep cross-view depth inference, but retain source Gaussian ownership
        # for rendering. Never fuse independent surfaces along the point axis.
        values = {}
        for field in fields(encoded.gaussians):
            value = getattr(encoded.gaussians, field.name)
            values[field.name] = None if value is None else value.reshape(len(camera_ids), -1, *value.shape[2:])
        encoded.gaussians = Gaussians(**values)
        for name in ("extrinsic_c2w", "extrinsic_w2c", "intrinsic"):
            value = encoded.camera_pose[name]
            encoded.camera_pose[name] = value.reshape(len(camera_ids), frames, *value.shape[2:])
        # Network rig normalization uses camera 0 as origin; a circular path
        # requires the same shared world translated to its estimated focus.
        poses = encoded.camera_pose["extrinsic_c2w"].clone()
        focus = estimate_camera_focus(poses[:, 0])
        poses[..., :3, 3] -= focus
        encoded.camera_pose["extrinsic_c2w"] = poses
        encoded.camera_pose["extrinsic_w2c"] = torch.linalg.inv(poses)
        scales = torch.ones(len(camera_ids), device=device)
    else:
        encoded = model.forward(
            images,
            timestamps=timestamps,
            num_cameras=1,
            input_extrinsics=shared_w2c,
            input_intrinsics=intrinsics,
            sparse_calibrated_batch=True,
            sparse_focus_world=torch.zeros(3, device=device),
            sparse_focus_roi=args.focus_roi,
            sparse_depth_scale_override=depth_scale_override,
            sparse_depth_anchors=anchors,
            sparse_local_gaussians=args.local_gaussians,
            sparse_inverse_depth_fit=args.inverse_depth_fit,
        )
        scales = model.encoder.last_sparse_depth_scale
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    memory_gb = torch.cuda.max_memory_allocated() / 2**30
    if args.disable_opacity_sh:
        encoded.gaussians.opacity_sh = None
    source_valid_mask = None
    if args.trim_source_border > 0:
        # Lens undistortion leaves black, unobserved pixels at the image rim.
        # Lifting those pixels turns the rim into curved black wires in 3D.
        # Only remove near-black components connected to the image boundary;
        # dark objects inside the scene remain intact.
        masks = []
        for camera in range(len(camera_ids)):
            for frame in range(frames):
                rgb = images[camera, frame].detach().cpu().permute(1, 2, 0).numpy()
                black = (rgb.max(-1) < 0.06).astype(np.uint8)
                n, labels = cv2.connectedComponents(black, connectivity=8)
                boundary_labels = np.unique(np.r_[labels[0], labels[-1], labels[:,0], labels[:,-1]])
                boundary_labels = boundary_labels[boundary_labels != 0]
                invalid = np.isin(labels, boundary_labels).astype(np.uint8)
                invalid = cv2.dilate(invalid, np.ones((3,3), np.uint8))
                # Do not follow a connected black component into dark trousers
                # or table legs. Limit inferred invalid pixels to the image rim.
                yy, xx = np.mgrid[:height, :width]
                rim = np.minimum.reduce([xx, yy, width-1-xx, height-1-yy]) < 8
                invalid *= rim.astype(np.uint8)
                border = args.trim_source_border
                invalid[:border] = invalid[-border:] = 1
                invalid[:,:border] = invalid[:,-border:] = 1
                masks.append(1 - invalid)
        keep = torch.from_numpy(np.stack(masks)).to(device=device, dtype=torch.float32)
        source_valid_mask = keep.reshape(len(camera_ids), frames, height, width)
        keep = keep.reshape(len(camera_ids), -1)
        encoded.gaussians.opacities *= keep

    forward_axes = shared_c2w[:, 0, :3, 2]
    gaps = torch.rad2deg(torch.acos(
        (forward_axes[:-1] * forward_axes[1:]).sum(-1).clamp(-1, 1)
    ))
    print(f"[RIG] cameras {camera_ids}")
    print(f"[RIG] physical focus {physical_focus.tolist()}, median radius {radius:.4f}")
    print(f"[RIG] adjacent optical-axis gaps {[round(x, 2) for x in gaps.cpu().tolist()]} deg")
    print(
        f"[FORWARD] {forward_seconds*1000:.2f} ms for {len(camera_ids)} cameras x "
        f"{frames} frame(s); peak CUDA {memory_gb:.2f} GiB"
    )
    bank = SparseAnchorBank(
        model=model,
        gaussians=encoded.gaussians,
        images=images,
        c2w=encoded.camera_pose["extrinsic_c2w"],
        intrinsics=encoded.camera_pose["intrinsic"],
        timestamps=timestamps,
        camera_ids=camera_ids,
        height=height,
        width=width,
        axis_gaps_deg=gaps,
        forward_seconds=forward_seconds,
        depth_scales=scales.clone(),
        local_gaussians=args.local_gaussians,
        source_valid_mask=source_valid_mask,
    )
    build_metrics = {
        "camera_ids": camera_ids,
        "input_frames": frames,
        "resolution": [width, height],
        "physical_focus": physical_focus.tolist(),
        "normalisation_radius": radius,
        "axis_gaps_deg": gaps.cpu().tolist(),
        "forward_seconds": forward_seconds,
        "peak_cuda_gib": memory_gb,
        "bfloat16_gaussian_head": args.bfloat16_gaussian_head,
        "depth_scales": scales.cpu().tolist() if args.encoding_mode == "isolated" else None,
        "encoding_mode": args.encoding_mode,
        "depth_alignment": args.depth_alignment,
        "anchor_region": args.anchor_region,
        "inverse_depth_fit": args.inverse_depth_fit,
        "match_reports": match_reports,
        "anchor_initialisation_seconds": anchor_initialisation_seconds,
        "depth_anchor_fits": model.encoder.last_sparse_anchor_residuals,
        "local_gaussians": args.local_gaussians,
        "sh_degree": args.sh_degree,
        "trim_source_border": args.trim_source_border,
        "transition_degrees": args.transition_degrees,
        "background_shell": args.background_shell,
        "display_roi": args.display_roi,
    }
    if args.background_shell:
        bank.background_gaussians = build_static_background(bank, args.workspace_radius)
    return bank, build_metrics


def run_stream_updates(
    first_bank: SparseAnchorBank,
    args,
    device: torch.device,
    model: NoPo4D,
    output_dir: Path,
) -> dict:
    """Refresh all camera anchors frame-by-frame while keeping scale locked."""
    if args.input_frames != 1:
        raise ValueError("--stream_updates currently requires --input_frames 1")
    writer = imageio.get_writer(
        str(output_dir / "sparse_stream_updates.mp4"),
        fps=args.fps, codec="libx264", quality=8, macro_block_size=2,
    )
    forward_ms = [first_bank.forward_seconds * 1000]
    try:
        for update in range(args.stream_updates):
            bank = first_bank
            if update:
                update_args = copy.copy(args)
                update_args.frame_start = args.frame_start + update * args.frame_stride
                bank, _ = build_anchor_bank(
                    update_args,
                    device,
                    model,
                    depth_scale_override=first_bank.depth_scales,
                )
                forward_ms.append(bank.forward_seconds * 1000)
            position = args.stream_view * (bank.cameras - 1)
            color, _alpha, metadata = render_position(
                bank, position, args.render_time, args
            )
            frame = np.ascontiguousarray(_to_u8(color))
            _add_hud(
                frame, bank, position, metadata, bank.forward_seconds * 1000,
                metric_label="forward refresh",
            )
            for _ in range(max(1, args.stream_hold_frames)):
                writer.append_data(frame)
    finally:
        writer.close()
    values = np.asarray(forward_ms)
    stats = {
        "updates": len(forward_ms),
        "forward_ms_median": float(np.median(values)),
        "forward_ms_p95": float(np.quantile(values, 0.95)),
        "update_fps_median": float(1000 / np.median(values)),
        "locked_depth_scales": first_bank.depth_scales.cpu().tolist(),
    }
    steady = values[2:] if len(values) > 2 else values
    stats.update({
        "steady_forward_ms_median": float(np.median(steady)),
        "steady_forward_ms_p95": float(np.quantile(steady, 0.95)),
        "steady_update_fps_median": float(1000 / np.median(steady)),
    })
    print(
        f"[STREAM] {len(values)} updates, median {stats['forward_ms_median']:.2f} ms "
        f"({stats['update_fps_median']:.2f} FPS), p95 {stats['forward_ms_p95']:.2f} ms"
    )
    print(
        f"[STREAM] steady state (drop first two): "
        f"{stats['steady_forward_ms_median']:.2f} ms "
        f"({stats['steady_update_fps_median']:.2f} FPS), "
        f"p95 {stats['steady_forward_ms_p95']:.2f} ms"
    )
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", type=Path, default=ROOT / "assets/custom_input1013")
    parser.add_argument("--calibration_json", type=Path, default=ROOT / "src/calibration_results-19-11-12_ByteDT.json")
    parser.add_argument("--camera_ids", type=_parse_ids, default=_parse_ids("38,40,42,44,46,48"))
    parser.add_argument("--input_frames", type=int, default=1)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_long_edge", type=int, default=504)
    parser.add_argument("--focus_xyz", type=float, nargs=3, default=None)
    parser.add_argument("--focus_roi", type=float, default=0.20)
    parser.add_argument("--encoding_mode", choices=("isolated", "joint"), default="isolated",
                        help="Joint calibrated attention can help new wide-FOV, overlapping rigs.")
    parser.add_argument("--depth_alignment", choices=("focus", "triangulated"), default="focus")
    parser.add_argument("--anchor_region", choices=("all", "workspace"), default="all")
    parser.add_argument("--inverse_depth_fit", action="store_true",
                        help="Validate a robust inverse-depth scale+shift against triangulated points.")
    parser.add_argument("--local_gaussians", action="store_true",
                        help="Keep all Gaussian attributes in the source-camera frame; transform targets instead.")
    parser.add_argument("--sh_degree", type=int, default=-1)
    parser.add_argument("--disable_opacity_sh", action="store_true")
    parser.add_argument("--background_shell", action="store_true",
                        help="Merge only static outer environment to cover disocclusion holes.")
    parser.add_argument("--workspace_radius", type=float, default=0.65,
                        help="Excluded work-area radius, as fraction of median camera radius.")
    parser.add_argument("--trim_source_border", type=int, default=0,
                        help="Remove invalid black lens-undistortion rim and this many border pixels.")
    parser.add_argument("--bfloat16_gaussian_head", action="store_true")
    parser.add_argument("--max_gaussian_scale", type=float, default=0.035)
    parser.add_argument("--fill_alpha_low", type=float, default=0.30)
    parser.add_argument("--fill_alpha_high", type=float, default=0.50)
    parser.add_argument("--primary_valid_alpha", type=float, default=0.72)
    parser.add_argument("--helper_alpha", type=float, default=0.72)
    parser.add_argument("--helper_erode_kernel", type=int, default=7)
    parser.add_argument("--depth_reference_kernel", type=int, default=11)
    parser.add_argument("--depth_margin", type=float, default=0.04)
    parser.add_argument("--transition_degrees", type=float, default=3.0)
    parser.add_argument("--wipe_feather", type=float, default=0.03)
    parser.add_argument(
        "--composition_mode",
        choices=("two_pass", "alpha_union", "pair_fused", "all_fused"),
        default="two_pass",
        help="two_pass is the safe default; the other modes are controlled alignment diagnostics.",
    )
    parser.add_argument("--path_margin", type=float, default=0.06)
    parser.add_argument("--crop", type=float, default=0.05)
    parser.add_argument("--display_roi", type=float, nargs=4, default=None,
                        metavar=("X0","Y0","X1","Y1"),
                        help="Fixed display-only crop after rendering; all cameras retain full input FOV.")
    parser.add_argument("--no_hud", action="store_true")
    parser.add_argument("--background", type=float, default=0.18)
    parser.add_argument("--render_frames", type=int, default=180)
    parser.add_argument("--render_time", type=float, default=0.5)
    parser.add_argument("--time_mode", choices=("fixed", "sweep"), default="fixed")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stream_updates", type=int, default=0,
                        help="Encode this many successive synchronized one-frame updates with one resident model.")
    parser.add_argument("--stream_view", type=float, default=0.5,
                        help="Normalised calibrated path position shown by the stream demo.")
    parser.add_argument("--stream_hold_frames", type=int, default=6)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "output_nas_sparse6_batch_stream")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--http_log", action="store_true")
    return parser.parse_args()


def _validate_args(args):
    if args.display_roi is not None:
        x0,y0,x1,y1 = args.display_roi
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("invalid display ROI")
    if args.inverse_depth_fit and (args.stream_updates or args.depth_alignment != "triangulated"):
        raise ValueError("inverse-depth fit requires triangulated anchors and currently no stream updates")
    if args.background_shell and (not args.local_gaussians or args.sh_degree != 0):
        raise ValueError("background shell requires --local_gaussians --sh_degree 0")
    if args.encoding_mode == "joint" and (not args.local_gaussians or args.stream_updates):
        raise ValueError("joint mode currently requires --local_gaussians and no --stream_updates")
    if not 0 <= args.trim_source_border <= 8:
        raise ValueError("--trim_source_border must be between 0 and 8")
    if not 0 <= args.crop < .5 or args.render_frames < 1:
        raise ValueError("invalid crop or render frame count")
    if args.transition_degrees <= 0:
        raise ValueError("--transition_degrees must be positive")
    if args.local_gaussians and args.composition_mode in {"pair_fused", "all_fused"}:
        raise ValueError("local Gaussian anchors cannot be concatenated in different camera frames")
    if args.input_frames < 1:
        raise ValueError("--input_frames must be positive")
    if not 0 < args.fill_alpha_low < args.fill_alpha_high < 1:
        raise ValueError("fill alpha thresholds must satisfy 0 < low < high < 1")
    for name in ("helper_erode_kernel", "depth_reference_kernel"):
        value = getattr(args, name)
        if value < 1 or value % 2 == 0:
            raise ValueError(f"--{name} must be a positive odd integer")
    if not 0 <= args.path_margin < (len(args.camera_ids) - 1) / 2:
        raise ValueError("--path_margin is outside the calibrated camera path")
    if args.stream_updates < 0 or not 0 <= args.stream_view <= 1:
        raise ValueError("invalid stream update settings")


def main():
    args = parse_args()
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA and gsplat are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = configure_model(args, device)
    bank, metrics = build_anchor_bank(args, device, model)
    metrics["reconstruction"] = run_reconstruction_check(bank, args.output_dir)
    if args.stream_updates:
        metrics["stream"] = run_stream_updates(
            bank, args, device, model, args.output_dir
        )
    if args.server:
        (args.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        run_server(bank, args)
    else:
        metrics["render"] = run_recording(bank, args, args.output_dir)
        (args.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
