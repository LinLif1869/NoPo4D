#!/usr/bin/env python
"""Calibrated multi-window 4D renderer.

The windows stay independent: their Gaussians are never concatenated.  A
physical target camera is mapped into every window's reconstruction frame and
only the owning anchor (plus one neighbour inside a narrow seam) is rendered.
This preserves the clear single-source result while covering a wider arc.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference import (  # noqa: E402
    Scene,
    _calibration_matrices,
    align_calibrated_cameras_to_prediction,
    build_pose_inputs,
    estimate_camera_focus,
    generate_orbit_trajectory,
    interpolate_calibrated_camera_path,
    load_calibrations,
    load_images,
)
from src.model.nopo4d import NoPo4D  # noqa: E402
from src.mutiwin.multiwindow_utils import blend_weights  # noqa: E402


@dataclass
class Window:
    name: str
    scene: Scene
    physical_c2w: torch.Tensor
    physical_K: torch.Tensor
    rotation: torch.Tensor
    scale: torch.Tensor
    translation: torch.Tensor

    def camera_to_local(self, c2w: torch.Tensor) -> torch.Tensor:
        """Map calibrated-world c2w into this predicted/local scene."""
        out = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
        out[:3, :3] = self.rotation @ c2w[:3, :3]
        out[:3, 3] = self.scale * (self.rotation @ c2w[:3, 3]) + self.translation
        return out


@dataclass(frozen=True)
class Anchor:
    camera_id: int
    window: int
    source_index: int


def fit_world_to_scene(physical: torch.Tensor, predicted: torch.Tensor):
    """Fit the same orientation-aware Sim(3) used by inference.py."""
    rotations = predicted[:, :3, :3] @ physical[:, :3, :3].transpose(-1, -2)
    u, _, vh = torch.linalg.svd(rotations.mean(0))
    sign = torch.det(u @ vh)
    correction = torch.diag(torch.stack((sign.new_ones(()), sign.new_ones(()), sign)))
    rotation = u @ correction @ vh
    physical_centres = (rotation @ physical[:, :3, 3].T).T
    target_centres = predicted[:, :3, 3]
    x = physical_centres - physical_centres.mean(0)
    y = target_centres - target_centres.mean(0)
    scale = ((x * y).sum() / x.square().sum().clamp_min(1e-8)).abs().clamp_min(1e-6)
    translation = target_centres.mean(0) - scale * physical_centres.mean(0)
    fitted = scale * physical_centres + translation
    rmse = torch.linalg.vector_norm(fitted - target_centres, dim=-1).square().mean().sqrt()
    return rotation, scale, translation, rmse


def configure_model(device: torch.device) -> NoPo4D:
    model = NoPo4D.from_pretrained("bralani01/nopo4d").to(device).eval()
    model.encoder.gaussian_head.voxelizer = None
    model.encoder.gaussian_head.use_multiview_filter = False
    model.encoder.motion_encoder.motion_probability_threshold = 0.46
    model.decoder.temporal_opacity_mode = "constant"
    model.decoder.max_gaussian_scale = 0.05
    # Black makes decoder RGB premultiplied foreground.  The display background
    # is applied after alpha composition, so uncovered pixels cannot form a
    # baked-in white halo.
    model.decoder.background_color.zero_()
    return model


@torch.no_grad()
def reconstruct_window(
    model: NoPo4D,
    name: str,
    image_dir: Path,
    calibrations: dict,
    camera_ids: list[int],
    device: torch.device,
) -> Window:
    images, calib = load_images(
        image_dir, device, len(camera_ids), 504, 4, None, 0, 1, 0, calibrations,
    )
    _, views, _, height, width = images.shape
    frames = views // len(camera_ids)
    timestamps = torch.linspace(0, 1, frames, device=device).repeat(len(camera_ids)).unsqueeze(0)
    w2c, intr = build_pose_inputs(
        calib, height, width, frames, "w2c", True, "none", "pixel",
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    enc = model.forward(
        images,
        timestamps=timestamps,
        num_cameras=len(camera_ids),
        input_extrinsics=w2c.unsqueeze(0).to(device),
        input_intrinsics=intr.unsqueeze(0).to(device),
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    scene = Scene(
        model=model,
        gaussians=enc.gaussians,
        ext=enc.camera_pose["extrinsic_c2w"],
        intr=enc.camera_pose["intrinsic"],
        H=height,
        W=width,
        num_cameras=len(camera_ids),
        num_frames=frames,
        culling=True,
        source_time_mode="fixed",
        source_time_index=1,
    )
    # Retain inference.py's normal calibrated-camera diagnostics.
    scene.calibrated_ext, scene.calibrated_intr = align_calibrated_cameras_to_prediction(
        calib["w2c"], calib["K"], scene.ext[:, ::frames], frames,
    )
    physical_c2w = torch.linalg.inv(calib["w2c"][::frames].to(device))
    physical_K = calib["K"][::frames].to(device)
    rotation, scale, translation, rmse = fit_world_to_scene(
        physical_c2w, scene.ext[0, ::frames],
    )
    print(
        f"[WINDOW] {name}: {camera_ids}, forward={elapsed:.3f}s, "
        f"world->local scale={scale.item():.4f}, RMSE={rmse.item():.6f}"
    )
    return Window(
        name, scene, physical_c2w, physical_K, rotation, scale, translation,
    )


@torch.no_grad()
def render_anchor(
    window: Window,
    source_index: int,
    target_world_c2w: torch.Tensor,
    target_K: torch.Tensor,
    timestamp: float,
):
    local_c2w = window.camera_to_local(target_world_c2w)[None, None]
    out = window.scene.model.render(
        gaussians=window.scene.source_gaussians(source_index, 1),
        extrinsics=local_c2w,
        intrinsics=target_K[None, None],
        image_shape=(window.scene.H, window.scene.W),
        timestamps=torch.tensor([[timestamp]], device=local_c2w.device),
    )
    return out.color[0, 0].clamp(0, 1), out.alpha[0, 0].clamp(0, 1), out.depth[0, 0]


@torch.no_grad()
def render_local_anchor(
    window: Window,
    source_index: int,
    local_c2w: torch.Tensor,
    target_K: torch.Tensor,
    timestamp: float,
):
    out = window.scene.model.render(
        gaussians=window.scene.source_gaussians(source_index, 1),
        extrinsics=local_c2w[None, None],
        intrinsics=target_K[None, None],
        image_shape=(window.scene.H, window.scene.W),
        timestamps=torch.tensor([[timestamp]], device=local_c2w.device),
    )
    return out.color[0, 0].clamp(0, 1), out.alpha[0, 0].clamp(0, 1), out.depth[0, 0]


def crop_and_resize(tensor: torch.Tensor, crop: float) -> torch.Tensor:
    if crop <= 0:
        return tensor
    height, width = tensor.shape[-2:]
    dy, dx = round(height * crop), round(width * crop)
    cut = tensor[..., dy:height - dy, dx:width - dx]
    return torch.nn.functional.interpolate(
        cut[None], size=(height, width), mode="bilinear", align_corners=False,
    )[0]


def compose(
    results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    weights: np.ndarray,
    crop: float,
    background: float,
) -> tuple[np.ndarray, float]:
    """Alpha-aware blend of at most two aligned window renders."""
    rgb_sum = torch.zeros_like(results[0][0])
    alpha_sum = torch.zeros_like(results[0][1])
    for (rgb, alpha, _), weight in zip(results, weights):
        rgb_sum.add_(rgb * float(weight))
        alpha_sum.add_(alpha * float(weight))
    rgb_sum = crop_and_resize(rgb_sum, crop)
    alpha_sum = crop_and_resize(alpha_sum.unsqueeze(0), crop)[0].clamp(0, 1)
    display = rgb_sum + float(background) * (1 - alpha_sum).unsqueeze(0)
    frame = (
        display.clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()
    )
    frame = np.ascontiguousarray(frame)
    hole_fraction = float((alpha_sum < 0.5).float().mean().item())
    return frame, hole_fraction


def compose_wipe(
    old: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    new: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    progress: float,
    feather: float,
    crop: float,
    background: float,
) -> tuple[np.ndarray, float]:
    """Replace one anchor with the next using a narrow spatial feather.

    Unlike a full-frame cross dissolve, only the moving vertical seam mixes
    the two depth sheets, so a duplicated person cannot cover the whole image.
    Unlike the former fade-through-background transition, coverage never drops
    to zero and therefore cannot produce a black flash.
    """
    width = old[0].shape[-1]
    x = torch.linspace(0, 1, width, device=old[0].device, dtype=old[0].dtype)
    feather = max(float(feather), 1e-4)
    threshold = float(progress) * (1 + 2 * feather) - feather
    amount = ((threshold - x) / feather + 0.5).clamp(0, 1)
    amount = amount.square() * (3 - 2 * amount)  # smoothstep
    mask = amount.view(1, 1, width)
    rgb = old[0] * (1 - mask) + new[0] * mask
    alpha_mask = mask[0]
    alpha = old[1] * (1 - alpha_mask) + new[1] * alpha_mask
    return compose([(rgb, alpha, old[2])], np.ones(1), crop, background)


def physical_anchor_path(calibrations, camera_ids, width, height, samples, device):
    c2ws, intrinsics = [], []
    for camera_id in camera_ids:
        w2c, K = _calibration_matrices(calibrations[camera_id], width, height)
        c2ws.append(torch.linalg.inv(w2c))
        intrinsics.append(K)
    c2ws = torch.stack(c2ws).unsqueeze(0).to(device)
    intrinsics = torch.stack(intrinsics).unsqueeze(0).to(device)
    return interpolate_calibrated_camera_path(c2ws, intrinsics, samples)


def local_sector_paths(windows: list[Window], anchors: list[Anchor], samples: int, degrees: float):
    paths = []
    for anchor in anchors:
        scene = windows[anchor.window].scene
        camera_c2w = scene.ext[:, ::scene.num_frames]
        centre = estimate_camera_focus(camera_c2w[0])
        up = -camera_c2w[0, :, :3, 1].mean(0)
        path = generate_orbit_trajectory(
            camera_c2w[:, anchor.source_index], centre, up, samples, degrees,
        )
        K = scene.intr[:, anchor.source_index * scene.num_frames]
        paths.append((path, K))
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window_a", type=Path, default=ROOT / "assets/nas_cam40_64_65_41_t118_127_s3")
    parser.add_argument("--window_b", type=Path, default=ROOT / "assets/nas_window_b_42_66_67_43_t118_127_s3")
    parser.add_argument("--calibration_json", type=Path, default=ROOT / "src/calibration_results-19-11-12_ByteDT.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output_nas_realtime_multiwindow_4d/realtime_4d.mp4")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--transition_width", type=float, default=0.18,
                        help="Seam width in camera-index units; keep narrow to avoid ghosting.")
    parser.add_argument("--path_mode", choices=("sectors", "calibrated"), default="sectors",
                        help="sectors is the clean final mode; calibrated exposes low-overlap gaps for diagnosis.")
    parser.add_argument("--orbit_degrees", type=float, default=4.0)
    parser.add_argument("--sector_fade", type=float, default=0.18,
                        help="Half-width of the no-black spatial wipe around each sector boundary.")
    parser.add_argument("--wipe_feather", type=float, default=0.04,
                        help="Image-width fraction used to feather the moving wipe seam.")
    parser.add_argument("--crop", type=float, default=0.08)
    parser.add_argument("--background", type=float, default=0.18)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The realtime renderer requires CUDA/gsplat")
    if args.frames < 2 or args.fps < 1:
        raise ValueError("--frames must be >=2 and --fps must be positive")
    device = torch.device("cuda")
    calibrations = load_calibrations(args.calibration_json)
    model = configure_model(device)

    # Directory order is intentional and must match the four local source IDs.
    windows = [
        reconstruct_window(model, "A", args.window_a, calibrations, [40, 41, 64, 65], device),
        reconstruct_window(model, "B", args.window_b, calibrations, [42, 66, 67, 43], device),
    ]
    # Follow the dense physical chain inside each reconstruction window.  The
    # old demo kept only cam40/41/42/43, which threw away the helper cameras
    # after encoding and reintroduced a 9--17 degree pose jump at every sector
    # boundary.  Keeping all eight anchors makes the renderer a true sliding
    # window path: 40 -> 64 -> 65 -> 41 | 42 -> 66 -> 67 -> 43.  Only the
    # capture gap between cam41 and cam42 remains intrinsically sparse.
    anchors = [
        Anchor(40, 0, 0), Anchor(64, 0, 2),
        Anchor(65, 0, 3), Anchor(41, 0, 1),
        Anchor(42, 1, 0), Anchor(66, 1, 1),
        Anchor(67, 1, 2), Anchor(43, 1, 3),
    ]
    height, width = windows[0].scene.H, windows[0].scene.W
    path_samples = 301
    path_c2w, path_K = physical_anchor_path(
        calibrations, [a.camera_id for a in anchors], width, height,
        path_samples, device,
    )
    sector_paths = local_sector_paths(windows, anchors, path_samples, args.orbit_degrees)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.output), fps=args.fps, codec="libx264", quality=8, macro_block_size=2)
    render_seconds, hole_fractions, clean_hole_fractions, transition_frames = [], [], [], 0
    try:
        for frame_index in range(args.frames):
            phase = frame_index / max(args.frames - 1, 1)
            triangle = 1.0 - abs(2.0 * phase - 1.0)  # cam40 -> cam43 -> cam40
            last_anchor = len(anchors) - 1
            position = triangle * last_anchor
            path_index = int(round(position / last_anchor * (path_samples - 1)))
            timestamp = 0.5 - 0.5 * np.cos(2 * np.pi * phase)

            torch.cuda.synchronize()
            started = time.perf_counter()
            wipe_progress = None
            is_transition = False
            if args.path_mode == "calibrated":
                all_weights = blend_weights(
                    position, np.arange(4, dtype=np.float64), args.transition_width,
                )
                active = np.flatnonzero(all_weights > 1e-6)
                selected_weights = all_weights[active]
                selected_weights /= selected_weights.sum()
                results = []
                for anchor_index in active:
                    anchor = anchors[int(anchor_index)]
                    results.append(render_anchor(
                        windows[anchor.window], anchor.source_index,
                        path_c2w[0, path_index], path_K[0, path_index], timestamp,
                    ))
                owner_names = "+".join(f"cam{anchors[int(i)].camera_id}" for i in active)
            else:
                # Four reliable local orbits cover the wide range without ever
                # extrapolating a narrow-FOV source all the way to the next
                # physical camera.  At boundaries use a spatial wipe: it keeps
                # full coverage without globally mixing inconsistent sheets.
                sector_position = min(triangle * len(anchors), len(anchors) - 1e-6)
                fade = max(args.sector_fade, 1e-6)
                boundary = int(round(sector_position))
                if 1 <= boundary < len(anchors) and abs(sector_position - boundary) <= fade:
                    old_index, new_index = boundary - 1, boundary
                    old_anchor, new_anchor = anchors[old_index], anchors[new_index]
                    old_path, old_K = sector_paths[old_index]
                    new_path, new_K = sector_paths[new_index]
                    results = [
                        render_local_anchor(
                            windows[old_anchor.window], old_anchor.source_index,
                            old_path[0, -1], old_K[0], timestamp,
                        ),
                        render_local_anchor(
                            windows[new_anchor.window], new_anchor.source_index,
                            new_path[0, 0], new_K[0], timestamp,
                        ),
                    ]
                    wipe_progress = float(
                        (sector_position - (boundary - fade)) / (2 * fade)
                    )
                    is_transition = True
                    active = np.asarray([old_index, new_index])
                    owner_names = f"cam{old_anchor.camera_id}>cam{new_anchor.camera_id} wipe"
                else:
                    anchor_index = int(sector_position)
                    local_fraction = sector_position - anchor_index
                    local_index = int(round(local_fraction * (path_samples - 1)))
                    anchor = anchors[anchor_index]
                    local_path, local_K = sector_paths[anchor_index]
                    results = [render_local_anchor(
                        windows[anchor.window], anchor.source_index,
                        local_path[0, local_index], local_K[0], timestamp,
                    )]
                    active = np.asarray([anchor_index])
                    owner_names = f"cam{anchor.camera_id} local {args.orbit_degrees:g}deg"
                selected_weights = np.ones(len(results)) / len(results)
            torch.cuda.synchronize()
            render_seconds.append(time.perf_counter() - started)
            if wipe_progress is None:
                frame, hole_fraction = compose(
                    results, selected_weights, args.crop, args.background,
                )
            else:
                frame, hole_fraction = compose_wipe(
                    results[0], results[1], wipe_progress, args.wipe_feather,
                    args.crop, args.background,
                )
            hole_fractions.append(hole_fraction)
            if not is_transition:
                clean_hole_fractions.append(hole_fraction)
            else:
                transition_frames += 1

            cv2.putText(frame, f"view {position:4.2f}/{last_anchor}  {owner_names}", (14, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
            status = (f"alpha holes {hole_fraction*100:4.1f}%" if not is_transition
                      else "no-black sector wipe")
            cv2.putText(frame, f"4D time {timestamp:4.2f}  {status}", (14, 53),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
            writer.append_data(frame)
            if args.interactive:
                cv2.imshow("NoPo4D realtime multi-window 4D", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(max(1, round(1000 / args.fps))) & 0xFF in (27, ord("q")):
                    break
    finally:
        writer.close()
        if args.interactive:
            cv2.destroyAllWindows()

    values = np.asarray(render_seconds)
    median = float(np.median(values))
    p95 = float(np.quantile(values, 0.95))
    print(f"[BENCH] online splat: median={median*1000:.2f} ms ({1/median:.1f} FPS), "
          f"p95={p95*1000:.2f} ms, max two renders/frame")
    clean = clean_hole_fractions or hole_fractions
    print(f"[ALPHA] reliable-sector mean holes={np.mean(clean)*100:.2f}%, "
          f"max={np.max(clean)*100:.2f}%; no-black wipe frames={transition_frames}")
    print(f"[DONE] {args.output}")


if __name__ == "__main__":
    main()
