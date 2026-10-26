"""Extended inference: camera-path MP4, full 4D state and diagnostics.

See src/inference_v2.md for conventions and server examples.
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import flow_to_image

from src.model.nopo4d import NoPo4D
from src.inference_utils.export import VideoWriter, cpu_fields
from src.inference_utils.trajectory import make_trajectory
from src.inference_utils.quality import reconstruction_report
from src.inference_utils.calibration import load_fixed_cameras

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image_dir', required=True, type=Path)
    p.add_argument('--num_cameras', required=True, type=int)
    p.add_argument('--output_dir', default=Path('output_v2'), type=Path)
    p.add_argument('--model', default='bralani01/nopo4d', help='HF model ID or from_pretrained-compatible directory')
    p.add_argument('--render_timestamps', type=int, help='Replay frames per input camera')
    p.add_argument('--manifest', type=Path, help='JSON list of {file, timestamp}, camera-major; file relative to image_dir')
    p.add_argument('--sort', choices=['natural', 'lexicographic'], default='natural')
    p.add_argument('--resize', nargs=2, type=int, metavar=('HEIGHT', 'WIDTH'), help='Explicit resize; both dimensions must be multiples of 14')
    p.add_argument('--calibration', type=Path, help='OpenCV extr.json; bypass camera prediction and use calibrated R,T,K')
    p.add_argument('--calibration_preprocessing', type=Path, help='Explicit source dimensions and crop boxes JSON')
    p.add_argument('--calibration_profile', help='Profile name in calibration_preprocessing, e.g. crop_short448')
    p.add_argument('--calibration_translation_scale', type=float, default=1.0,
                   help='Multiply calibrated T by this positive scale; does not align depth automatically')
    p.add_argument('--render_batch_size', type=int, default=1, help='Target views rendered at once; lower for limited VRAM')
    p.add_argument('--trajectory', choices=['camera_path', 'orbit'], default='camera_path')
    p.add_argument('--camera_order', nargs='+', type=int, help='Permutation of zero-based camera indices for path traversal')
    p.add_argument('--path_geometry', choices=['arc', 'linear'], default='arc')
    p.add_argument('--path_cycle', choices=['pingpong', 'closed', 'open'], default='pingpong')
    p.add_argument('--intrinsics_mode', choices=['fixed', 'interpolate'], default='fixed')
    p.add_argument('--reference_camera', type=int, default=0)
    p.add_argument('--video_frames', type=int, default=240)
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--time_mode', choices=['dynamic', 'freeze', 'pingpong'], default='dynamic')
    p.add_argument('--freeze_time', type=float, default=0.5)
    p.add_argument('--orbit_center', nargs=3, type=float)
    p.add_argument('--orbit_up', nargs=3, type=float)
    p.add_argument('--orbit_radius_scale', type=float, default=1.0)
    p.add_argument('--orbit_degrees', type=float, default=360)
    p.add_argument('--save_video_frames', action='store_true', help='Also preserve lossless orbit RGB PNGs')
    p.add_argument('--save_video_depth', action='store_true', help='Also preserve orbit depth/alpha float32 NPZ per frame')
    args = p.parse_args()
    if args.calibration:
        if not args.calibration_preprocessing or not args.calibration_profile:
            p.error('--calibration requires --calibration_preprocessing and --calibration_profile')
    elif args.calibration_preprocessing or args.calibration_profile or args.calibration_translation_scale != 1.0:
        p.error('Calibration options require --calibration')
    if not np.isfinite(args.calibration_translation_scale) or args.calibration_translation_scale <= 0:
        p.error('calibration_translation_scale must be finite and positive')
    for name in ('num_cameras', 'render_batch_size', 'video_frames', 'fps', 'orbit_radius_scale'):
        if getattr(args, name) <= 0:
            p.error(f'{name} must be positive')
    if args.render_timestamps is not None and args.render_timestamps <= 0:
        p.error('render_timestamps must be positive')
    if any(not 0 <= t <= 1 for t in [args.freeze_time]):
        p.error('freeze_time must be in [0,1]')
    if args.camera_order is not None and sorted(args.camera_order) != list(range(args.num_cameras)):
        p.error('camera_order must contain each camera index exactly once')
    if args.trajectory == 'camera_path' and args.num_cameras < 2:
        p.error('For a single camera select --trajectory orbit')
    if not 0 <= args.reference_camera < args.num_cameras:
        p.error('reference_camera is out of range')
    return args


def load_input(args, device):
    if args.manifest:
        entries = json.loads(args.manifest.read_text())
        paths = [args.image_dir / e['file'] for e in entries]
    else:
        paths = [p for p in args.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        key = (lambda p: [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', p.name)]) if args.sort == 'natural' else (lambda p: p.name)
        paths = sorted(paths, key=key)
    if not paths or len(paths) % args.num_cameras:
        raise ValueError('Image count must be nonzero and divisible by num_cameras.')
    frames = len(paths) // args.num_cameras
    timestamps = (torch.tensor([e['timestamp'] for e in entries], dtype=torch.float32)
                  if args.manifest else torch.linspace(0, 1, frames).repeat(args.num_cameras))
    ts = timestamps.reshape(args.num_cameras, frames)
    if not torch.isfinite(ts).all() or (ts < 0).any() or (ts > 1).any():
        raise ValueError('Manifest timestamps must be finite and normalized to [0,1].')
    if frames > 1 and not (ts[:, 1:] > ts[:, :-1]).all():
        raise ValueError('Timestamps must strictly increase within each camera.')
    if not torch.allclose(ts, ts[:1].expand_as(ts)):
        raise ValueError('Motion branch requires synchronized timestamps across cameras.')
    images = []
    for path in paths:
        with Image.open(path) as im:
            im = im.convert('RGB')
            if args.resize:
                h, w = args.resize
                if min(h, w) <= 0:
                    raise ValueError('Resize dimensions must be positive.')
                im = im.resize((w, h), Image.Resampling.LANCZOS)
            images.append(to_tensor(im))
    if len({tuple(im.shape) for im in images}) != 1:
        raise ValueError('Images differ in size; supply --resize HEIGHT WIDTH.')
    h, w = images[0].shape[-2:]
    if h % 14 or w % 14:
        raise ValueError(f'Input {h}x{w} is not divisible by DA3 patch size 14; use --resize.')
    return torch.stack(images)[None].to(device), timestamps[None].to(device), paths


def save_rgb(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(frame.detach().float().cpu().clamp(0, 1)).save(path)


def render_batches(model, gaussians, poses, intrinsics, times, shape, batch_size):
    for start in range(0, poses.shape[1], batch_size):
        end = start + batch_size
        result = model.render(gaussians, poses[:, start:end], intrinsics[:, start:end], shape, times[:, start:end])
        for j in range(result.color.shape[1]):
            yield start+j, result.color[0, j], result.depth[0, j], result.alpha[0, j]
        del result


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This entry point requires CUDA for gsplat rendering.')
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('Install ffmpeg with libx264 before running video export.')
    # Fail on missing trajectory dependency before loading expensive model weights.
    from scipy.spatial.transform import Slerp  # noqa: F401
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory is not empty: {out}. Choose a new directory to avoid mixing runs.')
    images, timestamps, paths = load_input(args, torch.device('cuda'))
    model_camera_kwargs, calibration_report = {}, None
    if args.calibration:
        model_camera_kwargs, calibration_report = load_fixed_cameras(
            args.calibration, args.calibration_preprocessing, args.calibration_profile,
            paths, args.num_cameras, images.shape[-2:], images.device,
            args.calibration_translation_scale)
    out.mkdir(parents=True, exist_ok=True)
    _, views, _, h, w = images.shape
    frames = views // args.num_cameras
    metadata = {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'image_shape': [h, w], 'input_files': [str(p.resolve()) for p in paths],
        'input_timestamps': timestamps[0].cpu().tolist(),
        'conventions': {'extrinsics': 'OpenCV c2w', 'intrinsics': 'pixel units',
                        'time': 'normalized, not seconds', 'world_scale': 'model reconstruction scale',
                        'layout': 'camera-major'},
        'status': 'started',
    }
    if calibration_report is not None:
        (out / 'calibration.json').write_text(json.dumps(calibration_report, indent=2) + '\n')
        metadata['calibration'] = calibration_report
        metadata['conventions']['world_scale'] = 'calibrated translation * calibration_translation_scale; depth scale unaligned'
    def write_metadata():
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    write_metadata()
    model = NoPo4D.from_pretrained(args.model).cuda().eval()
    model.decoder.chunk_size = args.render_batch_size
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    encoded = model(images, timestamps=timestamps, num_cameras=args.num_cameras, **model_camera_kwargs)
    torch.cuda.synchronize()
    metadata["encoder_seconds"] = time.perf_counter() - started
    metadata["encoder_peak_cuda_gib"] = torch.cuda.max_memory_allocated() / 2**30
    gs = encoded.gaussians
    if gs is None:
        raise RuntimeError('Encoder returned no Gaussians.')
    torch.save({'format_version': 1, 'gaussians': cpu_fields(gs)}, out / 'gaussians.pt')
    camera_data = {k: v.detach().cpu() for k, v in encoded.camera_pose.items() if isinstance(v, torch.Tensor)}
    torch.save({'camera_pose': camera_data, 'depth': {k: v.detach().cpu() for k, v in encoded.depth.items() if isinstance(v, torch.Tensor)},
                'optical_flow': {k: v.detach().cpu() for k, v in (encoded.optical_flow or {}).items() if isinstance(v, torch.Tensor)},
                'timestamps': timestamps.cpu()}, out / 'predictions.pt')
    np.savez_compressed(out / 'cameras.npz', **{k: v.numpy() for k, v in camera_data.items()}, timestamps=timestamps.cpu().numpy())
    for key, folder in [('motion_flow_fwd', 'flow_fwd'), ('motion_flow_bwd', 'flow_bwd')]:
        flow = (encoded.optical_flow or {}).get(key)
        if flow is not None:
            for i, item in enumerate(flow[0]):
                path = out / 'optical_flow' / folder / f'frame_{i:04d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                to_pil_image(flow_to_image(item.float().cpu())).save(path)
    metadata['quality'] = reconstruction_report(model, gs, encoded, images, timestamps, frames, out, args.render_batch_size)
    cameras = encoded.camera_pose['extrinsic_c2w'][0, ::frames]
    ks = encoded.camera_pose['intrinsic'][0, ::frames]
    replay_frames = args.render_timestamps or frames
    replay_ts = torch.linspace(timestamps.min().item(), timestamps.max().item(), replay_frames, device=images.device).repeat(args.num_cameras)[None]
    replay_poses = cameras.repeat_interleave(replay_frames, 0)[None]
    replay_ks = ks.repeat_interleave(replay_frames, 0)[None]
    (out / 'rendered_depth_alpha').mkdir()
    for i, rgb, depth, alpha in render_batches(model, gs, replay_poses, replay_ks, replay_ts, (h, w), args.render_batch_size):
        save_rgb(rgb, out / 'images' / f'view_{i:04d}.png')
        np.savez_compressed(out / 'rendered_depth_alpha' / f'view_{i:04d}.npz', depth=depth.cpu().numpy(), alpha=alpha.cpu().numpy())
    poses, intrinsics, path_info = make_trajectory(
        cameras, ks, encoded.depth['depth'][0, args.camera_order[0] * frames if args.camera_order else 0], args.video_frames, args.trajectory,
        args.camera_order, args.orbit_center, args.orbit_up, args.orbit_radius_scale, args.orbit_degrees,
        args.path_geometry, args.path_cycle, args.intrinsics_mode, args.reference_camera)
    phase = torch.linspace(0, 1, args.video_frames, device=images.device)
    if args.time_mode == 'freeze':
        video_ts = torch.full_like(phase, args.freeze_time)
    else:
        time_phase = 1-(2*phase-1).abs() if args.time_mode == 'pingpong' else phase
        video_ts = timestamps.min() + time_phase*(timestamps.max()-timestamps.min())
    video_ts = video_ts[None]
    np.savez_compressed(out / 'trajectory.npz', extrinsic_c2w=poses.cpu().numpy(), intrinsic=intrinsics.cpu().numpy(),
                        timestamps=video_ts.cpu().numpy(), image_shape=np.array([h, w]), fps=args.fps,
                        replay_timestamps=replay_ts.cpu().numpy())
    metadata['video'] = {'frames': args.video_frames, 'fps': args.fps,
                         'duration_seconds': args.video_frames / args.fps,
                         'input_frames_per_camera': frames, 'source_duration_seconds': None,
                         'scene_time_start': float(video_ts.min()), 'scene_time_end': float(video_ts.max())}
    metadata['trajectory'] = path_info
    write_metadata()
    video_holes = []
    with VideoWriter(out / 'camera_path.mp4', w, h, args.fps) as writer:
        for i, rgb, depth, alpha in render_batches(model, gs, poses, intrinsics, video_ts, (h, w), args.render_batch_size):
            video_holes.append(float((alpha < 0.5).float().mean()))
            writer.write(rgb)
            if args.save_video_frames:
                save_rgb(rgb, out / 'video_frames' / f'{i:06d}.png')
            if args.save_video_depth:
                folder = out / 'video_depth_alpha'
                folder.mkdir(exist_ok=True)
                np.savez_compressed(folder / f'{i:06d}.npz', depth=depth.cpu().numpy(), alpha=alpha.cpu().numpy())
            if i % 30 == 0:
                print(f'Video: {i+1}/{args.video_frames}', flush=True)
    metadata['video']['alpha_hole_fraction_per_frame'] = video_holes
    metadata['video']['alpha_hole_fraction_mean'] = float(np.mean(video_holes))
    metadata['status'] = 'complete'
    write_metadata()
    print(f'All results saved to {out.resolve()}')


if __name__ == '__main__':
    main()
