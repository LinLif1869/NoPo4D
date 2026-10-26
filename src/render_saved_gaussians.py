"""Render an inference_v2 export without loading an encoder or predicting again."""
import argparse
from dataclasses import fields
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from torchvision.transforms.functional import to_pil_image

from src.model.types import Gaussians
from src.model.decoder.config import Decoder4DGSCfg
from src.model.decoder.decoder_4dgs import Decoder4DGS
from src.inference_utils.export import VideoWriter


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_dir', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--freeze_time', type=lambda s: float(Fraction(s)), default=2/3)
    p.add_argument('--render_batch_size', type=int, default=1)
    p.add_argument('--decoder_config', type=Path, default=Path(__file__).resolve().parents[1] / 'config/model/decoder/decoder_4dgs.yaml')
    p.add_argument('--save_video_frames', action='store_true')
    args = p.parse_args()
    if not 0 <= args.freeze_time <= 1 or args.render_batch_size < 1:
        p.error('freeze_time must be in [0,1] and render_batch_size must be positive')
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    source, out = args.source_dir, args.output_dir
    if source.resolve() == out.resolve() or (out.exists() and any(out.iterdir())):
        raise ValueError('Use a new empty output directory, separate from source_dir')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for gsplat rendering')
    meta = json.loads((source / 'metadata.json').read_text())
    with np.load(source / 'trajectory.npz', allow_pickle=False) as data:
        trajectory = {key: data[key].copy() for key in data.files}
    with np.load(source / 'cameras.npz', allow_pickle=False) as data:
        cameras = {key: data[key].copy() for key in data.files}
    poses, ks = trajectory['extrinsic_c2w'], trajectory['intrinsic']
    shape = tuple(int(x) for x in trajectory['image_shape'])
    fps = float(trajectory['fps'])
    if poses.ndim != 4 or poses.shape[0] != 1 or poses.shape[2:] != (4, 4):
        raise ValueError('Expected a single-scene trajectory [1,N,4,4]')
    if ks.shape != poses.shape[:2] + (3, 3) or poses.shape[1] == 0:
        raise ValueError('Trajectory intrinsics and extrinsics do not match')
    if not np.isfinite(poses).all() or not np.isfinite(ks).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid saved trajectory')
    ncam = int(meta['arguments']['num_cameras'])
    times = cameras['timestamps']
    if times.ndim != 2 or times.shape[0] != 1 or times.shape[1] % ncam:
        raise ValueError('Saved timestamps do not match camera-major layout')
    per_camera = times.shape[1] // ncam
    if per_camera == 0 or not np.isfinite(times).all():
        raise ValueError('Invalid input timestamps')
    # Match the exact float32 input timestamp used by the diagnostic panels.
    # torch.linspace(0,1,4)[2] can differ slightly from float32(2/3).
    nearest = float(times.flat[np.argmin(np.abs(times - args.freeze_time))])
    freeze_time = nearest if abs(nearest - args.freeze_time) < 1e-6 else args.freeze_time
    if not float(times.min()) <= freeze_time <= float(times.max()):
        raise ValueError('Requested time is outside the saved input time range')
    indices = [c * per_camera + int(np.argmin(np.abs(times[0, c*per_camera:(c+1)*per_camera] - freeze_time))) for c in range(ncam)]
    replay_poses = cameras['extrinsic_c2w'][:, indices]
    replay_ks = cameras['intrinsic'][:, indices]

    payload = torch.load(source / 'gaussians.pt', map_location='cpu', weights_only=True)
    if payload.get('format_version') != 1:
        raise ValueError('Expected inference_v2 gaussians.pt format_version=1')
    saved = payload['gaussians']
    allowed = {f.name for f in fields(Gaussians)}
    if set(saved) - allowed:
        raise ValueError(f'Unknown Gaussian fields: {set(saved) - allowed}')
    gs = Gaussians(**{key: value.cuda() if isinstance(value, torch.Tensor) else value
                      for key, value in saved.items()})
    del payload, saved
    if gs.means.shape[0] != 1:
        raise ValueError('Only single-scene Gaussian exports are supported')
    cfg = yaml.safe_load(args.decoder_config.read_text())
    cfg['chunk_size'] = args.render_batch_size
    decoder = Decoder4DGS(Decoder4DGSCfg(**cfg)).cuda().eval()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'input_views').mkdir()
    report = {
        'source_dir': str(source.resolve()), 'gaussians': str((source / 'gaussians.pt').resolve()),
        'encoder_rerun': False, 'requested_time': args.freeze_time, 'effective_time': freeze_time,
        'trajectory_reused_exactly': True, 'trajectory_K_reused_exactly': True,
        'frames': poses.shape[1], 'fps': fps, 'image_shape': shape, 'decoder_config': cfg,
        'input_view_indices': indices,
        'input_view_source_files': [meta['input_files'][i] for i in indices],
        'diagnostic_panel_times': [float(times[0, c*per_camera+per_camera//2]) for c in range(ncam)],
        'status': 'started',
    }
    def write_report():
        (out / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    write_report()
    trajectory['timestamps'] = np.full(poses.shape[:2], freeze_time, dtype=np.float32)
    if 'replay_timestamps' in trajectory:
        trajectory['replay_timestamps'] = np.full((1, ncam), freeze_time, dtype=np.float32)
    np.savez_compressed(out / 'trajectory.npz', **trajectory)
    np.savez_compressed(out / 'input_views.npz', extrinsic_c2w=replay_poses,
                        intrinsic=replay_ks, timestamps=np.full((1, ncam), freeze_time, dtype=np.float32))

    def render(camera_poses, intrinsics):
        for start in range(0, camera_poses.shape[1], args.render_batch_size):
            stop = start + args.render_batch_size
            pose = torch.as_tensor(camera_poses[:, start:stop], device='cuda', dtype=torch.float32)
            k = torch.as_tensor(intrinsics[:, start:stop], device='cuda', dtype=torch.float32)
            ts = torch.full(pose.shape[:2], freeze_time, device='cuda', dtype=torch.float32)
            result = decoder(gs, pose, k, shape, ts=ts)
            for j in range(result.color.shape[1]):
                yield start+j, result.color[0,j], result.depth[0,j], result.alpha[0,j]
            del result

    for i, rgb, depth, alpha in render(replay_poses, replay_ks):
        to_pil_image(rgb.float().cpu().clamp(0,1)).save(out / 'input_views' / f'camera_{i:03d}.png')
        np.savez_compressed(out / 'input_views' / f'camera_{i:03d}_depth_alpha.npz',
                            depth=depth.cpu().numpy(), alpha=alpha.cpu().numpy())
    if args.save_video_frames:
        (out / 'video_frames').mkdir()
    holes = []
    with VideoWriter(out / 'camera_path.mp4', shape[1], shape[0], fps) as writer:
        for i, rgb, depth, alpha in render(poses, ks):
            writer.write(rgb)
            holes.append(float((alpha < .5).float().mean()))
            if args.save_video_frames:
                to_pil_image(rgb.float().cpu().clamp(0,1)).save(out / 'video_frames' / f'{i:06d}.png')
            if i % 30 == 0:
                print(f'Render: {i+1}/{poses.shape[1]} at t={freeze_time:.9f}', flush=True)
    report.update(status='complete', alpha_hole_fraction_per_frame=holes)
    write_report()
    print(f'Saved frozen-time renders to {out.resolve()}')


if __name__ == '__main__':
    main()
