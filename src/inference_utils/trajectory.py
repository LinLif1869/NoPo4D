"""Camera paths in model coordinates. OpenCV c2w: x right, y down, z forward."""
import numpy as np
import torch


def make_trajectory(cameras, intrinsics, depth, frames, mode='camera_path', order=None,
                    center=None, up=None, radius_scale=1.0, degrees=360.0, geometry='arc', cycle='pingpong',
                    intrinsics_mode='fixed', reference_camera=0):
    from scipy.spatial.transform import Rotation, Slerp
    cams = cameras.detach().float().cpu().numpy()
    ks = intrinsics.detach().float().cpu().numpy()
    reference_k = ks[reference_camera].copy()
    if order is not None:
        cams, ks = cams[order], ks[order]
    phase = np.arange(frames, dtype=np.float64) / frames
    poses = np.tile(np.eye(4), (frames, 1, 1))
    if mode == 'camera_path':
        if len(cams) < 2:
            raise ValueError('camera_path needs at least two cameras; use --trajectory orbit for one camera.')
        progress = np.linspace(0, 1, frames)
        if cycle == 'closed':
            route = np.concatenate([cams, cams[:1]])
            route_k = np.concatenate([ks, ks[:1]])
            u = phase * len(cams)
        else:
            route, route_k = cams, ks
            travel = 1-np.abs(2*progress-1) if cycle == 'pingpong' else progress
            u = travel*(len(cams)-1)
        knots = np.arange(len(route))
        poses[:, :3, :3] = Slerp(knots, Rotation.from_matrix(route[:, :3, :3]))(u).as_matrix()
        i = np.minimum(np.floor(u).astype(int), len(route)-2)
        amount = u-i
        origins = cams[:, :3, 3]
        if geometry == 'arc':
            if center is None:
                forward = cams[:, :3, 2]
                projectors = np.eye(3)[None]-forward[:, :, None]*forward[:, None, :]
                lhs = projectors.sum(0)
                if np.linalg.cond(lhs) > 1e5:
                    raise ValueError('Cannot infer stable arc center; supply --orbit_center or --path_geometry linear.')
                center = np.linalg.solve(lhs, np.einsum('nij,nj->i', projectors, origins))
                if np.any(np.sum((center-origins)*forward, axis=-1) <= 0):
                    raise ValueError('Estimated center lies behind a camera; provide --orbit_center or use linear path.')
            center = np.asarray(center, dtype=float)
            offsets = route[:, :3, 3]-center
            radii = np.linalg.norm(offsets, axis=-1)
            if np.any(radii < 1e-6):
                raise ValueError('Arc center coincides with a camera.')
            directions = offsets/radii[:, None]
            for j, (segment, t) in enumerate(zip(i, amount)):
                da, db = directions[segment:segment+2]
                dot = np.clip(da@db, -1, 1)
                if dot < -0.9999:
                    raise ValueError('Antipodal arc endpoints: add intermediate cameras or use linear path.')
                angle = np.arccos(dot)
                direction = ((1-t)*da+t*db if angle < 1e-5 else
                             (np.sin((1-t)*angle)*da+np.sin(t*angle)*db)/np.sin(angle))
                direction /= np.linalg.norm(direction)
                poses[j, :3, 3] = center+direction*((1-t)*radii[segment]+t*radii[segment+1])
        else:
            poses[:, :3, 3] = (1-amount[:,None])*route[i,:3,3]+amount[:,None]*route[i+1,:3,3]
        a = amount[:,None,None]
        out_k = (np.repeat(reference_k[None], frames, axis=0) if intrinsics_mode == 'fixed'
                 else (1-a)*route_k[i]+a*route_k[i+1])
        info = {'mode': mode, 'camera_order': order or list(range(len(cams))),
                'geometry': geometry, 'cycle': cycle, 'intrinsics_mode': intrinsics_mode,
                'reference_camera': reference_camera,
                'center': None if center is None else np.asarray(center).tolist()}
    else:
        origins, forward = cams[:, :3, 3], cams[:, :3, 2]
        if center is None:
            projectors = np.eye(3)[None] - forward[:, :, None]*forward[:, None, :]
            lhs = projectors.sum(0)
            if len(cams) > 1 and np.linalg.cond(lhs) < 1e5:
                center = np.linalg.solve(lhs, np.einsum('nij,nj->i', projectors, origins))
                if np.any(np.sum((center-origins)*forward, axis=-1) <= 0):
                    center = None
            if center is None:
                d = depth.detach().float().cpu().numpy()
                d = d[np.isfinite(d) & (d > 0)]
                if not d.size:
                    raise ValueError('Cannot infer orbit center: provide --orbit_center X Y Z.')
                center = origins[0] + forward[0]*np.median(d)
        center = np.asarray(center, dtype=float)
        up = np.asarray(up if up is not None else -cams[:, :3, 1].mean(0), dtype=float)
        if np.linalg.norm(up) < 1e-6:
            raise ValueError('Ambiguous up direction; provide --orbit_up X Y Z.')
        up /= np.linalg.norm(up)
        offset = origins[0] - center
        height = np.dot(offset, up)
        radial = offset - height*up
        radius = np.linalg.norm(radial)*radius_scale
        if radius < 1e-6:
            raise ValueError('Orbit radius is zero; change center or up.')
        x = radial / np.linalg.norm(radial)
        y = np.cross(up, x)
        angles = phase*np.deg2rad(degrees)
        positions = center + height*up + radius*(np.cos(angles)[:, None]*x + np.sin(angles)[:, None]*y)
        z = center-positions
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        right = np.cross(z, up)
        right /= np.linalg.norm(right, axis=-1, keepdims=True)
        down = np.cross(z, right)
        poses[:, :3, :3] = np.stack([right, down, z], axis=-1)
        poses[:, :3, 3] = positions
        out_k = np.repeat(reference_k[None], frames, axis=0)
        info = {'mode': mode, 'center': center.tolist(), 'up': up.tolist(), 'radius': float(radius), 'degrees': degrees}
    return (torch.as_tensor(poses, device=cameras.device, dtype=torch.float32)[None],
            torch.as_tensor(out_k, device=cameras.device, dtype=torch.float32)[None], info)
