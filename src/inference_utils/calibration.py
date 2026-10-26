"""Load OpenCV w2c calibration and transform pixel intrinsics after crop/resize.

No pose estimation, coordinate normalization, or image undistortion is performed.
The explicit preprocessing JSON is the source of truth for crop coordinates.
"""
import json
import warnings

import numpy as np
import torch
from PIL import Image


def load_fixed_cameras(calibration_path, preprocessing_path, profile, paths,
                       num_cameras, image_shape, device, translation_scale=1.0):
    calibration = json.loads(calibration_path.read_text())
    preprocessing = json.loads(preprocessing_path.read_text())
    source_w, source_h = preprocessing['source_size_wh']
    spec = preprocessing['profiles'][profile]
    expected_size = tuple(spec['output_size_wh'])
    frames = len(paths) // num_cameras
    height, width = image_shape
    if not np.isfinite(translation_scale) or translation_scale <= 0:
        raise ValueError('calibration_translation_scale must be finite and positive')
    records, extrinsics, intrinsics, camera_ids = [], [], [], []
    for index in range(num_cameras):
        group = paths[index * frames:(index + 1) * frames]
        # The dataset uses CAMERAID_FRAME.png. Validate every frame, not just the first.
        ids = {p.stem.split('_', 1)[0] for p in group}
        if len(ids) != 1:
            raise ValueError(f'Camera block {index} mixes camera IDs: {sorted(ids)}')
        camera_id = ids.pop()
        if camera_id in camera_ids:
            raise ValueError(f'Duplicate camera block: {camera_id}')
        camera_ids.append(camera_id)
        entry = calibration[camera_id]
        for path in group:
            with Image.open(path) as image:
                if image.size != expected_size:
                    raise ValueError(f'{path}: size {image.size} does not match profile '
                                     f'{profile} ({expected_size}); check preprocessing JSON')
        k = np.asarray(entry['K'], dtype=np.float64).reshape(3, 3)
        r = np.asarray(entry['R'], dtype=np.float64).reshape(3, 3)
        t = np.asarray(entry['T'], dtype=np.float64).reshape(3)
        calib_w, calib_h = entry['imgSize']
        x0, y0, x1, y1 = spec['boxes_xyxy'][camera_id]
        if not (0 <= x0 < x1 <= source_w and 0 <= y0 < y1 <= source_h):
            raise ValueError(f'Invalid crop rectangle for {camera_id}')
        if not (np.isfinite(k).all() and np.isfinite(r).all() and np.isfinite(t).all()):
            raise ValueError(f'Non-finite calibration for {camera_id}')
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1, atol=1e-4):
            raise ValueError(f'R is not a proper rotation for {camera_id}')
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError(f'Invalid intrinsics for {camera_id}')
        # K_final = resize @ crop @ calibration_to_source @ K_calibration.
        # Use separate sx, sy: 448x602 is not an exactly isotropic resize.
        scale = np.diag([source_w / calib_w, source_h / calib_h, 1.0])
        crop = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.]])
        resize = np.diag([width / (x1-x0), height / (y1-y0), 1.0])
        pixel_transform = resize @ crop @ scale
        final_k = pixel_transform @ k
        if abs(final_k[0, 1]) > 1e-6 or abs(final_k[1, 0]) > 1e-6:
            raise ValueError('This model requires zero-skew intrinsics')
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = r, t * translation_scale
        extrinsics.append(w2c)
        intrinsics.append(final_k)
        records.append({'index': index, 'camera_id': camera_id,
                        'calibration_size_wh': [calib_w, calib_h],
                        'crop_xyxy': [x0, y0, x1, y1],
                        'pixel_transform': pixel_transform.tolist(),
                        'intrinsic': final_k.tolist(), 'extrinsic_w2c': w2c.tolist(),
                        'distCoeff': entry.get('distCoeff', [])})
    report = {
        'mode': 'fixed_calibration', 'extrinsics_convention': 'OpenCV w2c: Xcam = R Xworld + T',
        'translation_scale': translation_scale, 'world_origin': 'unchanged calibration origin',
        'depth_scale_alignment': 'none; predicted depth may not match calibration translation units',
        'backbone_camera_conditioning': False, 'camera_prediction_head': 'bypassed',
        'preprocessing_profile': profile, 'source_size_wh': [source_w, source_h],
        'final_size_wh': [width, height], 'undistortion': False,
        'cameras': records,
    }
    warnings.warn('Fixed calibration: depth remains in model scale. Calibration translation '
                  'units are not automatically aligned with predicted depth.', stacklevel=2)
    if any(np.any(np.asarray(record['distCoeff']) != 0) for record in records):
        warnings.warn('Input crops are not undistorted; geometry/rendering uses a pinhole '
                      'approximation. Nonzero distCoeff is recorded but not applied.', stacklevel=2)
    def tensor_per_frame(values):
        return torch.as_tensor(np.stack(values), dtype=torch.float32, device=device).repeat_interleave(frames, dim=0)[None]
    return {'fixed_extrinsics': tensor_per_frame(extrinsics),
            'fixed_intrinsics': tensor_per_frame(intrinsics)}, report
