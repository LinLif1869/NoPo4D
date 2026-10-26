"""Input-view reconstruction diagnostics; not held-out novel-view evaluation."""
import json
import math
import torch
from torchvision.transforms.functional import to_pil_image


def reconstruction_report(model, gaussians, encoded, images, times, frames_per_camera, output, batch_size):
    folder = output / 'quality'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in range(0, images.shape[1], batch_size):
        stop = start + batch_size
        result = model.render(gaussians, encoded.camera_pose['extrinsic_c2w'][:, start:stop],
                              encoded.camera_pose['intrinsic'][:, start:stop],
                              tuple(images.shape[-2:]), times[:, start:stop])
        for j in range(result.color.shape[1]):
            i = start+j
            pred, gt, alpha = result.color[0,j].float(), images[0,i].float(), result.alpha[0,j].float()
            finite = bool(torch.isfinite(pred).all() & torch.isfinite(alpha).all())
            mse = float((pred-gt).square().mean()) if finite else None
            rows.append({'view': i, 'camera': i//frames_per_camera, 'frame': i%frames_per_camera,
                         'timestamp': float(times[0,i]), 'finite': finite, 'mse': mse,
                         'psnr_db': -10*math.log10(max(mse, 1e-10)) if finite else None,
                         'alpha_hole_fraction': float((alpha < .5).float().mean()) if finite else None})
            # One representative panel per camera, at an actual input timestamp.
            if i % frames_per_camera == frames_per_camera//2:
                panel = torch.cat([gt, pred, (pred-gt).abs()*4, alpha[None].expand(3,-1,-1)], dim=-1)
                to_pil_image(panel.nan_to_num().clamp(0,1).cpu()).save(folder / f'camera_{i//frames_per_camera:03d}_gt_render_error_alpha.png')
        del result
    def summarize(items):
        valid = all(r['finite'] for r in items)
        return {'finite': valid,
                'psnr_db': -10*math.log10(max(sum(r['mse'] for r in items)/len(items), 1e-10)) if valid else None,
                'alpha_hole_fraction': sum(r['alpha_hole_fraction'] for r in items)/len(items) if valid else None}
    report = {'scope': 'input-view reconstruction, not held-out evaluation',
              'alpha_hole_threshold': .5, 'error_display_multiplier': 4,
              'overall': summarize(rows),
              'per_camera': {str(c): summarize(rows[c*frames_per_camera:(c+1)*frames_per_camera])
                             for c in range(images.shape[1]//frames_per_camera)}, 'per_view': rows}
    (folder / 'reconstruction.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Input reconstruction:', report['overall'])
    return {'report': 'quality/reconstruction.json', **report['overall']}
