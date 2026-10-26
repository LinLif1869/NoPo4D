#!/usr/bin/env python3
"""Estimate image-pair overlap from geometrically verified feature coverage.

No model, GPU, calibration or camera selection required. See evaluate_camera_overlap.md.
Values measure matched-feature support, NOT exact shared visible surface area.
"""
import argparse
import csv
import itertools
import json
import re
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', path.name)]


def load_features(path, detector, max_edge):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    original_shape = image.shape[:2]
    scale = min(1., max_edge / max(original_shape))
    if scale < 1:
        image = cv2.resize(image, (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))), interpolation=cv2.INTER_AREA)
    keypoints, descriptors = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    return {'image': image, 'keypoints': keypoints, 'descriptors': descriptors,
            'original_shape': list(original_shape)}


def ratio_matches(a, b, ratio):
    if a is None or b is None or min(len(a), len(b)) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def selected(x, y):
        return {(pair[0].queryIdx, pair[0].trainIdx): pair[0]
                for pair in matcher.knnMatch(x, y, k=2)
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance}
    forward, backward = selected(a, b), selected(b, a)
    return [m for (i, j), m in forward.items() if (j, i) in backward]


def coverage(points, shape, grid):
    h, w = shape[:2]
    hull = cv2.convexHull(points.astype(np.float32))
    area = float(np.clip(cv2.contourArea(hull) / (h*w), 0, 1))
    cells = np.floor(points / np.array([w, h]) * grid).astype(int).clip(0, grid-1)
    occupied = len(np.unique(cells, axis=0)) / (grid*grid)
    return area, occupied, hull


def write_image(path, image):
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError(f'Cannot encode {path}')
    encoded.tofile(path)


def evaluate(a, b, args, visual_path):
    matches = ratio_matches(a['descriptors'], b['descriptors'], args.ratio)
    report = {'keypoints_a': len(a['keypoints']), 'keypoints_b': len(b['keypoints']),
              'mutual_ratio_matches': len(matches), 'inliers': 0, 'inlier_ratio': None,
              'coverage_a': None, 'coverage_b': None, 'coverage_mean': None,
              'coverage_min': None, 'grid_coverage_a': None, 'grid_coverage_b': None,
              'status': 'insufficient_matches', 'geometry': args.geometry}
    accepted = []
    canvases = [a['image'].copy(), b['image'].copy()]
    required = 8 if args.geometry == 'fundamental' else 4
    if len(matches) >= max(required, args.min_inliers):
        pa = np.float32([a['keypoints'][m.queryIdx].pt for m in matches])
        pb = np.float32([b['keypoints'][m.trainIdx].pt for m in matches])
        cv2.setRNGSeed(args.seed)
        if args.geometry == 'fundamental':
            matrix, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, args.ransac_px, .999, 10000)
        else:
            matrix, mask = cv2.findHomography(pa, pb, cv2.RANSAC, args.ransac_px, maxIters=10000, confidence=.999)
        if matrix is not None and mask is not None and np.isfinite(matrix).all():
            mask = mask.ravel().astype(bool)
            count = int(mask.sum())
            report.update(inliers=count, inlier_ratio=count/len(matches), status='weak_geometric_support')
            accepted = [m for m, keep in zip(matches, mask) if keep]
            if count >= args.min_inliers and count/len(matches) >= args.min_inlier_ratio:
                ca, ga, ha = coverage(pa[mask], a['image'].shape, args.grid)
                cb, gb, hb = coverage(pb[mask], b['image'].shape, args.grid)
                report.update(status='estimated', coverage_a=ca, coverage_b=cb,
                              coverage_mean=(ca+cb)/2, coverage_min=min(ca,cb),
                              grid_coverage_a=ga, grid_coverage_b=gb)
                for canvas, hull in zip(canvases, (ha, hb)):
                    overlay = canvas.copy()
                    cv2.fillConvexPoly(overlay, hull.astype(np.int32), (0, 180, 0))
                    cv2.addWeighted(overlay, .25, canvas, .75, 0, dst=canvas)
                    cv2.polylines(canvas, [hull.astype(np.int32)], True, (0,255,0), 2)
        else:
            report['status'] = 'geometry_failed'
    # Display only verified matches. If estimation fails, no area claim is made.
    shown = sorted(accepted, key=lambda m: m.distance)[:args.draw_matches]
    vis = cv2.drawMatches(canvases[0], a['keypoints'], canvases[1], b['keypoints'], shown, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    title = f"{report['status']} | matches {len(matches)} | inliers {report['inliers']}"
    if report['coverage_mean'] is not None:
        title += f" | hull A/B {report['coverage_a']:.1%}/{report['coverage_b']:.1%}"
    vis = cv2.copyMakeBorder(vis, 35, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(vis, title, (8,24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1, cv2.LINE_AA)
    write_image(visual_path, vis)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--image_dir', type=Path, help='One synchronized image per camera; all images in this directory are evaluated')
    source.add_argument('--images', nargs='+', type=Path, help='Explicit image paths, one synchronized frame per camera')
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--max_edge', type=int, default=1600)
    p.add_argument('--features', type=int, default=8000)
    p.add_argument('--ratio', type=float, default=.75)
    p.add_argument('--ransac_px', type=float, default=2., help='Threshold in resized-image pixels')
    p.add_argument('--min_inliers', type=int, default=20)
    p.add_argument('--min_inlier_ratio', type=float, default=.3)
    p.add_argument('--geometry', choices=['fundamental','homography'], default='fundamental')
    p.add_argument('--grid', type=int, default=12)
    p.add_argument('--draw_matches', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if min(args.max_edge, args.features, args.grid, args.draw_matches) <= 0 or args.min_inliers < 8:
        p.error('Counts must be positive; min_inliers must be >=8')
    if not 0 < args.ratio < 1 or not 0 < args.min_inlier_ratio <= 1 or not args.ransac_px > 0:
        p.error('Invalid ratio or RANSAC threshold')
    paths = args.images if args.images else sorted((x for x in args.image_dir.iterdir() if x.suffix.lower() in EXTENSIONS), key=natural_key)
    if len(paths) < 2 or len({x.resolve() for x in paths}) != len(paths):
        p.error('Provide at least two distinct images')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Use an empty output directory')
    detector = cv2.SIFT_create(nfeatures=args.features)
    data = [load_features(path, detector, args.max_edge) for path in paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / 'pairs'
    visual_dir.mkdir()
    matrix = np.full((len(paths), len(paths)), np.nan)
    np.fill_diagonal(matrix, 1.)
    results = []
    for i, j in itertools.combinations(range(len(paths)), 2):
        name = f'{i:03d}_{j:03d}.jpg'
        r = evaluate(data[i], data[j], args, visual_dir/name)
        r = {'index_a': i, 'index_b': j, 'image_a': str(paths[i]), 'image_b': str(paths[j]), **r, 'visualization': f'pairs/{name}'}
        results.append(r)
        if r['coverage_mean'] is not None:
            matrix[i,j] = matrix[j,i] = r['coverage_mean']
        value = 'unknown' if r['coverage_mean'] is None else f"{r['coverage_mean']:.1%}"
        print(f'{paths[i].name} <-> {paths[j].name}: {value}, {r["status"]}, inliers={r["inliers"]}')
    report = {'metric': 'SIFT mutual-ratio + RANSAC inlier convex-hull area coverage',
              'limitations': ['Not calibrated FOV overlap, not exact shared visible area or IoU.',
                             'Hull can span unmatched/occluded regions; textureless overlap can be missed.',
                             'Failed matching is unknown, not zero overlap. Use synchronized views.',
                             'Scores depend on texture, resolution, crop and thresholds. Inspect pair images.'],
              'parameters': {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k != 'images'},
              'images': [{'index': i, 'path': str(path.resolve()), 'original_hw': d['original_shape'],
                          'processed_hw': list(d['image'].shape[:2]), 'keypoints': len(d['keypoints'])}
                         for i,(path,d) in enumerate(zip(paths,data))], 'pairs': results}
    (args.output_dir/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (args.output_dir/'pairs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    with (args.output_dir/'coverage_matrix.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['image']+[f'{i}:{x.name}' for i,x in enumerate(paths)])
        for i,row in enumerate(matrix):
            writer.writerow([f'{i}:{paths[i].name}']+['' if not np.isfinite(x) else f'{x:.6f}' for x in row])
    print(f'Results: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
