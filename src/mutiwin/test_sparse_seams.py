"""Geometry and foreground-preservation checks; run with python -m unittest."""
import math
import unittest
from types import SimpleNamespace

import torch

from src.mutiwin.sparse_calibrated_stream import _depth_guarded_fill, _interpolate_pose
from src.model.encoder.encoder import EncoderNoPo4D
from src.model.encoder.types import SceneInput
from src.inference import align_calibrated_cameras_to_prediction, interpolate_calibrated_camera_path


class SparseSeamTests(unittest.TestCase):
    def test_heldout_camera_alignment_reuses_training_similarity(self):
        calibrated = torch.eye(4).repeat(3, 1, 1)
        calibrated[:, :3, 3] = torch.tensor([[0.,0.,0.], [2.,0.,0.], [0.,3.,0.]])
        prediction = calibrated.clone()
        prediction[:, :3, 3] = 2*calibrated[:, :3, 3] + torch.tensor([1.,2.,3.])
        target = torch.eye(4).repeat(2, 1, 1)
        target[:, :3, 3] = torch.tensor([[1.,1.,1.], [3.,2.,0.]])
        k = torch.eye(3).repeat(3,1,1)
        kt = torch.eye(3).repeat(2,1,1)
        ext, intr = align_calibrated_cameras_to_prediction(
            calibrated.inverse(), k, prediction[None], 1,
            target_w2c=target.inverse(), target_intrinsics=kt)
        torch.testing.assert_close(ext[0,:,:3,3], 2*target[:,:3,3] + torch.tensor([1.,2.,3.]))
        torch.testing.assert_close(intr[0], kt)

    def test_joint_camera_arc_and_fixed_focal_length(self):
        poses = torch.eye(4).repeat(1, 3, 1, 1)
        for i, angle in enumerate((-.6, 0., .6)):
            ca, sa = math.cos(angle), math.sin(angle)
            rot = torch.tensor([[ca, 0., sa], [0., 1., 0.], [-sa, 0., ca]])
            poses[0, i, :3, :3] = rot
            poses[0, i, :3, 3] = rot @ torch.tensor([0., 0., -2.])
        intr = torch.eye(3).repeat(1, 3, 1, 1)
        intr[0, :, 0, 0] = torch.tensor([300., 900., 300.])
        path, k = interpolate_calibrated_camera_path(poses, intr, 9, geometry="arc", reference_intrinsics=0)
        torch.testing.assert_close(path[0, :, :3, 3].norm(dim=-1), torch.full((9,), 2.))
        torch.testing.assert_close(path[0, ::4], poses[0])
        torch.testing.assert_close(k[0, :, 0, 0], torch.full((9,), 300.))
        old, _ = interpolate_calibrated_camera_path(poses, intr, 9)
        self.assertLess(float(old[0, 2, :3, 3].norm()), 2.)

    def test_depth_fit_keeps_all_batch_coordinates_and_validates_shift(self):
        batch, height, width = 3, 8, 9
        predicted = torch.linspace(.5, 2., width)[None, None, None].expand(batch, 1, height, width).clone()
        expected = (0.8 / predicted + 0.2).reciprocal()
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        anchors = torch.stack([xx.float(), yy.float(), expected[0,0]], -1).reshape(1,-1,3).repeat(batch,1,1)
        scene = SceneInput(image=torch.zeros(batch,1,3,height,width), num_cameras=1,
                           sparse_calibrated_batch=True, sparse_depth_anchors=anchors,
                           sparse_inverse_depth_fit=True)
        fake = SimpleNamespace(
            _predict_camera_poses=lambda *_: ({}, None),
            _predict_depth_maps=lambda *_: {"depth": predicted.clone(), "depth_conf": None},
            _sparse_focus_depth_scale=lambda *_: torch.ones(batch),
        )
        output = EncoderNoPo4D._run_heads(fake, scene, [], False)
        self.assertEqual(len(fake.last_sparse_anchor_residuals), batch)
        for fit in fake.last_sparse_anchor_residuals:
            self.assertEqual(fit["samples"], height*width)
            self.assertTrue(fit["inverse_affine_accepted"])
        torch.testing.assert_close(output.depth["depth"], expected, atol=1e-5, rtol=1e-5)

    def test_arc_keeps_radius_and_endpoints(self):
        poses = torch.eye(4).repeat(2, 1, 1)
        poses[0, :3, 3] = torch.tensor([0., 0., -1.])
        angle = math.radians(36)
        poses[1, :3, 3] = torch.tensor([math.sin(angle), 0., -math.cos(angle)])
        bank = SimpleNamespace(cameras=2, c2w=poses[:, None],
                               intrinsics=torch.eye(3).repeat(2, 1, 1, 1), local_gaussians=True)
        for amount in (0., .25, .5, .75, 1.):
            pose, _, _, _ = _interpolate_pose(bank, amount)
            self.assertAlmostEqual(float(pose[:3, 3].norm()), 1., places=6)
            if amount in (0., 1.):
                torch.testing.assert_close(pose, poses[int(amount)])

    def test_helper_cannot_replace_covered_foreground_or_front_surface(self):
        args = SimpleNamespace(primary_valid_alpha=.72, depth_reference_kernel=3,
                               helper_erode_kernel=1, helper_alpha=.72, depth_margin=.04,
                               fill_alpha_high=.5, fill_alpha_low=.3)
        alpha = torch.ones(9, 9)
        alpha[4, 4] = 0
        color = torch.zeros(3, 9, 9)
        primary = (color, alpha, alpha * 2)  # true depth=2, accumulated D=2A
        for helper_depth, expected_fill in ((1., False), (3., True)):
            helper = (torch.ones_like(color), torch.ones_like(alpha),
                      torch.full_like(alpha, helper_depth))
            result, result_alpha, fill = _depth_guarded_fill(primary, helper, args)
            torch.testing.assert_close(result[:, alpha >= .5], color[:, alpha >= .5])
            self.assertEqual(bool(fill[4, 4]), expected_fill)
            self.assertEqual(float(result_alpha[4, 4]), 1. if expected_fill else 0.)


if __name__ == "__main__":
    unittest.main()
