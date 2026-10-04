import unittest
import torch
from mute.trajectory import SourceTrajectory, latent_active_mask, edit_patch_weights


class TrajectoryTests(unittest.TestCase):
    def test_dilation_feather_and_exact_identity(self):
        active = torch.zeros(49, dtype=torch.bool)
        active[24] = True
        weight = edit_patch_weights(active, margin=1, feather=1)
        self.assertEqual(int((weight == 1).sum()), 9)
        self.assertEqual(int((weight == .5).sum()), 16)
        self.assertEqual(int((weight == 0).sum()), 24)
        trajectory = SourceTrajectory()
        source = torch.full((1, 4, 7, 7), 2.)
        trajectory.record(0, 99, source)
        target = torch.full_like(source, 6.)
        trajectory.restore(0, 99, target, weight)
        expected = 2 + 4 * weight.reshape(1, 1, 7, 7).expand_as(target)
        self.assertTrue(torch.equal(target, expected))
        identity = source.clone()
        trajectory.restore(0, 99, identity, weight)
        self.assertTrue(torch.equal(identity, source))
        self.assertTrue(torch.equal(edit_patch_weights(active), active.float()))

    def test_restores_background_preserves_foreground(self):
        source = torch.arange(64).reshape(1, 4, 4, 4).float()
        trajectory = SourceTrajectory()
        trajectory.record(0, 99, source)
        source.zero_()  # stored reference is independent
        target = torch.full_like(source, -3)
        active = [True, False, False, False]
        trajectory.restore(0, 99, target, active)
        mask = latent_active_mask(active, target).expand_as(target)
        self.assertTrue(torch.equal(target[mask], torch.full_like(target[mask], -3)))
        self.assertTrue(torch.equal(target[~mask], trajectory.latents[0][~mask]))

    def test_schedule_mismatch_fails(self):
        trajectory = SourceTrajectory()
        source = torch.randn(1, 4, 4, 4)
        trajectory.record(0, 99, source)
        with self.assertRaises(ValueError):
            trajectory.restore(0, 98, source.clone(), [True, False, False, False])
        with self.assertRaises(ValueError):
            trajectory.record(2, 97, source)
        target = source + 1
        expected = target.clone()
        trajectory.restore(0, 99, target, [True, False, False, False], start=1)
        self.assertTrue(torch.equal(target, expected))


if __name__ == "__main__":
    unittest.main()
