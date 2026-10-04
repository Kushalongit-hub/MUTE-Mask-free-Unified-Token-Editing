import unittest

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers import PixArtTransformer2DModel, PixArtAlphaPipeline, AutoencoderKL, DPMSolverMultistepScheduler

from mute.attention import AttentionStore, CaptureProcessor, DirectionalProcessor, directional_bias


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)

    def test_directions(self):
        active = torch.tensor([False, True, False])
        directional = directional_bias(active, "directional", torch.float32, "cpu")
        self.assertTrue(torch.isneginf(directional[0, 1]))
        self.assertEqual(directional[1, 0], 0)
        reverse = directional_bias(active, "reverse", torch.float32, "cpu")
        self.assertTrue(torch.equal(reverse, directional.T))
        symmetric = directional_bias(active, "symmetric", torch.float32, "cpu")
        self.assertTrue(torch.equal(symmetric, torch.minimum(directional, reverse)))

    def test_capture_uses_positive_branch_and_padding(self):
        attn = Attention(query_dim=4, cross_attention_dim=4, heads=1, dim_head=4)
        hidden, context = torch.randn(2, 4, 4), torch.randn(2, 3, 4)
        mask = torch.tensor([[[0., 0., -10000.]], [[0., 0., -10000.]]])
        store = AttentionStore([0], cfg=True, capture_steps=[0])
        store.add(attn, hidden, context, mask, 0, 0)
        q, k = attn.to_q(hidden[1]), attn.to_k(context[1])
        expected = ((q @ k.T) / 2 + mask[1]).softmax(-1)[:, 0]
        torch.testing.assert_close(store.maps[0], expected)

    def test_capture_preserves_output(self):
        attn = Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4)
        original = AttnProcessor2_0()
        hidden, context = torch.randn(2, 4, 8), torch.randn(2, 3, 8)
        expected = original(attn, hidden, context)
        actual = CaptureProcessor(original, AttentionStore([1], True, [0]), 0)(attn, hidden, context)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_background_independent_of_active_values(self):
        attn = Attention(query_dim=8, heads=2, dim_head=4)
        hidden = torch.randn(1, 4, 8)
        changed = hidden.clone()
        changed[:, 1] += 10
        proc = DirectionalProcessor(AttnProcessor2_0(), [False, True, False, False])
        a, b = proc(attn, hidden), proc(attn, changed)
        torch.testing.assert_close(a[:, [0, 2, 3]], b[:, [0, 2, 3]], rtol=0, atol=0)
        self.assertFalse(torch.equal(a[:, 1], b[:, 1]))

    def test_degenerate_localization_fails(self):
        store = AttentionStore([0])
        store.maps = [torch.ones(4)]
        with self.assertRaises(ValueError):
            store.localize()

    def test_schedule_delegates_outside_interval(self):
        attn = Attention(query_dim=8, heads=2, dim_head=4)
        hidden = torch.randn(1, 4, 8)
        original = AttnProcessor2_0()
        expected = original(attn, hidden)
        processor = DirectionalProcessor(original, [False, True, False, False],
                                         start_step=2, end_step=3, penalty=2.)
        for _ in range(2):
            torch.testing.assert_close(processor(attn, hidden), expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(processor(attn, hidden), expected))
        torch.testing.assert_close(processor(attn, hidden), expected, rtol=0, atol=0)
        processor.step = 0
        torch.testing.assert_close(processor(attn, hidden), expected, rtol=0, atol=0)

    def test_soft_penalty_reduces_but_does_not_remove_influence(self):
        attn = Attention(query_dim=8, heads=2, dim_head=4)
        with torch.no_grad():
            attn.to_q.weight.zero_()
        hidden = torch.randn(1, 4, 8)
        changed = hidden.clone()
        changed[:, 1] += 2
        original = AttnProcessor2_0()
        soft = DirectionalProcessor(original, [False, True, False, False], penalty=2.)
        ordinary_change = (original(attn, changed) - original(attn, hidden))[:, 0].norm()
        soft_change = (soft(attn, changed) - soft(attn, hidden))[:, 0].norm()
        self.assertGreater(float(soft_change.detach()), 0)
        self.assertLess(float(soft_change.detach()), float(ordinary_change.detach()))

    def test_tiny_pixart_integration(self):
        model = PixArtTransformer2DModel(num_attention_heads=2, attention_head_dim=8,
            in_channels=4, out_channels=8, num_layers=2, cross_attention_dim=16,
            sample_size=8, patch_size=2, caption_channels=16,
            norm_type="ada_norm_single", attention_bias=True).eval()
        hidden = torch.randn(2, 4, 8, 8)
        context = torch.randn(2, 5, 16)
        timestep = torch.tensor([10, 10])
        originals = dict(model.attn_processors)
        kwargs = dict(encoder_hidden_states=context, timestep=timestep,
                      added_cond_kwargs={"resolution": None, "aspect_ratio": None})
        with torch.no_grad():
            expected = model(hidden, **kwargs).sample
            store = AttentionStore([1], True, [0])
            model.set_attn_processor({key: CaptureProcessor(p, store, key) if "attn2" in key
                                     else DirectionalProcessor(p, None, "none")
                                     for key, p in originals.items()})
            actual = model(hidden, **kwargs).sample
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(len(store.maps), 2)
            _, active = store.localize()
            model.set_attn_processor({key: DirectionalProcessor(p, active) if "attn1" in key
                                     else p for key, p in originals.items()})
            edited = model(hidden, **kwargs).sample
            self.assertTrue(torch.isfinite(edited).all())
            self.assertFalse(torch.equal(edited, expected))

    def test_pipeline_without_text_encoder(self):
        model = PixArtTransformer2DModel(num_attention_heads=2, attention_head_dim=8,
            in_channels=4, out_channels=8, num_layers=2, cross_attention_dim=16,
            sample_size=8, patch_size=2, caption_channels=16,
            norm_type="ada_norm_single", attention_bias=True).eval()
        vae = AutoencoderKL(in_channels=3, out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(8, 8), latent_channels=4, norm_num_groups=4)
        pipe = PixArtAlphaPipeline(tokenizer=None, text_encoder=None, vae=vae,
                                  transformer=model, scheduler=DPMSolverMultistepScheduler())
        args = dict(prompt_embeds=torch.randn(1, 5, 16), prompt_attention_mask=torch.ones(1, 5),
                    negative_prompt_embeds=torch.randn(1, 5, 16),
                    negative_prompt_attention_mask=torch.ones(1, 5),
                    negative_prompt=None,
                    num_inference_steps=2, height=32, width=32, use_resolution_binning=False,
                    guidance_scale=4.5, output_type="np")
        pipe.set_progress_bar_config(disable=True)
        with torch.no_grad():
            expected = pipe(**args, generator=torch.Generator().manual_seed(42)).images
            store = AttentionStore([1], True, [0, 1])
            originals = dict(model.attn_processors)
            model.set_attn_processor({key: CaptureProcessor(p, store, key) if "attn2" in key else p
                                     for key, p in originals.items()})
            captured = pipe(**args, generator=torch.Generator().manual_seed(42)).images
        self.assertTrue((expected == captured).all())
        self.assertEqual(len(store.maps), 4)
        # Exercise the actual installed pipeline's post-scheduler callback:
        # full restoration must defeat changed conditioning and reach the
        # source decode exactly, proving in-place writes affect final latents.
        from mute.trajectory import SourceTrajectory
        model.set_attn_processor(dict(originals))
        trajectory = SourceTrajectory()
        with torch.no_grad():
            recorded = pipe(**args, generator=torch.Generator().manual_seed(42),
                            callback=trajectory.record, callback_steps=1).images
            changed_args = {**args, "prompt_embeds": args["prompt_embeds"] + 2}
            restored = pipe(**changed_args, generator=torch.Generator().manual_seed(42),
                callback=lambda step, t, latent: trajectory.restore(step, t, latent,
                    torch.zeros(16, dtype=torch.bool)), callback_steps=1).images
            identity = pipe(**args, generator=torch.Generator().manual_seed(42),
                callback=lambda step, t, latent: trajectory.restore(step, t, latent,
                    torch.tensor([True, False] * 8)), callback_steps=1).images
        self.assertEqual(len(trajectory.latents), 2)
        self.assertTrue((recorded == expected).all())
        self.assertTrue((restored == expected).all())
        self.assertTrue((identity == expected).all())


if __name__ == "__main__":
    unittest.main()
