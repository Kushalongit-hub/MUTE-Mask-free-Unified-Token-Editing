# Second development experiment: timing, strength, and semantic baseline

This follows the negative full-schedule hard-mask pilot. All settings are
development choices after viewing that pilot; this is not final evaluation.

Keep the pinned model, bf16 runtime, 512x512, 30 steps, guidance 4.5,
capture layers/calls, threshold 0.3, prompt pair and seed 42 unchanged initially.

Directional timing/strength factorial:

| Condition | Start call (zero-based) | Penalty |
|---|---:|---:|
| Previous hard full | 0 | infinity |
| Hard late | 15 | infinity |
| Soft full | 0 | 2 |
| Soft late | 15 | 2 |

Soft penalties subtract 2 from disallowed logits rather than eliminating the
edges. They do not multiply the resulting probabilities by a fixed amount after
normalization. From call 15 means the first 15 denoising calls remain original.

Each new condition includes an unchanged-source-prompt control. Ordinary
unchanged-prompt output must exactly match the captured source. Compare identity
disturbance and target edits separately. The soft-late run also retains symmetric
and reverse controls. If execution succeeds, repeat soft-late directional and
ordinary comparisons at seeds 17 and 123, without changing parameters.

Check cached source and target text embeddings differ and unconditional embeddings
match. Then compare ordinary conditioning at guidance 7.5 and 10. If the original
apple prompt still does not achieve a green apple, encode explicit green-skin,
red-to-blue ceramic mug, and red-to-blue cube development pairs. These are diagnostic
prompt changes, not repairs to a held-out benchmark. Re-encode the original pair
as a check against cached-conditioning corruption.

Save every run and settings, including failures. Background RGB MAE under inferred
masks remains diagnostic. Judge whether an edit actually happens and whether object
identity/shape/layout are preserved; do not treat tiny change from a failed edit as
an improvement. No reference masks, independent success labels, or significance
claims are available in this development phase.
