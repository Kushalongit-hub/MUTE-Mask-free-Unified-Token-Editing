# Source-trajectory restoration development

Record every post-scheduler source latent on CPU. During a target run, replace
background latent cells with the matching source step in-place after the target
scheduler update, starting at call 0 and continuing through call 29. Upsample the
fixed inferred 32x32 active patch mask to the 64x64 latent grid by nearest neighbor.
No dilation, feathering, RGB copying, or real-image inversion. The target DPM
solver keeps its own previous model-output history; this is post-step replacement,
not replacement of that history. Decoder spatial coupling may change pixels
outside the latent mask even if background latents are identical.

Use the pinned local PixArt checkpoint, 30 steps, guidance 4.5, bf16 runtime,
original capture layers/calls and threshold 0.3. Directional attention uses the
previous development choice: start 15, penalty 2. Compare four separate variants:
ordinary, directional only, restoration only, combined. Include unchanged-source
controls for each. Ordinary and restoration-only unchanged-prompt controls must
reproduce the source pixels exactly; combined identity can reveal disturbance
inside active cells and decoder spillover. Verify captured source equals original.

First run seed 42 mug and explicit green apple, then seed 42 cube. If successful,
repeat mug and explicit apple at seeds 17 and 123 without changing parameters.
Save trajectories, schedule, hashes, final background latent max-error and inferred
mask background pixel MAE. Inspect object color, geometry, boundaries and extra
objects separately. Reject nonfinite states and schedule mismatches. Any tuning
after viewing these results must receive a new protocol and output directory.

This is development, not held-out evaluation. These prompts were chosen after
the earlier sweep. Restoration is a separate preservation mechanism; gains from
it cannot be attributed to directional attention. No independent masks, success
annotations, prior-method comparison or novelty claim are established here.
