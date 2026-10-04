# Coverage and boundary development

The preceding restoration runs missed mug handles and left red object edges.
Test restoration-only at seed 42 on the mug: patch margin 1, margin 2, and
margin 2 plus an outer feather ring of width 1. One patch is 16 output pixels
(two latent cells); a margin radius 2 expands the editable core up to 32 pixels
per side. Feather weights decrease across outer Chebyshev rings; width 1 is a
single ring with target weight 0.5. Nearest-neighbor upsampling stays fixed.

Keep all source conditioning, seed/noise, model, 30 steps, guidance 4.5, threshold
0.3 and capture settings unchanged. Restoration remains all 30 calls. Directional
attention is disabled to isolate coverage. Compare with the existing zero-margin
restoration-only baseline. Save original localization and expanded/graded weights.
Report background RGB MAE against the ORIGINAL source mask for every setting;
do not shrink the evaluated background by expanding the edit mask. Exact protected
latent checks apply to weight-zero cells; the old-mask background can now change.

Use seed-42 mug visual results to choose the smallest tested setting with an
apparently fully blue handle and improved rim/boundaries, without obvious extra
objects. If none qualifies, keep the failure and do not call it solved. Regardless,
test margin 2 plus feather 1 on seed-42 explicit apple and seed-123 mug as fixed
diagnostic transfers (not confirmation on held-out scenes). Audit exact source
capture and ordinary/restoration-only unchanged-prompt identities in every run.

These are post-pilot development settings, not independent masks or held-out
evaluation. Color coverage/boundary judgments are assistant observations, not
external scoring. Expanded regions can change background; the original fixed-mask
metric and full images must both remain visible. No novelty or directional gain
is inferred from this restoration-only sweep.

Development decision after inspecting the three seed-42 mug settings: margin 1
retains a visible red handle sliver. Margin 2 without feather meets the visual
selection criterion; feather 1 looks similar, with no established boundary gain.
Keep margin 2 / feather 0 as the smallest qualifying setting for this case.
Add a seed-123 mug transfer of that selected setting in a new output directory.
This is another development check on an already inspected seed, not held-out
confirmation. The two originally fixed feathered transfers remain in the record.
