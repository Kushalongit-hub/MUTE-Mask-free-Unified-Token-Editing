# Source-only mask review, before target generation

All eight source runs completed and their masks satisfy the frozen numerical area
validity checks. The masks and source PNGs were hashed before any target run.
The implementing assistant inspected the eight source overlays together.

The color masks generally track the red painted surfaces. They are not full-object
silhouettes: the second teapot's red handle is partly disconnected from the body
component; white highlights/rims and dark parts are missed; toy-car wheels/windows
are not completely included; backpack straps/hardware are imperfectly covered.
The second vase source includes flowers whose separate colored regions are not
part of its largest red body component. Those background/other-part changes will
still count outside the fixed vase surface mask.

All eight cases are retained. No mask, HSV threshold, source prompt, seed or
method parameter was corrected after review. Source-area validity is an algorithm
check, not a human confirmation of segmentation quality. Interpret colored-region
IoU/centroid metrics as painted-surface proxies. Full silhouette/part retention,
boundary quality and independent semantic success remain separate review tasks.

Source overlay contact sheet: `runs/evaluation_v1/source_mask_review.png`.
