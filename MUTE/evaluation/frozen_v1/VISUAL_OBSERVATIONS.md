# Implementing assistant observations (not independent ratings)

These observations follow target inspection. No frozen mask, method, prompt,
seed or metric threshold is changed. The independent review sheet remains blank.

Teapot, seed 2027: restoration with margin 2 preserves much of the background
but shifts/deforms the pot and produces a detached red handle segment plus an
extra yellow object. Its joint proxy fails. Other variants' full-object quality
must be judged from their saved outputs rather than these notes alone.

Teapot, seed 4099: restoration with margin 2 and the combined method both pass
the frozen colored-region joint proxy, but the handle is broken/detached and
partly red. This is a proxy false positive for a clean whole-object edit. The
source-only mask review already recorded that the handle was partly excluded
from the largest source-red component. Body overlap, color coverage and low
background MAE do not detect that part-structure failure. Do not report proxy
passes as validated semantic/geometry successes.

Toy car, seed 2027: margin-2 restoration yields a visibly green car and a similar
background, but its pose/size and wheel appearance change. Colored-region IoU is
below the fixed threshold. Its low background error is not a joint success.

Backpack, seed 2027: margin-2 restoration keeps a similar background but leaves
the backpack predominantly brown, with very little blue. There is a detached
strap-like fragment below the bench. The source is a red leather-style backpack;
the brown bag is the edited output, not a source failure. Source-prompt compliance
still belongs in independent review, separate from numerical mask validity.

Backpack, seed 4099: zero-margin and margin-2 restoration produce visibly blue
bags with similar backgrounds and pass the frozen joint proxy (as does combined).
The front pockets, straps/buckles and material/design differ from the source.
Colored-surface overlap therefore does not establish same-object part-structure
or identity retention. These require the separate paired-image review.

The completed contact sheets show extra small cars in the seed-2027 directional
and combined outputs. Both toy-car seeds change vehicle pose/design and wheel
appearance. A colored-surface/background summary cannot substitute for extra-object
or part-structure review.

Both vase cases turn predominantly green but change height/neck/body shape. The
zero-margin variants have edge/shape remnants and broken outlines. In seed 4099,
margin-2 and combined remove the source flowers. Those are scene-content changes
even when their contribution to global background MAE is small.
