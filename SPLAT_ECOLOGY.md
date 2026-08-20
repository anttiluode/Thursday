# Gate 3 — Splat Ecology

**The weights are not scenery. They define the habitat.**

Gate 3 puts Thursday's information-metabolising creatures inside a frozen
`SplatWorld` decoder rather than a hand-authored grid.

SplatWorld already gives us an unusually literal learned world: a 128-D latent
point is decoded into 256 Gabor packets; near the training core those packets
phase-lock into face-like structure, while large-radius points become the
visible decorrelated "fire" between learned faces.

Gate 3 takes a reproducible 2-D ecological slice through that 128-D space:

```text
horizontal axis  = identity direction around a latent ring (wraps)
vertical axis    = latent radius |z|

bottom           = core / phase-locked habitat
middle           = ghost transition
upper            = fire / decorrelated habitat
```

Every map cell is actually passed through `splat_decoder.onnx`. The resulting
image is measured by three deliberately simple receivers:

- **FORM** — coarse spatial structure;
- **EDGE** — gradient energy;
- **CARRIER** — fine/high-frequency energy.

Those three measurements become the cell's renewable carrying capacities. The
world-map colors are therefore derived from the frozen learned decoder, not from
a hand-painted food map.

## The things that live there

Each creature has only:

- position in the 2-D learned habitat;
- energy;
- a three-component receptor/preference genome;
- a finite sensing budget (`2..8` nearby cells per tick);
- tiny HOLD memory of previously inspected local utility;
- age, generation and a short trail.

Fresh sensing costs energy. Movement costs energy. Regions where the learned
substrate changes rapidly cost a little extra. A creature consumes the resource
channels matching its receptors. Resources are depleted locally and regenerate
toward the carrying capacity supplied by the frozen model.

When a creature has enough energy it can reproduce. Offspring inherit mutated
resource preferences and sometimes a different information budget. Nothing is
trained by gradient descent while the ecology runs.

The point is not to claim "life inside CelebA." It is to make a much narrower
question visible:

> **Can learning performed for one purpose create a structured substrate that
> acts as a nontrivial habitat for independent information-limited processes?**

## Run

Gate 3 does **not** duplicate SplatWorld's 7.2 MB model. Point it at the original
`splat_decoder.onnx`:

```bash
python -m pip install -r requirements.txt
python splat_ecology.py --model E:\\path\\to\\SplatWorld\\splat_decoder.onnx
```

Or set:

```bash
set SPLATWORLD_MODEL=E:\\path\\to\\SplatWorld\\splat_decoder.onnx
python splat_ecology.py
```

The app also checks for `splat_decoder.onnx` in the current directory and a
sibling `SplatWorld/` directory. If none is found it boots a clearly-labelled
deterministic mock world so the life mechanics can still be inspected; click
**Load SplatWorld model** to switch to the real substrate.

The first real build decodes a `64 x 40` habitat and may take a little while on
CPU. It is cached under `.splat_ecology_cache/`; later launches reuse it.

## What to watch

The big left pane is the complete weight-world, which **you** are allowed to see.
The organisms are not.

- background color = the three model-derived carrying capacities;
- brightness loss = resources that have been eaten and not yet regenerated;
- white rings = the currently selected creature's paid sensory glimpses;
- white trail = where that creature has recently lived;
- creature color = its receptor genome;
- horizontal rules = the SplatWorld core/ghost/fire radii.

Click a creature. The upper-right pane decodes the actual SplatWorld image at
that creature's current 128-D position. If it walks upward, you should literally
watch its habitat move from phase-locked core toward fire.

The HOLD pane is intentionally parochial: black cells are unknown, remembered
cells fade as they age, and white boxes are what the creature paid to inspect on
this tick.

Now turn **price of information** upward. Some lineages should reduce their
lifetime usefulness simply because their inherited sensing budgets are too
expensive. Turn resource regeneration down and competition becomes visible.

## What would count as interesting

A colorful animation by itself proves nothing. The first receipts should ask:

1. **Trained-substrate control** — run exactly the same ecology on the real
   SplatWorld decoder, the same architecture with randomized weights if we can
   export one, and a shuffled resource field. Do population/niche statistics
   differ?
2. **Information-metabolism control** — remove sensing cost. Does selection on
   sensing budget disappear?
3. **Geometry control** — scramble cell locations while preserving the resource
   histogram. If the ecology behaves identically, we are using only resource
   abundance, not learned geometry.
4. **Fire/core migration** — do receptor lineages reproducibly occupy different
   SplatWorld shells, or is shell position irrelevant after matching resources?
5. **Inheritance** — does mutation generate stable receptor/sensing-budget
   niches, or merely noisy drift?

## Honest ledger

This is an artificial ecology. "FORM food" is a receiver measurement, not a
biological nutrient, and a 2-D slice is not the whole 128-D manifold. The useful
claim, if one survives controls, would be about **learned representation as
habitat**: structure acquired from a large training set can be reused as the
physics/carrying-capacity landscape for processes that were never part of the
original training objective.

That is the old dream in a form Thursday can actually kill.
