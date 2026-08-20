# Thursday

**Thursday is not a theory name. It is the record of one day spent making ideas pay rent.**

Thursday began on 20 August 2026 with a deliberate refusal to decide whether the interesting thing was a membrane, saturation, persistent consequence, a world model, or something else.

The rule was:

> **Mechanism → competitor → kill condition → receipt.**

By the end of the day the repo had become branch-shaped. `main` is only the map. Each branch preserves one experimental turn, including the mistakes that caused the next one.

## The arc

```text
finite information crossing a boundary
            │
            ▼
receiver-relative aperture + persistent HOLD
            │
            ▼
information itself has a metabolic cost
            │
            ▼
little agents survive by deciding what is worth sensing again
            │
            ▼
the boundary itself can remember / censor correction
            │
            ▼
learned neural weights become habitat for those agents
```

The day ended in a strange but concrete place:

> **Can learning performed for one purpose create a structured substrate that acts as a nontrivial habitat for independent, information-limited processes?**

No claim that the answer is yes has been earned yet. But the question is now runnable.

---

## Gate 0 — Breathing Information Aperture

**Branch:** [`gate0-breathing-aperture`](https://github.com/anttiluode/Thursday/tree/gate0-breathing-aperture)

A persistent internal image is repaired through a receiver-relative aperture under a fixed spectral budget.

What survived:

- different receivers pull the same finite spectral budget into different allocations;
- local contradiction can open a sparse repair aperture against persistent HOLD;
- the aperture can close again;
- transport saturation and receiver fidelity are not the same thing.

What failed:

- adaptivity is not automatically better than simple task-specific filters;
- the first localization rule spread globally;
- aperture closure does not prove that the receiver is fully satisfied.

This branch established the first useful Thursday vocabulary: **finite input budget, HOLD, receiver-relative consequence, and local repair**.

---

## Gate 1 — Tiny Lives / Information Metabolism

**Branch:** [`gate1-tiny-lives`](https://github.com/anttiluode/Thursday/tree/gate1-tiny-lives)

A small dynamic world contains creatures for which sensing is not free.

Three controls were separated:

- `reflex` — cheap local sensing, little persistent knowledge;
- `full` — observe everything and pay for it;
- `metabolism` — maintain HOLD and spend only a small fresh-information budget.

The important idea was not that one toy creature won one score. It was that **survival, task performance, and information expenditure form a frontier**.

The creature can live mostly from old internal state and buy a few carefully chosen corrections. That turned “attention” into an economic question:

> **Where is fresh reality worth its price?**

---

## Gate 2 — The Boundary Remembers

**Branch:** [`gate2-boundary-codec`](https://github.com/anttiluode/Thursday/tree/gate2-boundary-codec)

A custom `.thv` video experiment made the observation boundary itself stateful and inspectable.

It was not a useful video codec. That is part of the result.

The branch exposed two different kinds of temporal ghost that should not be confused:

1. **relaxation ghost** — residual history in a genuinely relaxing/adapting boundary;
2. **deadzone ghost** — a nonlinear self-sealing HOLD where the correction exists but the same gate that should admit it suppresses it.

In the implemented deadzone regime, when no residual crosses, reconstruction equals the predictor and the boundary update can become a fixed point. `rho` therefore does not control that ghost lifetime the way the first interpretation claimed.

The useful lesson was broader than compression:

> **A sparse system needs a theory of silence.**

“No update arrived” is not the same statement as “the old belief is still exactly true.” A held value needs provenance, uncertainty/validity bounds, and eventually a reason to probe again.

The Gate 2 branch is intentionally preserved as the machine that made this failure visible. Some contemporaneous comments in that branch reflect the earlier interpretation; this `main` README records the later reading.

---

## Gate 3 — Splat Ecology: Things Living in Learned Weights

**Branch:** [`gate3-splat-ecology`](https://github.com/anttiluode/Thursday/tree/gate3-splat-ecology)

This is where Thursday stopped using a hand-authored world.

The frozen [`SplatWorld`](https://github.com/anttiluode/SplatWorld) decoder was originally trained on faces. A 128-D latent point drives 256 Gabor packets; near the learned core they phase-lock into face-like structure, while farther out the familiar SplatWorld “fire” appears.

Gate 3 takes a reproducible 2-D slice through that learned latent world. Every habitat cell is actually decoded through the frozen model and measured by three simple receivers:

- **FORM** — coarse structure;
- **EDGE** — gradient structure;
- **CARRIER** — fine/high-frequency activity.

Those measurements become renewable carrying capacities.

Little creatures then live there with:

- energy;
- position in the learned habitat;
- inherited receptor preferences;
- finite sensing budgets;
- tiny persistent HOLD memories;
- reproduction, mutation and death.

The creatures do **not** see the full map shown by the GUI. They pay for a few local observations and act from those plus stale memory.

Clicking a creature decodes the actual SplatWorld image at its current 128-D position. From our point of view a lineage can therefore wander through face/core, ghost and fire. From the creature's point of view there are no faces at all — only a habitat whose economics were created by training done for an unrelated purpose.

This is the old dream in a testable form:

> **The weights are not scenery. They define the habitat.**

### The control that matters

The colorful ecology proves almost nothing by itself. The serious next comparison is the same life system on:

1. the trained SplatWorld substrate;
2. a spatially shuffled field with matched resource statistics;
3. a random/untrained decoder if a matched one is exported.

If niche structure, migration, specialization and sensing economics survive scrambling, then the learned weights were decorative. If reproducible ecological structure depends on the trained geometry, then “learned representation as habitat” becomes a real result.

---

## What Thursday seems to have found

Not a new grand theory.

A recurring systems problem:

> **A finite machine cannot continuously reacquire the world. It must live from persistent internal state, decide which differences deserve fresh information, and keep track of what silence does and does not certify.**

And one stranger extension:

> **A learned model can be treated not only as a function that produces outputs, but as a substrate whose learned geometry determines what is cheap, costly, stable, rich or barren for other processes living on top of it.**

The second statement is the one Gate 3 has made visible. It remains to be controlled.

## Why stop here

Because Thursday was supposed to resist the appetite machine.

There are many obvious next branches: safe HOLD, epistemic debt, receiver-aware event communication, ecological controls, organisms slowly modifying the substrate, communication between species, real model comparisons.

They are not a backlog.

The useful endpoint of this Thursday is already strange enough:

> A face model was trained, frozen, turned into geography, and populated by tiny information-metabolising processes that were never trained on faces.

Now the right thing is to measure whether the learned geography actually matters.

---

## Branch index

| Gate | Branch | Question |
|---|---|---|
| 0 | [`gate0-breathing-aperture`](https://github.com/anttiluode/Thursday/tree/gate0-breathing-aperture) | What information should cross for this receiver? |
| 1 | [`gate1-tiny-lives`](https://github.com/anttiluode/Thursday/tree/gate1-tiny-lives) | What if fresh information itself has a metabolic cost? |
| 2 | [`gate2-boundary-codec`](https://github.com/anttiluode/Thursday/tree/gate2-boundary-codec) | What does silence mean when the boundary remembers? |
| 3 | [`gate3-splat-ecology`](https://github.com/anttiluode/Thursday/tree/gate3-splat-ecology) | Can learned weights become habitat for unrelated living processes? |

---

**Thursday, 20 August 2026.**

Started with an aperture.

Ended with things living in a world made by learned weights.
