# Gate 0 — first receipts

Date: 2026-08-20

Command:

```bash
python gate0_receipts.py --budget 4.0 --size 96 --repair-steps 80 --csv gate0_results.csv
```

These are mechanism checks on the built-in synthetic scene, not a benchmark claim.

## Fixed spectral budget

Lower receiver loss is better. Every policy has total gain budget `4.0` across ten radial frequency bands.

| Receiver | Uniform | Low-pass | High-pass | Adaptive | Result |
|---|---:|---:|---:|---:|---|
| coarse | 3.163690 | **0.000102** | 8.268346 | 0.001110 | simple low-pass wins |
| edges | 0.414969 | 0.069918 | 1.027854 | **0.050021** | adaptive wins these baselines |
| texture | 0.422104 | 0.267180 | 0.570583 | **0.134596** | adaptive wins these baselines |
| squares | 0.754098 | **0.058205** | 1.054719 | 0.063757 | low-pass slightly wins |

This is already a useful correction to the story. Receiver dependence exists in the demo, but **adaptivity is not automatically useful**. If a receiver's optimum is obvious from its frequency structure, a hand-designed filter can be better and cheaper.

The current adaptive optimizer is also expensive: one spectral update probes ten counterfactual band openings and may perform extra backtracking evaluations. That cost is deliberately not hidden.

## HOLD + local aperture

A receiver-specific contradiction is inserted in the top-right quarter after the spectral membrane has latched. The internal state is not reset. `peak aperture mean` is the largest fraction-like mean permeability over the whole image; `final` is after 80 repair steps.

| Receiver | Pre-change loss | Immediately after change | Final loss | Peak aperture mean | Final aperture mean |
|---|---:|---:|---:|---:|---:|
| coarse | 0.001110 | 0.439674 | 0.001412 | 0.050645 | 0.000134 |
| edges | 0.050021 | 0.315336 | 0.060807 | 0.038778 | 0.001068 |
| texture | 0.134596 | 0.734338 | 0.328480 | 0.051428 | 0.001646 |
| squares | 0.063757 | 0.116787 | 0.082541 | 0.004713 | 0.000493 |

The aperture now does what the picture says: **local contradiction opens a small region and permeability then recedes**. The earlier implementation failed this test because global feature normalization and ongoing spectral reconfiguration manufactured disagreement across the image; those were removed.

But the texture row matters. The aperture closes while receiver loss remains substantially above its pre-change value. That is not "the receiver no longer cares." It means:

> **the held state has assimilated the evidence the current membrane is capable of delivering, while the fixed spectral budget still prevents full receiver fidelity.**

So Thursday now has two distinct saturation notions that must not be conflated:

1. **transport saturation** — further opening of the current aperture has nothing new to deliver;
2. **receiver saturation/fidelity** — additional source information would no longer improve the downstream task.

A future version should test whether a *local* reallocation of spectral budget can close that gap without triggering global recomputation.

## Status

Gate 0 is **alive as a demo, unproven as an advantage**.

What survived:

- receiver swap changes fixed-budget spectral allocation;
- local external contradiction can trigger sparse spatial repair against persistent HOLD;
- the aperture closes again instead of remaining globally open;
- capacity limitation and repair completion can be observed separately.

What did not survive:

- "adaptive is always better" — false on the coarse and current squares receivers;
- the first localization rule — it spread globally and was replaced;
- the idea that aperture closure necessarily means receiver fidelity has recovered — false for texture at this budget.

The next serious gate is therefore not another prettier GUI. It is:

> **Does local receiver-driven spectral reallocation beat source-only allocation and simple task-specific filters once probe cost is charged?**
