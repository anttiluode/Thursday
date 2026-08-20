# Thursday

**Thursday is not a theory name. It is a place to make a thought pay rent.**

This repository starts on Thursday, 20 August 2026, deliberately without choosing whether the underlying story is "membrane", "kyllästyspiste", "persistent consequence computation", "world model", or something else.

The rule is simpler:

> Promote an idea here only when it becomes a mechanism with a competitor, a measurement, and a way to die.

## Gate 0 — The breathing information aperture

The first object grows out of `sigh_image`: an image crosses a radial Fourier equalizer before a downstream structure detector sees it. The old equalizer is visually suggestive, but *we* choose its gains.

Gate 0 removes that privilege.

```text
                 fixed spectral budget
                         │
WORLD ──► [ adaptive spectral membrane ] ──► candidate evidence
                         ▲                         │
                         │                         ▼
                  marginal utility          persistent HOLD
                         │                         │
                         └──── RECEIVER ◄─────────┘
                                  │
                           local contradiction
                                  │
                                  ▼
                         spatial aperture
                       opens → repairs → closes
```

A named receiver (`coarse`, `edges`, `texture`, or `squares`) measures what it can still distinguish. The membrane estimates the marginal task improvement from opening each frequency band a little further, then reallocates a **fixed total permeability budget**. Change receivers and the same world should pull the permeability curve into a different shape.

The internal image is persistent. No current evidence is not treated as evidence that the world disappeared. When the external world changes, receiver-visible disagreement opens a **local aperture**. Evidence is admitted there; once the held state has been repaired enough that the receiver stops caring, the disagreement falls and the aperture closes.

There are deliberately two timescales. Global spectral allocation adapts after a receiver/budget change and then **latches when further reallocation no longer measurably improves that receiver**. Ordinary local surprise does not reopen global learning; it is repaired through the currently latched membrane. Otherwise the membrane would manufacture disagreement everywhere by changing its own encoding while trying to localize an external change.

That last loop is the point of the demo.

## What this is *not* claiming

None of the components are individually new.

- Sensory adaptation and efficient coding have long studied how finite dynamic range is reallocated as stimulus statistics change.
- Rate–distortion, information bottleneck, semantic communication, and video/image coding for machines all formalize versions of "preserve what matters to a receiver/task rather than every source bit."
- Tracking, filtering, predictive coding, active sensing, and world models all have machinery for persistent state and innovation.

So the claim is deliberately narrower:

> **Can live marginal utility at a particular receiver control what information crosses a boundary, while persistent state turns missing evidence into HOLD and receiver-visible contradiction into a temporary local increase in permeability?**

The interesting object, if there is one, is the *closed loop* rather than any noun we attach to it.

## Run

```bash
python -m pip install -r requirements.txt
python breathing_aperture.py
```

The app starts with a synthetic image containing both coarse and fine structure. You can also load your own image.

Try this sequence:

1. Choose `coarse`, click **Auto breathe**, and watch the fixed spectral budget drift toward bands useful to coarse structure.
2. Switch to `texture` or `edges`. The yellow marginal-utility marks and blue permeability bars should reshape without changing the total budget.
3. Click **Disturb corner**. The external world changes but HOLD does not.
4. Watch the aperture flare in the disturbed corner, the held internal state repair there, and the aperture recede.
5. Lower the spectral budget until a receiver can no longer be repaired well. That is a crude, operational saturation boundary.

## Gate 0 kill conditions

This repository should keep the failures, not erase them.

Gate 0 is weak or dead if any of these happen:

1. **Receiver null:** changing receivers does not reproducibly change the learned permeability allocation.
2. **Budget null:** a fixed-budget adaptive allocation does no better for a receiver than simple hand-designed low/high/band-pass baselines.
3. **HOLD null:** persistent state does not reduce fresh evidence needed after ordinary small changes.
4. **Aperture null:** local contradiction does not produce meaningfully local work, or the aperture fails to close after receiver fidelity recovers.
5. **Closed-loop null:** using receiver marginal utility is no better than allocating rate from source statistics alone.
6. **Cost null:** the probe needed to estimate marginal utility costs more than the computation/communication it saves.

Number 6 is especially dangerous. This first version uses ten counterfactual receiver evaluations per spectral update. That is acceptable for a visible experiment and **not** acceptable as a claimed efficient architecture. A surviving Gate 1 would need cheaper online utility estimation.

## Measurements to add next

The GUI is a mechanism demo. The next script should turn it into receipts:

- task loss vs transmitted spectral budget;
- fresh pixels / coefficients admitted after a local disturbance;
- aperture area over time (the actual "breath");
- adaptive membrane vs all-pass, low-pass, source-energy allocation, and oracle task-aware allocation;
- HOLD vs frame-independent reconstruction;
- receiver swap matrix: allocation learned for receiver A evaluated on receiver B;
- probe cost included in the accounting.

If the adaptive system merely redraws a pretty EQ curve, Thursday should say so and move on.

## Why keep the repo name `Thursday`?

Because naming the repo after the current explanation would bias the next experiment toward defending that explanation. The archive has done that often enough.

`Thursday` is allowed to contain a membrane today, a rotation block tomorrow, and neither by Friday. The invariant is the method:

> **Mechanism → competitor → kill condition → receipt.**

## Nearby work

A few useful reality checks for the first gate:

- Wark, Lundstrom & Fairhall (2007), *Sensory adaptation* — adaptive coding under changing stimulus statistics. https://doi.org/10.1016/j.conb.2007.07.001
- Weber, Krishnamurthy & Fairhall (2019), *Coding Principles in Adaptation*. https://doi.org/10.1146/annurev-vision-091718-014818
- Ye et al. (CVPR 2023), *AccelIR: Task-Aware Image Compression for Accelerating Neural Restoration*. https://openaccess.thecvf.com/content/CVPR2023/html/Ye_AccelIR_Task-Aware_Image_Compression_for_Accelerating_Neural_Restoration_CVPR_2023_paper.html
- Windsheimer, Brand & Kaup (2024), *On Annotation-free Optimization of Video Coding for Machines*. https://arxiv.org/abs/2406.07938

These are not citations of novelty. They are there to stop us from mistaking neighboring established ideas for our result.

## Origin

The motivating thread across the recent work was not "everything is a membrane." It was the repeatedly surviving cut:

> **Spend computation/information at the boundary where a difference becomes visible to a receiver.**

Gate 0 asks whether that sentence can become an adaptive, visible mechanism rather than another elegant re-description.
