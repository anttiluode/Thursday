# Gate 2 — The Boundary Remembers

This is a deliberately strange video codec experiment.

A conventional predictive codec remembers previous decoded frames. Gate 2 makes the **sensory boundary itself** an explicit dynamical state:

```text
current world x_t
      │
      ▼
remembered boundary m_t ──► predictor
      │                        │
      │                        ▼
      └────────────── x_t - m_t = innovation
                               │
                     quantise / dead-zone / sparsify
                               │
                               ▼
                         transmitted evidence
                               │
                               ▼
                         reconstructed world
                               │
                               ▼
                  m_(t+1) = rho*m_t + (1-rho)*recon
```

The crucial implementation constraint is that the encoder updates its boundary from the **reconstructed frame**, not the hidden source. The decoder performs the same update from transmitted data only. There is no secret encoder-only memory.

## `.thv` format

Thursday Video v1 is a small streaming container, not a wrapper around H.264.

- magic: `THV1`
- JSON stream header: resolution, FPS, colour format, codec name
- repeated `FRM1` packets
- each packet stores frame index, keyframe flag, `rho`, quantisation step, deadzone, payload length and CRC32
- keyframes: zlib-compressed raw BGR8
- delta frames: sparse signed `int16` innovation values + packed nonzero mask, then zlib

Because `rho`, quantisation and deadzone live in every packet, the GUI can alter the boundary while recording and a later decoder still reproduces the correct state.

This is not intended to beat AV1. The format exists so the state and failure modes are inspectable.

## Run

```bash
python thv_gui.py
```

You can:

- open normal video through OpenCV;
- use webcam 0;
- change boundary memory (`rho`) live;
- change residual quantisation and deadzone live;
- force keyframes;
- record webcam/video directly to `.thv`;
- reopen `.thv` and decode it without the source video.

The six views are:

1. current external frame;
2. boundary memory before new evidence;
3. raw innovation requested by the current world;
4. reconstructed world;
5. current reconstruction error (only available while source exists);
6. residual that actually crossed the boundary.

## The afterimage knob

High `rho` makes boundary state slow to forget. A high deadzone refuses to transmit small/medium corrections. Together they deliberately create temporal ghosts:

```text
old object disappears
      ↓
current world says "gone"
      ↓
boundary still says "here"
      ↓
innovation is too expensive / below deadzone
      ↓
decoder continues to see old object
```

That failure is useful here. Gate 2 asks when remembered sensor state can be distinguished from current world state, rather than pretending temporal residue does not exist.

There is a unit test that forces this condition on purpose.

## Important distinction

The boundary pane is **not a reconstructed video frame that must itself look honest**. It is the state of the observation machinery. The only final video-like object is the decoded world. This is the reason an afterimage-like intermediate representation can be acceptable in a receiver-oriented system even though it would be ugly in a conventional pixel codec.

## What to try first

With a webcam:

1. Start around `rho=0.94`, `q=8`, `deadzone=4`.
2. Hold a high-contrast object still for a few seconds.
3. Raise `rho` toward `0.98–0.995` and deadzone toward `20–60`.
4. Remove the object quickly.
5. Watch **BOUNDARY MEMORY** keep it and **RAW INNOVATION** demand its removal.
6. If the correction is suppressed, the object persists in **DECODED WORLD**.
7. Lower deadzone again: reality is allowed back in and the ghost is repaid.

That is the computational version of the question: *when is it safe to live from the boundary's memory, and when must the system pay to ask reality again?*

## Tests

```bash
pytest -q tests/test_thv_codec.py
```

The current tests cover:

- exact keyframe round-trip;
- encoder/decoder boundary-state synchrony over deltas;
- `.thv` file round-trip;
- an explicit afterimage case when innovation is suppressed;
- parameter changes carried in the stream rather than hidden in the encoder;
- starting a saved recording mid-session from a fresh keyframe.

## Next kill tests

1. **Memory null:** compare against previous-frame prediction at matched rate. If the long boundary gives no useful rate/receiver advantage, it is decoration.
2. **Ghost cost:** sweep `rho × deadzone` and measure persistence after real objects disappear.
3. **Boundary-aware receiver:** give a downstream model both decoded world and boundary state; ask whether it can distinguish world change from boundary relaxation better than a receiver seeing only decoded pixels.
4. **Receiver distortion:** stop scoring only PSNR. Test motion, identity, object presence and event detection at matched `.thv` size.
5. **Real rate:** compare actual file bytes, including keyframes, packet headers and sparse masks—not a symbolic budget.
