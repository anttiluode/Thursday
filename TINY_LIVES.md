# Gate 1 — Tiny Lives / Information Metabolism

The creature does not survive by knowing everything. It survives by deciding what is worth knowing again.

This is the first non-GUI core for the "information metabolism" thought.

## World

A `32 x 32` dynamic grid contains sparse food, moving hazards and walls. Food is consumed and occasionally respawns. Hazards move stochastically.

## Body

Every creature has energy. Staying alive, moving, bumping into walls and sensing all cost energy. Eating replenishes energy. A hazard hit removes energy.

## Internal world

The creature owns persistent maps for food, hazard and wall belief plus an `age` map.

**HOLD means content is not erased merely because it was not sensed this step.**

An old belief can therefore still guide behaviour while its age simultaneously increases the pressure to check it again.

## Three controls

All three creatures use the same downstream movement rule. They differ mainly in how they obtain information.

### `reflex`

Sees only a local `3 x 3` patch and clears its belief back to priors every step. Cheap, but nearly memoryless.

### `full`

Reads every cell on every step. It has excellent fresh evidence, but the information bill is charged to the same energy account as life.

### `metabolism`

Keeps persistent HOLD and receives a fixed information budget. With the default settings it senses only **12 cells per step**: the local `3 x 3` patch plus three extra targeted glimpses.

The targeted cells are selected from internal state only. The selector is not allowed to inspect hidden world truth. Hunger increases the value of checking remembered/stale food; stale remembered danger matters especially near the body; unknown and old cells create a smaller exploration pressure.

## Important toy assumption

A targeted distant cell currently costs one information unit regardless of distance. Think of it as an abstract glimpse, not a physically correct retina or ray caster. If Gate 1 survives, realistic sensing geometry is a later control.

## Run

```bash
python tiny_lives_core.py
```

The module runs six matched seeds and prints mean receipts for `reflex`, `full`, and `metabolism`.

Tests:

```bash
pytest -q tests/test_tiny_lives_core.py
```

## Smoke result, not a Gate 1 verdict

On the current defaults, the development smoke run gave roughly:

| policy | mean life (max 500) | food eaten | cells sensed / step |
|---|---:|---:|---:|
| reflex | 302 | 2.5 | 9 |
| full | 76 | 4.3 | 1024 |
| metabolism | 476 | 14.2 | 12 |

Do **not** read this as a benchmark claim. The parameters were chosen to make sensing cost visible and to verify that the mechanism is not dead on arrival. A real receipt needs many seeds, parameter sweeps, cost ablations and controls where sensing cost is removed.

The interesting tension is already present:

- `reflex` can sometimes survive very cheaply by stumbling onto food;
- `full` obtains abundant information but can spend itself to death;
- `metabolism` pays slightly more than reflex while using memory and selective refresh to find much more food.

So the first Gate 1 question is not "does metabolism win every scalar metric?" It is:

> **Does persistent HOLD plus selective refresh produce a better survival / task / information frontier than either memoryless local sensing or continual full observation?**

## Kill conditions to add next

1. Remove sensing energy cost. If `full` still loses badly, the movement/world model is unfair.
2. Give `reflex` the same 12-cell budget without memory. If it matches metabolism, HOLD is not load-bearing.
3. Keep memory but choose the three extra glimpses randomly. If it matches targeted sensing, the need-to-know policy is decorative.
4. Freeze the world. If HOLD does not reduce sensing need in a stable world, the mechanism is wrong.
5. Increase world volatility. A real information metabolism should spend more when old beliefs become unreliable.

The next file should be the visualizer: world, HOLD, aperture/need-to-know map and energy history moving together so the creature's information metabolism is something we can literally watch.
