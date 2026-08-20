"""Gate 1 core: tiny creatures with an information metabolism.

The deliberately small question is whether an agent can live from a persistent
internal world and spend fresh sensing only where it is useful, instead of
re-reading the whole world every step.

This module contains no GUI and no learned model. It provides:

- a small dynamic grid world with food, hazards and walls;
- persistent HOLD-style belief state;
- three sensing policies with the same downstream movement rule:
    * ``reflex``     -- tiny local sensing, no persistent memory;
    * ``full``       -- reads the whole world every step;
    * ``metabolism`` -- persistent belief + fixed per-step information budget;
- explicit accounting for cells sensed, sensing energy, food, hazards and life;
- deterministic episode helpers for receipts and later visualisation.

The simulation is intentionally a toy. Sensing a distant cell is treated as a
unit-cost "glimpse" rather than a physical camera ray. Gate 1 should first test
the information-accounting idea; geometry and realistic sensors can come later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

PolicyName = Literal["reflex", "full", "metabolism"]
Coord = tuple[int, int]

# 4-neighbour movement plus HOLD/stay.
MOVES: tuple[Coord, ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class WorldConfig:
    size: int = 32
    food_count: int = 10
    hazard_count: int = 10
    wall_fraction: float = 0.035
    hazard_move_probability: float = 0.18
    food_respawn_probability: float = 0.04
    max_food: int = 14


@dataclass(frozen=True)
class AgentConfig:
    initial_energy: float = 30.0
    max_energy: float = 64.0
    basal_cost: float = 0.12
    move_cost: float = 0.05
    bump_cost: float = 0.03
    sensing_cost_per_cell: float = 0.0008
    food_energy: float = 10.0
    hazard_cost: float = 8.0
    local_radius: int = 1
    info_budget: int = 12
    memory_horizon: float = 45.0
    belief_food_prior: float = 0.08
    belief_hazard_prior: float = 0.025
    distance_penalty: float = 0.035
    danger_weight: float = 7.0
    food_weight: float = 2.8
    exploration_weight: float = 0.40


@dataclass
class WorldObservation:
    coord: Coord
    wall: bool
    food: bool
    hazard: bool


@dataclass
class EpisodeResult:
    policy: str
    seed: int
    steps_alive: int
    food_eaten: int
    hazard_hits: int
    cells_sensed: int
    sensing_energy: float
    movement_energy: float
    basal_energy: float
    final_energy: float
    mean_cells_sensed_per_step: float
    food_per_1000_cells: float
    survival_per_1000_cells: float
    alive: bool


class TinyWorld:
    """Small dynamic world used by all three policies."""

    def __init__(self, config: WorldConfig, seed: int = 0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        s = config.size
        if s < 8:
            raise ValueError("world size must be at least 8")

        self.walls = np.zeros((s, s), dtype=bool)
        self.food = np.zeros((s, s), dtype=bool)
        self.hazards = np.zeros((s, s), dtype=bool)

        # Keep the border open so wall generation does not create a sealed box.
        wall_target = int(round(s * s * config.wall_fraction))
        candidates = [(y, x) for y in range(1, s - 1) for x in range(1, s - 1)]
        self.rng.shuffle(candidates)
        for y, x in candidates[:wall_target]:
            self.walls[y, x] = True

        self._scatter(self.food, config.food_count)
        self._scatter(self.hazards, config.hazard_count, avoid=self.food)

    @property
    def size(self) -> int:
        return self.config.size

    def _scatter(self, layer: np.ndarray, count: int, avoid: np.ndarray | None = None) -> None:
        open_mask = ~self.walls & ~layer
        if avoid is not None:
            open_mask &= ~avoid
        coords = np.argwhere(open_mask)
        if len(coords) == 0:
            return
        count = min(count, len(coords))
        picks = self.rng.choice(len(coords), size=count, replace=False)
        for idx in np.atleast_1d(picks):
            y, x = coords[int(idx)]
            layer[y, x] = True

    def random_open_cell(self, *, avoid_hazard: bool = True) -> Coord:
        mask = ~self.walls
        if avoid_hazard:
            mask &= ~self.hazards
        coords = np.argwhere(mask)
        y, x = coords[int(self.rng.integers(len(coords)))]
        return int(y), int(x)

    def in_bounds(self, coord: Coord) -> bool:
        y, x = coord
        return 0 <= y < self.size and 0 <= x < self.size

    def observe(self, coords: Iterable[Coord]) -> list[WorldObservation]:
        out: list[WorldObservation] = []
        seen: set[Coord] = set()
        for coord in coords:
            if coord in seen or not self.in_bounds(coord):
                continue
            seen.add(coord)
            y, x = coord
            out.append(
                WorldObservation(
                    coord=coord,
                    wall=bool(self.walls[y, x]),
                    food=bool(self.food[y, x]),
                    hazard=bool(self.hazards[y, x]),
                )
            )
        return out

    def consume_food(self, coord: Coord) -> bool:
        y, x = coord
        if self.food[y, x]:
            self.food[y, x] = False
            return True
        return False

    def step_dynamics(self) -> None:
        """Move hazards sparsely and occasionally respawn food."""
        if self.rng.random() < self.config.hazard_move_probability:
            new_hazards = np.zeros_like(self.hazards)
            ys, xs = np.where(self.hazards)
            order = self.rng.permutation(len(ys))
            for i in order:
                y, x = int(ys[i]), int(xs[i])
                moves = list(MOVES)
                self.rng.shuffle(moves)
                placed = False
                for dy, dx in moves:
                    yy, xx = y + dy, x + dx
                    if not self.in_bounds((yy, xx)):
                        continue
                    if self.walls[yy, xx] or new_hazards[yy, xx]:
                        continue
                    new_hazards[yy, xx] = True
                    placed = True
                    break
                if not placed:
                    new_hazards[y, x] = True
            self.hazards = new_hazards

        if (
            int(self.food.sum()) < self.config.max_food
            and self.rng.random() < self.config.food_respawn_probability
        ):
            self._scatter(self.food, 1, avoid=self.hazards)


class BeliefState:
    """Persistent internal world.

    Unobserved cells are HOLD: the last content remains. ``age`` records how
    long since a cell was refreshed and therefore lets a policy decide that an
    old belief deserves another look without silently deleting it.
    """

    def __init__(self, size: int, config: AgentConfig):
        self.size = size
        self.config = config
        self.food = np.full((size, size), config.belief_food_prior, dtype=np.float32)
        self.hazard = np.full((size, size), config.belief_hazard_prior, dtype=np.float32)
        self.wall = np.zeros((size, size), dtype=np.float32)
        self.known = np.zeros((size, size), dtype=bool)
        self.age = np.full((size, size), config.memory_horizon, dtype=np.float32)
        self.aperture = np.zeros((size, size), dtype=np.float32)

    def tick(self) -> None:
        self.age += 1.0
        self.aperture.fill(0.0)

    def clear_for_reflex(self) -> None:
        """Erase content for the memoryless control, preserving only priors."""
        self.food.fill(self.config.belief_food_prior)
        self.hazard.fill(self.config.belief_hazard_prior)
        self.wall.fill(0.0)
        self.known.fill(False)
        self.age.fill(self.config.memory_horizon)
        self.aperture.fill(0.0)

    def update(self, observations: Sequence[WorldObservation]) -> None:
        for obs in observations:
            y, x = obs.coord
            self.food[y, x] = 1.0 if obs.food else 0.0
            self.hazard[y, x] = 1.0 if obs.hazard else 0.0
            self.wall[y, x] = 1.0 if obs.wall else 0.0
            self.known[y, x] = True
            self.age[y, x] = 0.0
            self.aperture[y, x] = 1.0

    def mark_food_consumed(self, coord: Coord) -> None:
        y, x = coord
        self.food[y, x] = 0.0
        self.known[y, x] = True
        self.age[y, x] = 0.0

    def stale(self) -> np.ndarray:
        return np.clip(self.age / self.config.memory_horizon, 0.0, 1.0)

    def uncertainty(self) -> np.ndarray:
        # Exact recent observations are low uncertainty; old or never-observed
        # cells progressively become worth checking again.
        stale = self.stale()
        unknown = (~self.known).astype(np.float32)
        return np.maximum(stale, unknown)


@dataclass
class CreatureMetrics:
    steps: int = 0
    food_eaten: int = 0
    hazard_hits: int = 0
    cells_sensed: int = 0
    sensing_energy: float = 0.0
    movement_energy: float = 0.0
    basal_energy: float = 0.0
    wall_bumps: int = 0


class TinyCreature:
    """One agent body plus a sensing policy and persistent belief."""

    def __init__(
        self,
        policy: PolicyName,
        world: TinyWorld,
        config: AgentConfig,
        seed: int = 0,
        position: Coord | None = None,
    ):
        if policy not in ("reflex", "full", "metabolism"):
            raise ValueError(f"unknown policy: {policy}")
        self.policy = policy
        self.world = world
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.position = position if position is not None else world.random_open_cell()
        self.energy = config.initial_energy
        self.belief = BeliefState(world.size, config)
        self.metrics = CreatureMetrics()
        self.alive = True

    @property
    def hunger(self) -> float:
        return float(np.clip(1.0 - self.energy / self.config.max_energy, 0.0, 1.0))

    def _local_cells(self, radius: int | None = None) -> list[Coord]:
        r = self.config.local_radius if radius is None else radius
        y0, x0 = self.position
        coords: list[Coord] = []
        for y in range(y0 - r, y0 + r + 1):
            for x in range(x0 - r, x0 + r + 1):
                if self.world.in_bounds((y, x)):
                    coords.append((y, x))
        return coords

    def _metabolic_need_map(self) -> np.ndarray:
        """Receiver-relative need-to-know score derived only from internal state.

        The policy cannot inspect world truth while deciding what to sense.
        Hunger makes remembered/stale food more valuable to refresh. Stale
        danger near the body is expensive to ignore. Unknown/stale cells also
        receive a smaller exploration pressure.
        """
        s = self.world.size
        yy, xx = np.mgrid[0:s, 0:s]
        y0, x0 = self.position
        dist = np.abs(yy - y0) + np.abs(xx - x0)
        proximity = np.exp(-dist / max(2.0, s / 7.0)).astype(np.float32)

        stale = self.belief.stale()
        uncertainty = self.belief.uncertainty()
        hunger = self.hunger

        # Check old positive beliefs aggressively: stale food may have vanished,
        # and stale hazards may have moved. Also spend some budget on nearby
        # uncertainty and exploration so a creature can discover new food.
        food_need = (0.25 + 1.75 * hunger) * self.belief.food * (0.20 + stale)
        danger_need = self.config.danger_weight * self.belief.hazard * (0.30 + stale) * (
            0.25 + 0.75 * proximity
        )
        exploration = self.config.exploration_weight * uncertainty * (
            0.25 + 0.75 * proximity
        )

        # A little periodic refresh pressure prevents permanent fixation on one
        # patch while keeping distant random access more expensive.
        distance_cost = self.config.distance_penalty * (dist / max(1, s - 1))
        score = food_need + danger_need + exploration + 0.15 * stale - distance_cost
        score[self.belief.wall > 0.5] *= 0.1
        return score.astype(np.float32)

    def choose_sensing_cells(self) -> list[Coord]:
        if self.policy == "full":
            return [(y, x) for y in range(self.world.size) for x in range(self.world.size)]

        local = self._local_cells()
        if self.policy == "reflex":
            return local

        budget = max(len(local), int(self.config.info_budget))
        budget = min(budget, self.world.size * self.world.size)
        if len(local) >= budget:
            return local[:budget]

        score = self._metabolic_need_map().copy()
        for y, x in local:
            score[y, x] = -np.inf

        # Tiny deterministic random jitter breaks large score ties without
        # peeking at world truth.
        jitter = self.rng.random(score.shape, dtype=np.float32) * 1e-5
        flat = (score + jitter).ravel()
        k = budget - len(local)
        if k <= 0:
            return local
        if k >= flat.size:
            picks = np.argsort(flat)[::-1]
        else:
            candidate = np.argpartition(flat, -k)[-k:]
            picks = candidate[np.argsort(flat[candidate])[::-1]]

        out = list(local)
        for idx in picks[:k]:
            y, x = divmod(int(idx), self.world.size)
            out.append((y, x))
        return out

    def _nearest_food_target(self) -> Coord | None:
        # Only strong positive beliefs are treated as remembered food.
        coords = np.argwhere(self.belief.food > 0.60)
        if len(coords) == 0:
            return None
        y0, x0 = self.position
        d = np.abs(coords[:, 0] - y0) + np.abs(coords[:, 1] - x0)
        # Prefer younger beliefs when distance ties.
        ages = self.belief.age[coords[:, 0], coords[:, 1]]
        order = np.lexsort((ages, d))
        y, x = coords[int(order[0])]
        return int(y), int(x)

    def _exploration_target(self) -> Coord:
        # Explore the stalest/least-known useful area rather than random walking.
        score = self.belief.uncertainty().copy()
        score[self.belief.wall > 0.5] = -1.0
        y0, x0 = self.position
        yy, xx = np.mgrid[0:self.world.size, 0:self.world.size]
        dist = np.abs(yy - y0) + np.abs(xx - x0)
        score -= 0.01 * dist
        jitter = self.rng.random(score.shape, dtype=np.float32) * 1e-5
        idx = int(np.argmax(score + jitter))
        return divmod(idx, self.world.size)

    def choose_move(self) -> Coord:
        target = self._nearest_food_target()
        if target is None:
            target = self._exploration_target()

        y0, x0 = self.position
        ty, tx = target
        best: list[Coord] = []
        best_score = -np.inf
        for dy, dx in MOVES:
            y, x = y0 + dy, x0 + dx
            if not self.world.in_bounds((y, x)):
                continue
            # Planning uses belief only. Unknown real walls can still cause a bump.
            if self.belief.wall[y, x] > 0.5:
                continue

            dist_to_food = abs(y - ty) + abs(x - tx)
            hazard = float(self.belief.hazard[y, x])
            # Also avoid remembered danger immediately adjacent to the candidate.
            y1, y2 = max(0, y - 1), min(self.world.size, y + 2)
            x1, x2 = max(0, x - 1), min(self.world.size, x + 2)
            nearby_danger = float(self.belief.hazard[y1:y2, x1:x2].max(initial=0.0))
            food_here = float(self.belief.food[y, x])

            score = (
                -float(dist_to_food)
                + self.config.food_weight * self.hunger * food_here
                - self.config.danger_weight * hazard
                - 1.4 * nearby_danger
            )
            if (dy, dx) == (0, 0):
                score -= 0.15

            if score > best_score + 1e-9:
                best_score = score
                best = [(dy, dx)]
            elif abs(score - best_score) <= 1e-9:
                best.append((dy, dx))

        return best[int(self.rng.integers(len(best)))] if best else (0, 0)

    def sense(self) -> int:
        if self.policy == "reflex":
            self.belief.clear_for_reflex()
        cells = self.choose_sensing_cells()
        obs = self.world.observe(cells)
        self.belief.update(obs)
        n = len(obs)
        sensing_cost = n * self.config.sensing_cost_per_cell
        self.energy -= sensing_cost
        self.metrics.cells_sensed += n
        self.metrics.sensing_energy += sensing_cost
        return n

    def act(self) -> None:
        dy, dx = self.choose_move()
        y0, x0 = self.position
        candidate = (y0 + dy, x0 + dx)

        moved = False
        if candidate != self.position:
            y, x = candidate
            if self.world.in_bounds(candidate) and not self.world.walls[y, x]:
                self.position = candidate
                moved = True
            else:
                self.energy -= self.config.bump_cost
                self.metrics.movement_energy += self.config.bump_cost
                self.metrics.wall_bumps += 1
                if self.world.in_bounds(candidate):
                    # A physical collision gives a tiny local observation for free:
                    # bodies can discover a wall by running into it.
                    yy, xx = candidate
                    self.belief.wall[yy, xx] = 1.0
                    self.belief.known[yy, xx] = True
                    self.belief.age[yy, xx] = 0.0

        if moved:
            self.energy -= self.config.move_cost
            self.metrics.movement_energy += self.config.move_cost

        y, x = self.position
        if self.world.hazards[y, x]:
            self.energy -= self.config.hazard_cost
            self.metrics.hazard_hits += 1
            self.belief.hazard[y, x] = 1.0
            self.belief.known[y, x] = True
            self.belief.age[y, x] = 0.0

        if self.world.consume_food(self.position):
            self.energy = min(self.config.max_energy, self.energy + self.config.food_energy)
            self.metrics.food_eaten += 1
            self.belief.mark_food_consumed(self.position)

    def step(self) -> bool:
        if not self.alive:
            return False

        self.world.step_dynamics()
        self.belief.tick()
        self.sense()

        self.energy -= self.config.basal_cost
        self.metrics.basal_energy += self.config.basal_cost
        self.act()
        self.metrics.steps += 1

        if self.energy <= 0.0:
            self.alive = False
        return self.alive


def run_episode(
    policy: PolicyName,
    *,
    seed: int = 0,
    max_steps: int = 600,
    world_config: WorldConfig | None = None,
    agent_config: AgentConfig | None = None,
) -> EpisodeResult:
    """Run one deterministic episode for a policy."""
    wc = world_config or WorldConfig()
    ac = agent_config or AgentConfig()
    world = TinyWorld(wc, seed=seed)

    # Spawn near the centre when possible so all policies start at the same
    # location for a given world. If blocked/hazardous, choose an open cell.
    centre = (wc.size // 2, wc.size // 2)
    if world.walls[centre] or world.hazards[centre]:
        centre = world.random_open_cell()

    # Separate policy RNG stream from world dynamics while remaining repeatable.
    policy_offset = {"reflex": 101, "full": 202, "metabolism": 303}[policy]
    creature = TinyCreature(policy, world, ac, seed=seed + policy_offset, position=centre)

    for _ in range(max_steps):
        if not creature.step():
            break

    m = creature.metrics
    steps = m.steps
    cells = m.cells_sensed
    return EpisodeResult(
        policy=policy,
        seed=seed,
        steps_alive=steps,
        food_eaten=m.food_eaten,
        hazard_hits=m.hazard_hits,
        cells_sensed=cells,
        sensing_energy=m.sensing_energy,
        movement_energy=m.movement_energy,
        basal_energy=m.basal_energy,
        final_energy=float(creature.energy),
        mean_cells_sensed_per_step=float(cells / max(1, steps)),
        food_per_1000_cells=float(1000.0 * m.food_eaten / max(1, cells)),
        survival_per_1000_cells=float(1000.0 * steps / max(1, cells)),
        alive=creature.alive,
    )


def compare_policies(
    seeds: Iterable[int] = range(8),
    *,
    max_steps: int = 600,
    world_config: WorldConfig | None = None,
    agent_config: AgentConfig | None = None,
) -> list[EpisodeResult]:
    """Run matched-seed episodes for all three Gate 1 policies."""
    results: list[EpisodeResult] = []
    for seed in seeds:
        for policy in ("reflex", "full", "metabolism"):
            results.append(
                run_episode(
                    policy,
                    seed=int(seed),
                    max_steps=max_steps,
                    world_config=world_config,
                    agent_config=agent_config,
                )
            )
    return results


def summarise(results: Sequence[EpisodeResult]) -> dict[str, dict[str, float]]:
    """Return simple mean receipts grouped by policy."""
    grouped: dict[str, list[EpisodeResult]] = {}
    for r in results:
        grouped.setdefault(r.policy, []).append(r)

    summary: dict[str, dict[str, float]] = {}
    for policy, rows in grouped.items():
        summary[policy] = {
            "episodes": float(len(rows)),
            "mean_steps_alive": float(np.mean([r.steps_alive for r in rows])),
            "mean_food_eaten": float(np.mean([r.food_eaten for r in rows])),
            "mean_hazard_hits": float(np.mean([r.hazard_hits for r in rows])),
            "mean_cells_sensed": float(np.mean([r.cells_sensed for r in rows])),
            "mean_cells_per_step": float(np.mean([r.mean_cells_sensed_per_step for r in rows])),
            "mean_food_per_1000_cells": float(np.mean([r.food_per_1000_cells for r in rows])),
            "mean_survival_per_1000_cells": float(
                np.mean([r.survival_per_1000_cells for r in rows])
            ),
        }
    return summary


if __name__ == "__main__":
    rows = compare_policies(range(6), max_steps=500)
    for policy, values in summarise(rows).items():
        print(f"\n{policy}")
        for key, value in values.items():
            print(f"  {key:28s} {value:.4f}")
