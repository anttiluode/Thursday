"""Thursday Gate 3: little organisms living in a learned SplatWorld substrate.

The frozen SplatWorld decoder is treated as *terrain*, not as an image generator
owned by the creatures. A 2-D ecological slice is embedded in its 128-D latent
space. Every grid cell is decoded once and converted into three renewable
resource capacities derived from the decoder's output:

    FORM    coarse spatial structure
    EDGE    gradient energy
    CARRIER fine/high-frequency energy

Creatures only sense a few nearby cells per tick. They keep tiny HOLD memories
of previously sensed utility, pay energy for fresh observations, consume local
resources, reproduce with mutations, and die when energy reaches zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Protocol, Tuple
import math
import os

import cv2
import numpy as np

LATENT = 128
RESOURCE_NAMES = ("FORM", "EDGE", "CARRIER")


class Decoder(Protocol):
    def __call__(self, zs: np.ndarray) -> np.ndarray:
        """Return float images shaped (N, 3, H, W), normally in [0, 1]."""


class OnnxSplatDecoder:
    """Small wrapper matching SplatWorld's public ONNX decoder."""

    def __init__(self, path: str | os.PathLike[str]):
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for the real SplatWorld model. "
                "Install it with: python -m pip install onnxruntime"
            ) from exc
        self.path = path
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, zs: np.ndarray) -> np.ndarray:
        zs = np.ascontiguousarray(zs, dtype=np.float32)
        return self.session.run([self.output_name], {self.input_name: zs})[0]


class MockSplatDecoder:
    """Deterministic wave-ish fallback for tests and first launch without ONNX."""

    def __init__(self, size: int = 48, seed: int = 17):
        self.size = int(size)
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(0, 1 / math.sqrt(LATENT), (12, LATENT)).astype(np.float32)

    def __call__(self, zs: np.ndarray) -> np.ndarray:
        zs = np.asarray(zs, dtype=np.float32)
        n = len(zs)
        h = w = self.size
        yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
        out = np.empty((n, 3, h, w), dtype=np.float32)
        pars = zs @ self.proj.T
        radii = np.linalg.norm(zs, axis=1)
        for i in range(n):
            p = pars[i]
            r = radii[i]
            img = np.zeros((3, h, w), np.float32)
            decohere = np.clip((r - 15.0) / 28.0, 0.0, 1.0)
            for k in range(4):
                cx = 0.45 * np.tanh(p[k])
                cy = 0.45 * np.tanh(p[(k + 3) % 12])
                th = p[(k + 5) % 12]
                freq = 3.0 + 5.0 * abs(np.tanh(p[(k + 7) % 12])) + decohere * 7.0
                sig = 0.22 + 0.09 * (k % 2)
                xr = np.cos(th) * (xx - cx) + np.sin(th) * (yy - cy)
                env = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig * sig))
                phase = p[(k + 9) % 12] + decohere * p[(k + 10) % 12] * 3.0
                wave = env * np.cos(2 * np.pi * freq * xr + phase)
                for c in range(3):
                    img[c] += wave * (0.15 + 0.08 * np.tanh(p[(k + c + 1) % 12]))
            core = np.exp(-(xx * xx + yy * yy) / (0.35 + 0.02 * np.clip(r, 0, 20)))
            img += (0.48 + 0.30 * core)[None]
            out[i] = np.clip(img, 0, 1)
        return out


@dataclass(frozen=True)
class SubstrateConfig:
    width: int = 64
    height: int = 40
    radius_min: float = 5.0
    radius_max: float = 48.0
    identity_tilt: float = 0.30
    batch: int = 48
    seed: int = 7


@dataclass
class LearnedSubstrate:
    config: SubstrateConfig
    basis: np.ndarray
    capacity: np.ndarray
    volatility: np.ndarray
    thumbnails: Optional[np.ndarray] = None

    @property
    def width(self) -> int:
        return self.config.width

    @property
    def height(self) -> int:
        return self.config.height

    def radius_for_y(self, y: float) -> float:
        denom = max(1.0, self.height - 1.0)
        t = np.clip(float(y) / denom, 0.0, 1.0)
        return self.config.radius_min + t * (self.config.radius_max - self.config.radius_min)

    def shell_for_y(self, y: float) -> str:
        r = self.radius_for_y(y)
        if r < 15:
            return "core"
        if r < 35:
            return "ghost"
        if r < 100:
            return "fire"
        return "void"

    def latent_at(self, x: float, y: float) -> np.ndarray:
        theta = 2.0 * math.pi * ((float(x) % self.width) / self.width)
        d0, e1, e2 = self.basis
        d = self.config.identity_tilt * d0 + math.cos(theta) * e1 + math.sin(theta) * e2
        d = d / (np.linalg.norm(d) + 1e-12)
        return (d * self.radius_for_y(y)).astype(np.float32)

    def index(self, x: float, y: float) -> Tuple[int, int]:
        ix = int(round(float(x))) % self.width
        iy = int(np.clip(round(float(y)), 0, self.height - 1))
        return ix, iy

    def cache_payload(self) -> dict[str, np.ndarray]:
        return {
            "basis": self.basis.astype(np.float32),
            "capacity": self.capacity.astype(np.float32),
            "volatility": self.volatility.astype(np.float32),
        }

    @classmethod
    def from_cache(cls, path: str | os.PathLike[str], config: SubstrateConfig) -> "LearnedSubstrate":
        data = np.load(path)
        return cls(config=config, basis=data["basis"], capacity=data["capacity"], volatility=data["volatility"])

    @classmethod
    def build(cls, decoder: Decoder, config: SubstrateConfig, progress=None) -> "LearnedSubstrate":
        rng = np.random.default_rng(config.seed)
        q, _ = np.linalg.qr(rng.normal(size=(LATENT, 3)))
        basis = q.T.astype(np.float32)
        probe = cls(config=config, basis=basis,
                    capacity=np.zeros((config.height, config.width, 3), np.float32),
                    volatility=np.zeros((config.height, config.width), np.float32))
        zs = np.stack([
            probe.latent_at(x, y)
            for y in range(config.height)
            for x in range(config.width)
        ]).astype(np.float32)

        feats = []
        total = len(zs)
        for start in range(0, total, config.batch):
            ims = decoder(zs[start:start + config.batch])
            feats.append(_image_features(ims))
            if progress is not None:
                progress(min(total, start + config.batch), total)
        feat = np.concatenate(feats, axis=0).reshape(config.height, config.width, 3)
        capacity = _robust_channel_normalize(feat)
        capacity = 0.08 + 0.92 * np.power(capacity, 0.80)

        volatility = np.zeros((config.height, config.width), np.float32)
        for c in range(3):
            gy, gx = np.gradient(capacity[..., c])
            volatility += gx * gx + gy * gy
        volatility = np.sqrt(volatility)
        lo, hi = np.percentile(volatility, [5, 95])
        volatility = np.clip((volatility - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
        return cls(config=config, basis=basis, capacity=capacity.astype(np.float32), volatility=volatility)


def _image_features(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected (N,3,H,W), got {images.shape}")
    out = np.empty((len(images), 3), np.float32)
    for i, chw in enumerate(images):
        rgb = np.transpose(chw, (1, 2, 0))
        gray = cv2.cvtColor(np.clip(rgb, 0, 1), cv2.COLOR_RGB2GRAY)
        coarse = cv2.GaussianBlur(gray, (0, 0), 2.2)
        fine = gray - cv2.GaussianBlur(gray, (0, 0), 0.8)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        out[i, 0] = float(np.std(coarse))
        out[i, 1] = float(np.mean(np.sqrt(gx * gx + gy * gy)))
        out[i, 2] = float(np.std(fine))
    return out


def _robust_channel_normalize(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[-1]):
        lo, hi = np.percentile(x[..., c], [4, 96])
        y[..., c] = np.clip((x[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
    return y


@dataclass
class EcologyConfig:
    initial_population: int = 28
    max_population: int = 180
    initial_energy: float = 7.5
    basal_cost: float = 0.048
    move_cost: float = 0.018
    bump_cost: float = 0.03
    info_cost: float = 0.014
    volatility_cost: float = 0.025
    bite: float = 0.025
    food_gain: float = 5.5
    regeneration: float = 0.014
    reproduce_energy: float = 11.0
    reproduce_chance: float = 0.035
    mutation_sigma: float = 0.10
    memory_horizon: float = 45.0
    curiosity: float = 0.10
    crowd_cost: float = 0.035
    seed: int = 11


@dataclass
class MemoryCell:
    utility: float
    tick: int


@dataclass
class Creature:
    ident: int
    x: float
    y: float
    energy: float
    pref: np.ndarray
    sense_budget: int
    generation: int = 0
    age: int = 0
    memory: Dict[int, MemoryCell] = field(default_factory=dict)
    sensed: list[Tuple[int, int]] = field(default_factory=list)
    trail: list[Tuple[float, float]] = field(default_factory=list)
    last_intake: float = 0.0
    last_info: int = 0

    def color_rgb(self) -> Tuple[int, int, int]:
        p = np.clip(self.pref / (self.pref.max() + 1e-8), 0, 1)
        return tuple(int(v) for v in (55 + 200 * p))


@dataclass
class EcologyStats:
    tick: int
    population: int
    births: int
    deaths: int
    total_information: int
    mean_energy: float
    mean_budget: float
    core_count: int
    ghost_count: int
    fire_count: int


class SplatEcology:
    NEIGHBOURS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1))

    def __init__(self, substrate: LearnedSubstrate, config: EcologyConfig | None = None):
        self.substrate = substrate
        self.config = config or EcologyConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.resources = substrate.capacity.copy()
        self.creatures: list[Creature] = []
        self.next_ident = 1
        self.tick = 0
        self.births = 0
        self.deaths = 0
        self.total_information = 0
        self.history: list[EcologyStats] = []
        self._spawn_initial(self.config.initial_population)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.config.seed)
        self.resources = self.substrate.capacity.copy()
        self.creatures.clear()
        self.next_ident = 1
        self.tick = self.births = self.deaths = self.total_information = 0
        self.history.clear()
        self._spawn_initial(self.config.initial_population)

    def _spawn_initial(self, n: int) -> None:
        for _ in range(n):
            self.creatures.append(self._new_random_creature())

    def _new_random_creature(self) -> Creature:
        pref = self.rng.dirichlet(np.array([1.2, 1.2, 1.2])).astype(np.float32)
        c = Creature(
            ident=self.next_ident,
            x=float(self.rng.uniform(0, self.substrate.width)),
            y=float(self.rng.uniform(0, self.substrate.height - 1)),
            energy=float(self.config.initial_energy * self.rng.uniform(0.85, 1.15)),
            pref=pref,
            sense_budget=int(self.rng.integers(3, 7)),
        )
        self.next_ident += 1
        return c

    def add_creatures(self, n: int = 5) -> None:
        room = max(0, self.config.max_population - len(self.creatures))
        for _ in range(min(n, room)):
            self.creatures.append(self._new_random_creature())

    def _cell_key(self, x: int, y: int) -> int:
        return y * self.substrate.width + x

    def _candidate_cells(self, c: Creature) -> list[Tuple[int, int]]:
        cx, cy = self.substrate.index(c.x, c.y)
        out = []
        for dx, dy in self.NEIGHBOURS:
            xx = (cx + dx) % self.substrate.width
            yy = int(np.clip(cy + dy, 0, self.substrate.height - 1))
            if (xx, yy) not in out:
                out.append((xx, yy))
        return out

    def _true_utility(self, c: Creature, cell: Tuple[int, int]) -> float:
        x, y = cell
        nourishment = float(np.dot(self.resources[y, x], c.pref))
        danger = self.config.volatility_cost * float(self.substrate.volatility[y, x]) * 6.0
        return nourishment - danger

    def _remembered_utility(self, c: Creature, cell: Tuple[int, int]) -> float:
        x, y = cell
        rec = c.memory.get(self._cell_key(x, y))
        if rec is None:
            return 0.45 * float(np.dot(self.substrate.capacity[y, x], c.pref))
        age = max(0, self.tick - rec.tick)
        confidence = math.exp(-age / max(1e-6, self.config.memory_horizon))
        prior = 0.45 * float(np.dot(self.substrate.capacity[y, x], c.pref))
        return confidence * rec.utility + (1.0 - confidence) * prior

    def _sense(self, c: Creature, candidates: list[Tuple[int, int]]) -> None:
        budget = int(np.clip(c.sense_budget, 1, len(candidates)))
        cx, cy = self.substrate.index(c.x, c.y)
        current = (cx, cy)
        chosen = [current]
        hunger = np.clip((self.config.reproduce_energy - c.energy) / self.config.reproduce_energy, 0, 1)
        scored = []
        for cell in candidates:
            if cell == current:
                continue
            rec = c.memory.get(self._cell_key(*cell))
            if rec is None:
                age_term = 1.0
                remembered = 0.25
            else:
                age_term = min(1.5, (self.tick - rec.tick) / max(1.0, self.config.memory_horizon))
                remembered = max(0.0, rec.utility)
            debt = (0.65 + 1.35 * hunger) * (0.55 + age_term) * (0.45 + remembered)
            debt += float(self.rng.uniform(0, 0.04))
            scored.append((debt, cell))
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen.extend(cell for _, cell in scored[:max(0, budget - 1)])

        c.sensed = chosen
        c.last_info = len(chosen)
        self.total_information += len(chosen)
        for cell in chosen:
            c.memory[self._cell_key(*cell)] = MemoryCell(self._true_utility(c, cell), self.tick)

    def _crowding(self, x: int, y: int, occupancy: np.ndarray) -> float:
        return self.config.crowd_cost * float(occupancy[y, x])

    def _move(self, c: Creature, candidates: list[Tuple[int, int]], occupancy: np.ndarray) -> None:
        if self.rng.random() < self.config.curiosity:
            target = candidates[int(self.rng.integers(len(candidates)))]
        else:
            vals = [self._remembered_utility(c, cell) - self._crowding(*cell, occupancy) for cell in candidates]
            vals = np.asarray(vals) + self.rng.normal(0, 0.015, len(vals))
            target = candidates[int(np.argmax(vals))]
        oldx, oldy = self.substrate.index(c.x, c.y)
        c.x, c.y = float(target[0]), float(target[1])
        if target != (oldx, oldy):
            c.energy -= self.config.move_cost
        c.trail.append((c.x, c.y))
        if len(c.trail) > 36:
            del c.trail[:-36]

    def _feed(self, c: Creature) -> None:
        x, y = self.substrate.index(c.x, c.y)
        avail = self.resources[y, x]
        desire = np.maximum(c.pref, 0.01)
        weighted = avail * desire
        total = float(weighted.sum())
        if total <= 1e-9:
            c.last_intake = 0.0
            return
        amount = min(self.config.bite, total)
        uptake = weighted * (amount / total)
        self.resources[y, x] = np.maximum(0.0, avail - uptake)
        c.last_intake = float(uptake.sum())
        c.energy += self.config.food_gain * c.last_intake

    def _maybe_reproduce(self, c: Creature) -> Optional[Creature]:
        if len(self.creatures) >= self.config.max_population:
            return None
        if c.energy < self.config.reproduce_energy or self.rng.random() > self.config.reproduce_chance:
            return None
        child_energy = c.energy * 0.43
        c.energy *= 0.57
        pref = c.pref + self.rng.normal(0, self.config.mutation_sigma, 3)
        pref = np.clip(pref, 0.015, None)
        pref = (pref / pref.sum()).astype(np.float32)
        budget = c.sense_budget
        if self.rng.random() < 0.38:
            budget += int(self.rng.choice([-1, 1]))
        budget = int(np.clip(budget, 2, 8))
        x = (c.x + int(self.rng.choice([-1, 0, 1]))) % self.substrate.width
        y = np.clip(c.y + int(self.rng.choice([-1, 0, 1])), 0, self.substrate.height - 1)
        child = Creature(
            ident=self.next_ident, x=float(x), y=float(y), energy=float(child_energy),
            pref=pref, sense_budget=budget, generation=c.generation + 1,
        )
        self.next_ident += 1
        self.births += 1
        return child

    def step(self) -> EcologyStats:
        self.tick += 1
        self.resources += self.config.regeneration * (self.substrate.capacity - self.resources)
        self.resources = np.clip(self.resources, 0, self.substrate.capacity)

        occupancy = np.zeros((self.substrate.height, self.substrate.width), np.int16)
        for c in self.creatures:
            x, y = self.substrate.index(c.x, c.y)
            occupancy[y, x] += 1

        newborns: list[Creature] = []
        survivors: list[Creature] = []
        birth_slots = max(0, self.config.max_population - len(self.creatures))
        for c in list(self.creatures):
            c.age += 1
            c.energy -= self.config.basal_cost
            candidates = self._candidate_cells(c)
            self._sense(c, candidates)
            c.energy -= self.config.info_cost * c.last_info
            self._move(c, candidates, occupancy)
            x, y = self.substrate.index(c.x, c.y)
            c.energy -= self.config.volatility_cost * float(self.substrate.volatility[y, x])
            self._feed(c)
            child = self._maybe_reproduce(c) if len(newborns) < birth_slots else None
            if child is not None:
                newborns.append(child)
            if c.energy > 0:
                survivors.append(c)
            else:
                self.deaths += 1
        self.creatures = survivors + newborns
        stats = self.stats()
        self.history.append(stats)
        if len(self.history) > 1000:
            del self.history[:-1000]
        return stats

    def stats(self) -> EcologyStats:
        if self.creatures:
            energies = [c.energy for c in self.creatures]
            budgets = [c.sense_budget for c in self.creatures]
        else:
            energies = [0.0]; budgets = [0.0]
        shells = {"core": 0, "ghost": 0, "fire": 0, "void": 0}
        for c in self.creatures:
            shells[self.substrate.shell_for_y(c.y)] += 1
        return EcologyStats(
            tick=self.tick,
            population=len(self.creatures),
            births=self.births,
            deaths=self.deaths,
            total_information=self.total_information,
            mean_energy=float(np.mean(energies)),
            mean_budget=float(np.mean(budgets)),
            core_count=shells["core"], ghost_count=shells["ghost"],
            fire_count=shells["fire"] + shells["void"],
        )


def find_splatworld_model(explicit: str | None = None) -> Optional[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("SPLATWORLD_MODEL")
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates.extend([
        cwd / "splat_decoder.onnx",
        here / "splat_decoder.onnx",
        cwd.parent / "SplatWorld" / "splat_decoder.onnx",
        here.parent / "SplatWorld" / "splat_decoder.onnx",
    ])
    seen = set()
    for p in candidates:
        try:
            rp = p.expanduser().resolve()
        except OSError:
            continue
        if rp in seen:
            continue
        seen.add(rp)
        if rp.is_file():
            return rp
    return None


def render_selected(decoder: Decoder, substrate: LearnedSubstrate, creature: Creature) -> np.ndarray:
    out = decoder(substrate.latent_at(creature.x, creature.y)[None])[0]
    rgb = np.transpose(out, (1, 2, 0))
    return np.clip(rgb, 0, 1).astype(np.float32)


def run_headless(decoder: Decoder | None = None, steps: int = 500, seed: int = 11,
                 substrate_config: SubstrateConfig | None = None) -> EcologyStats:
    decoder = decoder or MockSplatDecoder()
    scfg = substrate_config or SubstrateConfig(width=36, height=24, batch=48)
    substrate = LearnedSubstrate.build(decoder, scfg)
    eco = SplatEcology(substrate, EcologyConfig(seed=seed))
    for _ in range(steps):
        eco.step()
        if not eco.creatures:
            break
    return eco.stats()
