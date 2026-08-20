"""Headless receipts for Thursday Gate 0.

Runs fixed-budget spectral allocation baselines and the local HOLD/aperture
repair experiment for every named receiver.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from breathing_aperture import BreathingMembrane, RECEIVERS, receiver_loss


def baseline_gains(kind: str, bands: int, budget: float) -> np.ndarray:
    if kind == "uniform":
        return np.full(bands, budget / bands, dtype=np.float32)
    gains = np.zeros(bands, dtype=np.float32)
    remaining = budget
    order = range(bands) if kind == "lowpass" else range(bands - 1, -1, -1)
    for i in order:
        take = min(1.0, remaining)
        gains[i] = take
        remaining -= take
        if remaining <= 1e-8:
            break
    return gains


def run_receiver(receiver: str, size: int, budget: float, repair_steps: int) -> dict[str, float | str]:
    m = BreathingMembrane(size=size, bands=10, budget=budget)
    m.set_receiver(receiver)

    losses = {}
    for name in ("uniform", "lowpass", "highpass"):
        g = baseline_gains(name, m.bands, budget)
        losses[name] = receiver_loss(m.world, m.filter_from_gains(m.world, g), receiver)

    spectral_steps = 0
    for spectral_steps in range(1, 201):
        m.adapt_spectrum()
        if m.spectral_saturated:
            break
    adaptive_loss = receiver_loss(m.world, m.apply_membrane(m.world), receiver)

    m.reset_belief()
    pre = receiver_loss(m.world, m.belief, receiver)
    m.disturb("top-right")
    disturbed = receiver_loss(m.world, m.belief, receiver)

    aperture = []
    repair_loss = []
    for _ in range(repair_steps):
        stats = m.step()
        aperture.append(stats.aperture_mean)
        repair_loss.append(stats.receiver_loss)

    return {
        "receiver": receiver,
        "uniform_loss": losses["uniform"],
        "lowpass_loss": losses["lowpass"],
        "highpass_loss": losses["highpass"],
        "adaptive_loss": adaptive_loss,
        "spectral_steps": spectral_steps,
        "pre_disturb_loss": pre,
        "disturbed_loss": disturbed,
        "final_repair_loss": repair_loss[-1],
        "peak_aperture_mean": max(aperture),
        "final_aperture_mean": aperture[-1],
        "aperture_area_time": float(np.sum(aperture)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--budget", type=float, default=4.0)
    parser.add_argument("--repair-steps", type=int, default=80)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rows = [run_receiver(r, args.size, args.budget, args.repair_steps) for r in RECEIVERS]
    cols = list(rows[0].keys())
    print("\t".join(cols))
    for row in rows:
        print("\t".join(str(row[c]) if isinstance(row[c], str) else f"{float(row[c]):.6f}" for c in cols))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
