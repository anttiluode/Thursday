"""Thursday Gate 0: receiver-driven breathing information aperture.

A small, falsifiable demo derived from the old sigh_image graphical EQ idea.
The membrane has a fixed spectral budget, a persistent HOLD state, and a
named receiver. The receiver's marginal improvement reallocates the spectral
budget. Local contradiction opens a spatial aperture; as the held state is
repaired, the aperture closes again.

Run:
    python breathing_aperture.py

No model weights or training are required.
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

EPS = 1e-8


RECEIVERS = ("coarse", "edges", "texture", "squares")


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < EPS:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def to_gray(image: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return normalize01(image)


def receiver_feature(image: np.ndarray, receiver: str) -> np.ndarray:
    """Receiver-visible consequence. Output stays image-shaped for local error."""
    image = np.asarray(image, dtype=np.float32)
    if receiver == "coarse":
        return cv2.GaussianBlur(image, (0, 0), 3.5)
    if receiver == "edges":
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        # Fixed scaling is important: a local new edge must not renormalize the
        # receiver's definition of every old edge in the image.
        return np.clip(cv2.magnitude(gx, gy) / 4.0, 0.0, 1.0)
    if receiver == "texture":
        lap = np.abs(cv2.Laplacian(image, cv2.CV_32F, ksize=3))
        return np.clip(cv2.GaussianBlur(lap, (0, 0), 0.8) / 4.0, 0.0, 1.0)
    if receiver == "squares":
        # A fixed-scale intersection/corner receiver descended from sigh_image's
        # structure detector, without global normalization side effects.
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        response = np.abs(gx * gy) / 8.0
        return np.clip(cv2.GaussianBlur(response, (0, 0), 0.8), 0.0, 1.0)
    raise ValueError(f"Unknown receiver: {receiver}")


def receiver_loss(reference: np.ndarray, candidate: np.ndarray, receiver: str) -> float:
    a = receiver_feature(reference, receiver)
    b = receiver_feature(candidate, receiver)
    denom = float(np.var(a)) + 1e-4
    return float(np.mean((a - b) ** 2) / denom)


def project_capped_simplex(values: np.ndarray, budget: float, cap: float = 1.0) -> np.ndarray:
    """Euclidean projection onto {0<=x<=cap, sum x=budget}."""
    values = np.asarray(values, dtype=np.float64)
    budget = float(np.clip(budget, 0.0, cap * len(values)))
    if budget <= 0:
        return np.zeros_like(values, dtype=np.float32)
    if budget >= cap * len(values):
        return np.full_like(values, cap, dtype=np.float32)

    lo = float(values.min() - cap)
    hi = float(values.max())
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        x = np.clip(values - mid, 0.0, cap)
        if x.sum() > budget:
            lo = mid
        else:
            hi = mid
    return np.clip(values - hi, 0.0, cap).astype(np.float32)


@dataclass
class MembraneStats:
    receiver_loss: float
    aperture_mean: float
    aperture_peak: float
    marginal_utility: float
    budget_used: float


class BreathingMembrane:
    def __init__(self, size: int = 128, bands: int = 10, budget: float = 3.0):
        self.size = int(size)
        self.bands = int(bands)
        self.budget = float(budget)
        self.receiver = "coarse"

        f = np.fft.fftfreq(self.size)
        fy, fx = np.meshgrid(f, f, indexing="ij")
        radius = np.sqrt(fx * fx + fy * fy)
        self.radius = radius / (radius.max() + EPS)
        # Equal-width radial bands. Each Fourier coefficient belongs to one band.
        self.band_index = np.minimum((self.radius * self.bands).astype(np.int32), self.bands - 1)

        self.gains = project_capped_simplex(np.linspace(1.0, 0.1, self.bands), self.budget)
        self.world = self.synthetic_world()
        self.clean_world = self.world.copy()
        self.belief = self.apply_membrane(self.world)
        self.aperture = np.zeros_like(self.world, dtype=np.float32)
        self.last_mismatch = np.zeros_like(self.world, dtype=np.float32)
        self.last_utilities = np.zeros(self.bands, dtype=np.float32)
        self.spectral_saturated = False

    def synthetic_world(self) -> np.ndarray:
        s = self.size
        y, x = np.mgrid[0:s, 0:s]
        img = 0.12 + 0.25 * (x / max(s - 1, 1)) + 0.10 * (y / max(s - 1, 1))
        cv2.circle(img, (int(0.27 * s), int(0.32 * s)), int(0.13 * s), 0.85, -1)
        cv2.rectangle(img, (int(0.55 * s), int(0.18 * s)), (int(0.86 * s), int(0.43 * s)), 0.60, -1)
        cv2.line(img, (int(0.08 * s), int(0.75 * s)), (int(0.90 * s), int(0.58 * s)), 0.95, 2)
        # Fine checker patch gives the high-frequency receiver something to want.
        yy, xx = np.mgrid[0 : int(0.24 * s), 0 : int(0.24 * s)]
        checker = ((xx // 3 + yy // 3) % 2).astype(np.float32)
        y0, x0 = int(0.67 * s), int(0.10 * s)
        h, w = checker.shape
        img[y0 : y0 + h, x0 : x0 + w] = 0.2 + 0.7 * checker
        return np.clip(img.astype(np.float32), 0.0, 1.0)

    def set_world(self, image: np.ndarray) -> None:
        self.world = to_gray(image, self.size)
        self.clean_world = self.world.copy()
        self.belief = self.apply_membrane(self.world)
        self.aperture.fill(0.0)
        self.last_mismatch.fill(0.0)

    def set_budget(self, budget: float) -> None:
        self.budget = float(np.clip(budget, 0.05, self.bands))
        self.gains = project_capped_simplex(self.gains, self.budget)
        self.spectral_saturated = False

    def set_receiver(self, receiver: str) -> None:
        if receiver not in RECEIVERS:
            raise ValueError(receiver)
        self.receiver = receiver
        self.spectral_saturated = False

    def filter_from_gains(self, image: np.ndarray, gains: np.ndarray) -> np.ndarray:
        fft = np.fft.fft2(image.astype(np.float32))
        filt = gains[self.band_index]
        out = np.fft.ifft2(fft * filt).real.astype(np.float32)
        return np.clip(out, 0.0, 1.0)

    def apply_membrane(self, image: np.ndarray) -> np.ndarray:
        return self.filter_from_gains(image, self.gains)

    def spectral_utilities(self, delta: float = 0.08) -> tuple[np.ndarray, float]:
        """Finite-difference value of opening each frequency band a little more."""
        base = self.apply_membrane(self.world)
        base_loss = receiver_loss(self.world, base, self.receiver)
        utilities = np.zeros(self.bands, dtype=np.float32)
        for i in range(self.bands):
            if self.gains[i] >= 0.999:
                continue
            trial = self.gains.copy()
            trial[i] = min(1.0, trial[i] + delta)
            candidate = self.filter_from_gains(self.world, trial)
            trial_loss = receiver_loss(self.world, candidate, self.receiver)
            utilities[i] = max(0.0, (base_loss - trial_loss) / delta)
        return utilities, base_loss

    def adapt_spectrum(self, learning_rate: float = 0.55, improvement_floor: float = 1e-5) -> float:
        # Once the current receiver cannot buy measurable improvement by
        # reallocating the fixed budget, latch the global membrane. A receiver
        # or budget change releases it. Local surprise does not.
        if self.spectral_saturated:
            return float(self.last_utilities.max(initial=0.0))

        utilities, base_loss = self.spectral_utilities()
        self.last_utilities = utilities
        marginal = float(utilities.max(initial=0.0))

        # Projected finite-difference ascent on receiver utility, with a tiny
        # backtracking line search so a big step cannot pretend to be progress.
        lr = learning_rate
        accepted = None
        accepted_loss = base_loss
        for _ in range(8):
            proposal = project_capped_simplex(self.gains + lr * utilities, self.budget)
            candidate = self.filter_from_gains(self.world, proposal)
            proposal_loss = receiver_loss(self.world, candidate, self.receiver)
            if proposal_loss < base_loss:
                accepted = proposal
                accepted_loss = proposal_loss
                break
            lr *= 0.5

        improvement = base_loss - accepted_loss
        if accepted is None or improvement < improvement_floor:
            self.spectral_saturated = True
            return marginal

        self.gains = accepted
        return marginal

    def local_mismatch(self, candidate: np.ndarray | None = None) -> np.ndarray:
        # Compare HOLD to what the *current membrane can actually deliver*, not
        # to an impossible all-pass world. Otherwise compression error itself
        # would hold the aperture open everywhere forever.
        if candidate is None:
            candidate = self.apply_membrane(self.world)
        ref = receiver_feature(candidate, self.receiver)
        held = receiver_feature(self.belief, self.receiver)
        mismatch = np.abs(ref - held)
        mismatch = cv2.GaussianBlur(mismatch, (0, 0), 1.5)
        # Absolute receiver-space error. Dynamic renormalization would make a
        # shrinking local error look globally large as it approaches zero.
        return np.clip(mismatch, 0.0, 1.0).astype(np.float32)

    def step(self) -> MembraneStats:
        marginal = self.adapt_spectrum()
        candidate = self.apply_membrane(self.world)

        mismatch = self.local_mismatch(candidate)
        self.last_mismatch = mismatch
        # Contradiction opens the aperture. Small errors are treated as HOLD.
        target = np.clip((mismatch - 0.015) / 0.20, 0.0, 1.0)
        # Fast opening, slower closing makes the mechanism visible as "breathing".
        opening = target > self.aperture
        self.aperture[opening] = 0.65 * self.aperture[opening] + 0.35 * target[opening]
        self.aperture[~opening] = 0.88 * self.aperture[~opening] + 0.12 * target[~opening]

        # The membrane only replaces held state where the aperture is open.
        write = np.clip(0.08 + 0.92 * self.aperture, 0.0, 1.0) * self.aperture
        self.belief = self.belief * (1.0 - write) + candidate * write
        self.belief = np.clip(self.belief, 0.0, 1.0).astype(np.float32)

        loss = receiver_loss(self.world, self.belief, self.receiver)
        return MembraneStats(
            receiver_loss=loss,
            aperture_mean=float(self.aperture.mean()),
            aperture_peak=float(self.aperture.max()),
            marginal_utility=marginal,
            budget_used=float(self.gains.sum()),
        )

    def disturb(self, corner: str = "top-right") -> None:
        s = self.size
        patch = max(12, s // 4)
        coords = {
            "top-left": (0, 0),
            "top-right": (0, s - patch),
            "bottom-left": (s - patch, 0),
            "bottom-right": (s - patch, s - patch),
        }
        y0, x0 = coords.get(corner, coords["top-right"])
        yy, xx = np.mgrid[0:patch, 0:patch]
        # Give each receiver a contradiction it has a reason to care about.
        # This keeps Gate 0 about receiver-relative repair rather than about one
        # arbitrary perturbation being accidentally invisible to a task.
        if self.receiver == "coarse":
            pattern = np.full((patch, patch), 0.88, dtype=np.float32)
        elif self.receiver == "edges":
            pattern = 0.10 + 0.85 * ((xx // max(3, patch // 4)) % 2).astype(np.float32)
        elif self.receiver == "texture":
            pattern = 0.10 + 0.85 * ((xx // 2 + yy // 2) % 2).astype(np.float32)
        else:  # squares
            pattern = np.full((patch, patch), 0.12, dtype=np.float32)
            q = max(2, patch // 8)
            pattern[q:-q, q:-q] = 0.90
            pattern[2*q:-2*q, 2*q:-2*q] = 0.18
        # External world changes; HOLD is intentionally untouched.
        self.world[y0 : y0 + patch, x0 : x0 + patch] = pattern

    def restore_world(self) -> None:
        self.world = self.clean_world.copy()

    def reset_belief(self) -> None:
        self.belief = self.apply_membrane(self.world)
        self.aperture.fill(0.0)
        self.last_mismatch.fill(0.0)


class BreathingApertureApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Thursday Gate 0 — Breathing Information Aperture")
        self.root.geometry("1220x900")
        self.model = BreathingMembrane()
        self.running = False
        self.after_id = None
        self.tk_images: dict[str, ImageTk.PhotoImage] = {}
        self._build()
        self.refresh()

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(outer, text="Membrane controls", padding=8)
        controls.pack(fill=tk.X)

        ttk.Button(controls, text="Load image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Disturb corner", command=self.disturb).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Restore world", command=self.restore).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Reset HOLD", command=self.reset_belief).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Step", command=self.step_once).pack(side=tk.LEFT, padx=4)
        self.auto_btn = ttk.Button(controls, text="Auto breathe", command=self.toggle_auto)
        self.auto_btn.pack(side=tk.LEFT, padx=4)

        ttk.Label(controls, text="Receiver:").pack(side=tk.LEFT, padx=(18, 3))
        self.receiver_var = tk.StringVar(value=self.model.receiver)
        receiver_box = ttk.Combobox(controls, textvariable=self.receiver_var, values=RECEIVERS, state="readonly", width=10)
        receiver_box.pack(side=tk.LEFT)
        receiver_box.bind("<<ComboboxSelected>>", self.on_receiver)

        ttk.Label(controls, text="Spectral budget:").pack(side=tk.LEFT, padx=(18, 3))
        self.budget_var = tk.DoubleVar(value=self.model.budget)
        budget = ttk.Scale(controls, from_=0.5, to=self.model.bands, variable=self.budget_var, command=self.on_budget, length=180)
        budget.pack(side=tk.LEFT)

        self.status = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.status).pack(fill=tk.X, pady=(6, 2))

        panels = ttk.Frame(outer)
        panels.pack(fill=tk.BOTH, expand=True)
        for r in range(2):
            panels.grid_rowconfigure(r, weight=1)
        for c in range(2):
            panels.grid_columnconfigure(c, weight=1)

        self.labels = {}
        names = [
            ("world", "WORLD — external state", 0, 0),
            ("belief", "HOLD — persistent internal state", 0, 1),
            ("aperture", "APERTURE — local permeability", 1, 0),
            ("receiver", "RECEIVER ERROR — what still matters", 1, 1),
        ]
        for key, title, row, col in names:
            frame = ttk.LabelFrame(panels, text=title, padding=5)
            frame.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
            label = ttk.Label(frame)
            label.pack(expand=True)
            self.labels[key] = label

        eq_frame = ttk.LabelFrame(outer, text="Fixed-budget spectral permeability (low → high frequency)", padding=5)
        eq_frame.pack(fill=tk.X, pady=(8, 0))
        self.eq_canvas = tk.Canvas(eq_frame, width=1120, height=105, bg="#202020", highlightthickness=0)
        self.eq_canvas.pack(fill=tk.X, expand=True)

    def load_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not path:
            return
        image = np.array(Image.open(path).convert("RGB"))
        self.model.set_world(image)
        self.refresh()

    def disturb(self) -> None:
        self.model.disturb("top-right")
        self.refresh()

    def restore(self) -> None:
        self.model.restore_world()
        self.refresh()

    def reset_belief(self) -> None:
        self.model.reset_belief()
        self.refresh()

    def on_receiver(self, _event=None) -> None:
        self.model.set_receiver(self.receiver_var.get())
        self.refresh()

    def on_budget(self, _value=None) -> None:
        self.model.set_budget(self.budget_var.get())
        self.refresh()

    def step_once(self) -> None:
        stats = self.model.step()
        self.refresh(stats)

    def toggle_auto(self) -> None:
        self.running = not self.running
        self.auto_btn.config(text="Stop" if self.running else "Auto breathe")
        if self.running:
            self._tick()
        elif self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def _tick(self) -> None:
        if not self.running:
            return
        stats = self.model.step()
        self.refresh(stats)
        self.after_id = self.root.after(140, self._tick)

    def _heat(self, x: np.ndarray) -> np.ndarray:
        u8 = (normalize01(x) * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _show(self, key: str, array: np.ndarray, heat: bool = False) -> None:
        if heat:
            rgb = self._heat(array)
        else:
            u8 = (np.clip(array, 0, 1) * 255).astype(np.uint8)
            rgb = cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB)
        im = Image.fromarray(rgb).resize((420, 300), Image.Resampling.LANCZOS)
        tk_im = ImageTk.PhotoImage(im)
        self.tk_images[key] = tk_im
        self.labels[key].config(image=tk_im)

    def draw_eq(self) -> None:
        c = self.eq_canvas
        c.delete("all")
        w = max(c.winfo_width(), 900)
        h = 100
        n = self.model.bands
        bw = w / n
        max_u = float(self.model.last_utilities.max(initial=0.0)) + EPS
        for i, g in enumerate(self.model.gains):
            x0, x1 = i * bw + 3, (i + 1) * bw - 3
            y1 = h - 8
            y0 = y1 - float(g) * 72
            c.create_rectangle(x0, y0, x1, y1, fill="#5aa9e6", outline="")
            util_h = 18 * float(self.model.last_utilities[i]) / max_u
            c.create_rectangle(x0, 3, x1, 3 + util_h, fill="#ffd166", outline="")
            c.create_text((x0 + x1) / 2, h - 1, text=str(i), fill="white", anchor="s")
        c.create_text(8, 8, text="yellow = marginal receiver utility", fill="white", anchor="nw")

    def refresh(self, stats: MembraneStats | None = None) -> None:
        self._show("world", self.model.world)
        self._show("belief", self.model.belief)
        self._show("aperture", self.model.aperture, heat=True)
        self._show("receiver", self.model.last_mismatch, heat=True)
        self.draw_eq()
        if stats is None:
            loss = receiver_loss(self.model.world, self.model.belief, self.model.receiver)
            stats = MembraneStats(loss, float(self.model.aperture.mean()), float(self.model.aperture.max()), float(self.model.last_utilities.max(initial=0.0)), float(self.model.gains.sum()))
        self.status.set(
            f"receiver={self.model.receiver}   task loss={stats.receiver_loss:.4f}   "
            f"aperture mean={stats.aperture_mean:.3f} peak={stats.aperture_peak:.3f}   "
            f"max marginal utility={stats.marginal_utility:.4f}   "
            f"budget={stats.budget_used:.2f}/{self.model.bands}"
        )


def main() -> None:
    root = tk.Tk()
    BreathingApertureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
