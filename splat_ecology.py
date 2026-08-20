#!/usr/bin/env python3
"""Thursday Gate 3 — Splat Ecology.

Little information-metabolising organisms live on a 2-D slice through the
frozen SplatWorld decoder. The GUI makes the learned substrate visible as
terrain, lets creatures consume/deplete model-derived resource channels, and
shows the actual SplatWorld render at a selected creature's latent position.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from splat_ecology_core import (
    EcologyConfig, LearnedSubstrate, MockSplatDecoder, OnnxSplatDecoder,
    RESOURCE_NAMES, SplatEcology, SubstrateConfig, find_splatworld_model,
    render_selected,
)

MAP_W, MAP_H = 720, 500
VIEW = 420
MIND_W, MIND_H = 420, 250
PLOT_W, PLOT_H = 420, 250


class SplatEcologyApp:
    def __init__(self, root: tk.Tk, model_path: str | None = None, force_mock: bool = False):
        self.root = root
        self.root.title("Thursday Gate 3 — Splat Ecology: things living in learned weights")
        self.root.geometry("1480x940")
        self.running = False
        self.after_id = None
        self.decoder = None
        self.substrate = None
        self.ecology = None
        self.selected_id = None
        self.last_creature_view = None
        self.frame_counter = 0
        self.model_path = None
        self.force_mock = force_mock

        self.info_cost_var = tk.DoubleVar(value=0.014)
        self.regen_var = tk.DoubleVar(value=0.014)
        self.mutation_var = tk.DoubleVar(value=0.10)
        self.ticks_var = tk.IntVar(value=2)
        self.status_var = tk.StringVar(value="Preparing substrate…")
        self.creature_var = tk.StringVar(value="No creature selected")

        self._build_ui()
        path = None if force_mock else find_splatworld_model(model_path)
        if path is not None:
            self._load_model_path(path)
        else:
            self.decoder = MockSplatDecoder(size=48)
            self.model_path = None
            self._build_world(mock=True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(outer, text="Life / substrate controls", padding=6)
        controls.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(controls, text="Load SplatWorld model", command=self.choose_model).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="Run / pause", command=self.toggle).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="Step", command=self.step_once).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="Reset life", command=self.reset_life).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="+10 spores", command=lambda: self.add_spores(10)).pack(side=tk.LEFT, padx=8)

        self._slider(controls, "price of information", self.info_cost_var, 0.0, 0.06, 170)
        self._slider(controls, "resource regen", self.regen_var, 0.002, 0.06, 150)
        self._slider(controls, "mutation", self.mutation_var, 0.0, 0.28, 150)
        ttk.Label(controls, text="ticks/frame").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(controls, from_=1, to=20, textvariable=self.ticks_var, width=4).pack(side=tk.LEFT)

        ttk.Label(outer, textvariable=self.status_var).pack(fill=tk.X, pady=(0, 4))

        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=2)
        body.rowconfigure(1, weight=1)

        world_frame = ttk.LabelFrame(body, text="WEIGHT-WORLD — learned carrying capacity + living population", padding=4)
        world_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 5), pady=2)
        self.world_label = ttk.Label(world_frame)
        self.world_label.pack(expand=True)
        self.world_label.bind("<Button-1>", self.select_from_click)

        view_frame = ttk.LabelFrame(body, text="WHAT THE SELECTED CREATURE'S LOCATION LOOKS LIKE IN SPLATWORLD", padding=4)
        view_frame.grid(row=0, column=1, sticky="nsew", pady=2)
        self.view_label = ttk.Label(view_frame)
        self.view_label.pack(expand=True)
        ttk.Label(view_frame, textvariable=self.creature_var, justify=tk.LEFT).pack(fill=tk.X, padx=4, pady=3)

        lower = ttk.Frame(body)
        lower.grid(row=1, column=1, sticky="nsew", pady=2)
        lower.columnconfigure(0, weight=1)
        lower.columnconfigure(1, weight=1)

        mind_frame = ttk.LabelFrame(lower, text="HOLD / NEED-TO-KNOW — selected creature", padding=3)
        mind_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        self.mind_label = ttk.Label(mind_frame)
        self.mind_label.pack(expand=True)

        plot_frame = ttk.LabelFrame(lower, text="ECOLOGY — population / energy / sensing", padding=3)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        self.plot_label = ttk.Label(plot_frame)
        self.plot_label.pack(expand=True)

    def _slider(self, parent, text, var, lo, hi, length):
        box = ttk.Frame(parent)
        box.pack(side=tk.LEFT, padx=7)
        ttk.Label(box, text=text).pack()
        ttk.Scale(box, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL, length=length).pack()

    def choose_model(self):
        path = filedialog.askopenfilename(
            title="Select SplatWorld splat_decoder.onnx",
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")],
        )
        if path:
            self._load_model_path(Path(path))

    def _load_model_path(self, path: Path):
        was_running = self.running
        self.running = False
        try:
            self.status_var.set(f"Loading SplatWorld model: {path}")
            self.root.update_idletasks()
            self.decoder = OnnxSplatDecoder(path)
            self.model_path = Path(path)
            self._build_world(mock=False)
        except Exception as exc:
            messagebox.showerror("SplatWorld model", str(exc))
            self.decoder = MockSplatDecoder(size=48)
            self.model_path = None
            self._build_world(mock=True)
        self.running = was_running

    def _cache_path(self, cfg: SubstrateConfig) -> Path | None:
        if self.model_path is None:
            return None
        stat = self.model_path.stat()
        tag = f"{stat.st_size}_{int(stat.st_mtime)}_{cfg.width}x{cfg.height}_s{cfg.seed}"
        folder = Path(".splat_ecology_cache")
        folder.mkdir(exist_ok=True)
        return folder / f"substrate_{tag}.npz"

    def _build_world(self, mock: bool):
        cfg = SubstrateConfig(width=64, height=40, batch=48, seed=7)
        cache = self._cache_path(cfg)
        try:
            if cache is not None and cache.exists():
                self.status_var.set(f"Loading cached learned world: {cache.name}")
                self.root.update_idletasks()
                substrate = LearnedSubstrate.from_cache(cache, cfg)
            else:
                def progress(done, total):
                    self.status_var.set(f"Decoding learned terrain… {done}/{total}")
                    self.root.update_idletasks()
                substrate = LearnedSubstrate.build(self.decoder, cfg, progress=progress)
                if cache is not None:
                    np.savez_compressed(cache, **substrate.cache_payload())
        except Exception as exc:
            if not mock:
                messagebox.showerror("Build SplatWorld habitat", str(exc))
                self.decoder = MockSplatDecoder(size=48)
                self.model_path = None
                return self._build_world(mock=True)
            raise
        self.substrate = substrate
        self.ecology = SplatEcology(substrate, EcologyConfig())
        self.selected_id = self.ecology.creatures[0].ident if self.ecology.creatures else None
        source = "MOCK substrate" if mock else f"REAL SplatWorld weights: {self.model_path.name}"
        self.status_var.set(f"{source} | {cfg.width}×{cfg.height} ecological slice | click a creature to inhabit its view")
        self.refresh(force_view=True)

    def _sync_params(self):
        if self.ecology is None:
            return
        self.ecology.config.info_cost = float(self.info_cost_var.get())
        self.ecology.config.regeneration = float(self.regen_var.get())
        self.ecology.config.mutation_sigma = float(self.mutation_var.get())

    def toggle(self):
        self.running = not self.running
        if self.running:
            self._schedule()

    def _schedule(self):
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
        self.after_id = self.root.after(35, self._tick)

    def _tick(self):
        self.after_id = None
        if not self.running or self.ecology is None:
            return
        self._sync_params()
        for _ in range(max(1, int(self.ticks_var.get()))):
            self.ecology.step()
            if not self.ecology.creatures:
                self.running = False
                break
        self.refresh()
        if self.running:
            self._schedule()

    def step_once(self):
        if self.ecology is None:
            return
        self._sync_params()
        self.ecology.step()
        self.refresh(force_view=True)

    def reset_life(self):
        if self.ecology is None:
            return
        self.ecology.reset()
        self.selected_id = self.ecology.creatures[0].ident if self.ecology.creatures else None
        self.refresh(force_view=True)

    def add_spores(self, n):
        if self.ecology is None:
            return
        self.ecology.add_creatures(n)
        if self.selected_id is None and self.ecology.creatures:
            self.selected_id = self.ecology.creatures[0].ident
        self.refresh(force_view=True)

    def selected(self):
        if self.ecology is None:
            return None
        for c in self.ecology.creatures:
            if c.ident == self.selected_id:
                return c
        if self.ecology.creatures:
            self.selected_id = self.ecology.creatures[0].ident
            return self.ecology.creatures[0]
        return None

    def select_from_click(self, event):
        if self.ecology is None or not self.ecology.creatures:
            return
        x = np.clip(event.x / max(1, self.world_label.winfo_width()) * self.substrate.width, 0, self.substrate.width - 1)
        y = np.clip(event.y / max(1, self.world_label.winfo_height()) * self.substrate.height, 0, self.substrate.height - 1)
        best = min(self.ecology.creatures,
                   key=lambda c: min(abs(c.x - x), self.substrate.width - abs(c.x - x)) ** 2 + (c.y - y) ** 2)
        self.selected_id = best.ident
        self.refresh(force_view=True)

    def refresh(self, force_view=False):
        if self.ecology is None:
            return
        self.frame_counter += 1
        self._set_image(self.world_label, self.render_world(), (MAP_W, MAP_H))
        self._set_image(self.mind_label, self.render_mind(), (MIND_W, MIND_H))
        self._set_image(self.plot_label, self.render_plot(), (PLOT_W, PLOT_H))
        c = self.selected()
        if c is not None and (force_view or self.frame_counter % 5 == 0 or self.last_creature_view is None):
            try:
                self.last_creature_view = render_selected(self.decoder, self.substrate, c)
            except Exception as exc:
                self.status_var.set(f"Creature-view decode failed: {exc}")
        if self.last_creature_view is not None:
            self._set_image(self.view_label, self.last_creature_view, (VIEW, VIEW))
        self._update_text(c)

    def render_world(self):
        cap = self.substrate.capacity
        ratio = self.ecology.resources / (cap + 1e-6)
        depletion = np.clip(np.mean(ratio, axis=2), 0, 1)
        rgb = np.clip(np.power(cap, 0.78) * (0.42 + 0.58 * depletion[..., None]), 0, 1)
        img = (rgb * 255).astype(np.uint8)
        img = cv2.resize(img, (MAP_W, MAP_H), interpolation=cv2.INTER_LINEAR)

        def py_from_radius(r):
            t = (r - self.substrate.config.radius_min) / (self.substrate.config.radius_max - self.substrate.config.radius_min)
            return int(np.clip(t, 0, 1) * (MAP_H - 1))
        for r, name in ((15, "CORE / GHOST"), (35, "GHOST / FIRE")):
            yy = py_from_radius(r)
            cv2.line(img, (0, yy), (MAP_W - 1, yy), (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(img, name, (8, max(15, yy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img, "identity wraps horizontally", (MAP_W - 195, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245,245,245), 1, cv2.LINE_AA)

        selected = self.selected()
        if selected is not None:
            pts = []
            for x, y in selected.trail:
                px = int((x % self.substrate.width) / self.substrate.width * MAP_W)
                py = int(y / max(1, self.substrate.height - 1) * MAP_H)
                pts.append((px, py))
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, (255,255,255), 1, cv2.LINE_AA)
            for x, y in selected.sensed:
                px = int((x + 0.5) / self.substrate.width * MAP_W)
                py = int((y + 0.5) / self.substrate.height * MAP_H)
                cv2.circle(img, (px, py), 8, (255,255,255), 1, cv2.LINE_AA)

        for c in self.ecology.creatures:
            px = int((c.x % self.substrate.width) / self.substrate.width * MAP_W)
            py = int(c.y / max(1, self.substrate.height - 1) * MAP_H)
            col = c.color_rgb()
            cv2.circle(img, (px, py), 4, col, -1, cv2.LINE_AA)
            if c.ident == self.selected_id:
                cv2.circle(img, (px, py), 8, (255,255,255), 2, cv2.LINE_AA)
        return img.astype(np.float32) / 255.0

    def render_mind(self):
        img = np.zeros((MIND_H, MIND_W, 3), np.uint8)
        c = self.selected()
        if c is None:
            cv2.putText(img, "EXTINCTION", (110, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (220,220,220), 2)
            return img.astype(np.float32)/255
        radius = 7
        cx, cy = self.substrate.index(c.x, c.y)
        cell = min(MIND_W // (2*radius+1), MIND_H // (2*radius+1))
        ox = (MIND_W - cell*(2*radius+1))//2
        oy = (MIND_H - cell*(2*radius+1))//2
        sensed = set(c.sensed)
        for j, dy in enumerate(range(-radius, radius+1)):
            for i, dx in enumerate(range(-radius, radius+1)):
                x = (cx + dx) % self.substrate.width
                y = int(np.clip(cy + dy, 0, self.substrate.height-1))
                rec = c.memory.get(y*self.substrate.width+x)
                x0, y0 = ox+i*cell, oy+j*cell
                if rec is None:
                    col = (5,5,5)
                else:
                    age = max(0, self.ecology.tick-rec.tick)
                    freshness = math.exp(-age/max(1.0,self.ecology.config.memory_horizon))
                    v = int(np.clip((rec.utility+0.15)/1.15,0,1)*220)
                    col = (int(v*0.35*freshness), int(v*freshness), int(v*0.65*freshness))
                cv2.rectangle(img,(x0,y0),(x0+cell-1,y0+cell-1),col,-1)
                if (x,y) in sensed:
                    cv2.rectangle(img,(x0,y0),(x0+cell-1,y0+cell-1),(255,255,255),1)
        cv2.rectangle(img,(ox+radius*cell,oy+radius*cell),(ox+(radius+1)*cell-1,oy+(radius+1)*cell-1),(80,180,255),2)
        cv2.putText(img,"black=unknown  bright=fresh HOLD  white=sensed now",(8,MIND_H-8),cv2.FONT_HERSHEY_SIMPLEX,0.36,(220,220,220),1,cv2.LINE_AA)
        return img.astype(np.float32)/255

    def render_plot(self):
        img = np.zeros((PLOT_H, PLOT_W, 3), np.uint8)
        hist = self.ecology.history[-220:]
        if len(hist) < 2:
            return img.astype(np.float32)/255
        pop = np.array([h.population for h in hist], np.float32)
        eng = np.array([h.mean_energy for h in hist], np.float32)
        bud = np.array([h.mean_budget for h in hist], np.float32)
        series = [(pop, (240,120,80), "population"), (eng, (80,220,130), "mean energy"), (bud, (120,160,255), "sense budget")]
        top = 16; bottom = PLOT_H-30
        for arr, col, label in series:
            vmax = max(1e-6, float(np.max(arr)))
            pts=[]
            for i,v in enumerate(arr):
                x=int(i/max(1,len(arr)-1)*(PLOT_W-18)+8)
                y=int(bottom-(v/vmax)*(bottom-top))
                pts.append((x,y))
            cv2.polylines(img,[np.array(pts,np.int32)],False,col,2,cv2.LINE_AA)
        cv2.putText(img,"population",(8,PLOT_H-14),cv2.FONT_HERSHEY_SIMPLEX,0.37,(240,120,80),1)
        cv2.putText(img,"energy",(115,PLOT_H-14),cv2.FONT_HERSHEY_SIMPLEX,0.37,(80,220,130),1)
        cv2.putText(img,"sensing",(190,PLOT_H-14),cv2.FONT_HERSHEY_SIMPLEX,0.37,(120,160,255),1)
        st=self.ecology.stats()
        cv2.putText(img,f"births {st.births}  deaths {st.deaths}  info {st.total_information}",(8,14),cv2.FONT_HERSHEY_SIMPLEX,0.38,(220,220,220),1,cv2.LINE_AA)
        return img.astype(np.float32)/255

    def _update_text(self, c):
        st = self.ecology.stats()
        source = "mock" if self.model_path is None else "SplatWorld"
        self.status_var.set(
            f"{source} | tick={st.tick} pop={st.population} births={st.births} deaths={st.deaths} "
            f"info={st.total_information} | core={st.core_count} ghost={st.ghost_count} fire={st.fire_count}"
        )
        if c is None:
            self.creature_var.set("EXTINCTION — Reset life or add spores.")
            return
        x,y=self.substrate.index(c.x,c.y)
        r=self.substrate.radius_for_y(c.y)
        p=", ".join(f"{name.lower()}={c.pref[i]:.2f}" for i,name in enumerate(RESOURCE_NAMES))
        self.creature_var.set(
            f"id={c.ident} generation={c.generation} age={c.age} energy={c.energy:.2f} shell={self.substrate.shell_for_y(c.y)} |z|={r:.1f}\n"
            f"receptors: {p} | sense budget={c.sense_budget} cells/tick | last intake={c.last_intake:.3f}\n"
            f"position=({x},{y}); white rings on the world are the cells it paid to inspect this tick."
        )

    @staticmethod
    def _set_image(label, rgb, size):
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr*255,0,255).astype(np.uint8)
        im=Image.fromarray(arr).resize(size,Image.Resampling.BILINEAR)
        tkimg=ImageTk.PhotoImage(im)
        label.configure(image=tkimg)
        label.image=tkimg

    def close(self):
        self.running=False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
        self.root.destroy()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",help="path to SplatWorld splat_decoder.onnx")
    ap.add_argument("--mock",action="store_true",help="force deterministic mock substrate")
    ap.add_argument("--headless",action="store_true",help="run a short non-GUI smoke ecology")
    ap.add_argument("--steps",type=int,default=500)
    args=ap.parse_args()
    if args.headless:
        from splat_ecology_core import run_headless
        dec=MockSplatDecoder() if args.mock or not args.model else OnnxSplatDecoder(args.model)
        st=run_headless(decoder=dec,steps=args.steps)
        print(st)
        return
    root=tk.Tk()
    SplatEcologyApp(root,model_path=args.model,force_mock=args.mock)
    root.mainloop()

if __name__=="__main__":
    main()
