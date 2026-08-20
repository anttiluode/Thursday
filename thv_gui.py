"""GUI for Thursday Gate 2 stateful-boundary codec.

Open a video, open a webcam, or play a .thv file. During live/video input the
interface shows the source, remembered boundary, innovation, reconstruction,
error and actually transmitted residual. The stream can be recorded directly to
Thursday's own .thv container.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from thv_codec import BoundaryDecoder, BoundaryEncoder, CodecParams, THVReader, THVWriter


class BoundaryCodecApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Thursday Gate 2 — Stateful Boundary Codec (.thv)")
        self.root.geometry("1500x980")

        self.capture: cv2.VideoCapture | None = None
        self.reader: THVReader | None = None
        self.reader_decoder: BoundaryDecoder | None = None
        self.encoder: BoundaryEncoder | None = None
        self.writer: THVWriter | None = None
        self.source_kind = "none"
        self.source_name = "none"
        self.playing = False
        self.pending_after_id = None
        self.last_tick = None
        self.measured_fps = 0.0
        self.source_fps = 30.0
        self.codec_size: tuple[int, int] | None = None
        self.frames_processed = 0
        self.total_wire_bytes = 0
        self.total_raw_bytes = 0
        self.last_diag = None

        self.rho_var = tk.DoubleVar(value=0.94)
        self.qstep_var = tk.DoubleVar(value=8.0)
        self.deadzone_var = tk.DoubleVar(value=4.0)
        self.keyint_var = tk.IntVar(value=120)
        self.width_var = tk.IntVar(value=320)
        self.status_var = tk.StringVar(value="Open a video or webcam. The boundary remembers.")
        self.record_var = tk.StringVar(value="Record .thv")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        source = ttk.LabelFrame(outer, text="Source / recording", padding=6)
        source.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(source, text="Open video", command=self.open_video).pack(side=tk.LEFT, padx=3)
        ttk.Button(source, text="Webcam", command=self.open_webcam).pack(side=tk.LEFT, padx=3)
        ttk.Button(source, text="Open .thv", command=self.open_thv).pack(side=tk.LEFT, padx=3)
        ttk.Button(source, textvariable=self.record_var, command=self.toggle_record).pack(side=tk.LEFT, padx=12)
        ttk.Button(source, text="Force keyframe", command=self.force_keyframe).pack(side=tk.LEFT, padx=3)
        ttk.Button(source, text="Step", command=self.step_once).pack(side=tk.LEFT, padx=12)
        ttk.Button(source, text="Play / pause", command=self.toggle_play).pack(side=tk.LEFT, padx=3)
        ttk.Button(source, text="Reset state", command=self.reset_codec_state).pack(side=tk.LEFT, padx=3)

        params = ttk.LabelFrame(outer, text="Boundary / transmission controls", padding=6)
        params.pack(fill=tk.X, pady=(0, 6))
        self._slider(params, "memory rho", self.rho_var, 0.0, 0.995, 240, 0)
        self._slider(params, "q step", self.qstep_var, 0.5, 48.0, 210, 1)
        self._slider(params, "deadzone", self.deadzone_var, 0.0, 80.0, 210, 2)

        ttk.Label(params, text="keyframe interval").grid(row=0, column=6, sticky="e", padx=(10, 2))
        ttk.Spinbox(params, from_=1, to=10000, textvariable=self.keyint_var, width=7).grid(row=0, column=7, sticky="w")
        ttk.Label(params, text="working width").grid(row=0, column=8, sticky="e", padx=(10, 2))
        ttk.Combobox(params, textvariable=self.width_var, width=6, state="readonly", values=[160, 240, 320, 480, 640, 960]).grid(row=0, column=9, sticky="w")
        ttk.Label(params, text="High rho + high deadzone = long, cheap memory / obvious ghosts").grid(row=1, column=6, columnspan=4, sticky="w", padx=10)

        ttk.Label(outer, textvariable=self.status_var, anchor="w").pack(fill=tk.X, pady=(0, 6))

        grid = ttk.Frame(outer)
        grid.pack(fill=tk.BOTH, expand=True)
        for r in range(2):
            grid.rowconfigure(r, weight=1)
        for c in range(3):
            grid.columnconfigure(c, weight=1)

        self.panels = {}
        specs = [
            ("source", "WORLD — current external frame", 0, 0),
            ("boundary", "BOUNDARY MEMORY — predictor before new evidence", 0, 1),
            ("innovation", "RAW INNOVATION — what current world asks to change", 0, 2),
            ("recon", "DECODED WORLD — after admitted innovation", 1, 0),
            ("error", "ERROR — what the codec currently gets wrong", 1, 1),
            ("transmit", "TRANSMITTED RESIDUAL — what actually crossed", 1, 2),
        ]
        for name, title, row, col in specs:
            lf = ttk.LabelFrame(grid, text=title, padding=3)
            lf.grid(row=row, column=col, sticky="nsew", padx=3, pady=3)
            label = ttk.Label(lf, anchor="center")
            label.pack(fill=tk.BOTH, expand=True)
            self.panels[name] = label

    def _slider(self, parent, label, var, lo, hi, length, col):
        ttk.Label(parent, text=label).grid(row=0, column=col * 2, sticky="e", padx=(3, 2))
        s = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL, length=length)
        s.grid(row=0, column=col * 2 + 1, sticky="w")
        value = ttk.Label(parent, width=7)
        value.grid(row=1, column=col * 2 + 1, sticky="w")
        def update(*_):
            value.config(text=f"{var.get():.3g}")
        var.trace_add("write", update)
        update()

    def close_source(self):
        self.playing = False
        if self.pending_after_id is not None:
            try:
                self.root.after_cancel(self.pending_after_id)
            except Exception:
                pass
            self.pending_after_id = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        self.reader_decoder = None
        self.encoder = None
        self.codec_size = None
        self.source_kind = "none"
        self.stop_recording()

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Open video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"), ("All files", "*.*")],
        )
        if not path:
            return
        self.close_source()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Thursday", f"Could not open {path}")
            return
        self.capture = cap
        self.source_kind = "video"
        self.source_name = Path(path).name
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.source_fps = fps if fps > 1e-3 else 30.0
        self._reset_counters()
        self.playing = True
        self._schedule(1)

    def open_webcam(self):
        self.close_source()
        if os.name == "nt":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Thursday", "Could not open webcam 0")
            return
        self.capture = cap
        self.source_kind = "webcam"
        self.source_name = "webcam 0"
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.source_fps = fps if fps > 1e-3 else 30.0
        self._reset_counters()
        self.playing = True
        self._schedule(1)

    def open_thv(self):
        path = filedialog.askopenfilename(title="Open Thursday video", filetypes=[("Thursday video", "*.thv"), ("All files", "*.*")])
        if not path:
            return
        self.close_source()
        try:
            reader = THVReader(path)
        except Exception as e:
            messagebox.showerror("Thursday", str(e))
            return
        self.reader = reader
        self.reader_decoder = BoundaryDecoder(reader.width, reader.height)
        self.source_kind = "thv"
        self.source_name = Path(path).name
        self.source_fps = reader.fps if reader.fps > 1e-3 else 30.0
        self.codec_size = (reader.width, reader.height)
        self._reset_counters()
        self.playing = True
        self._schedule(1)

    def _reset_counters(self):
        self.frames_processed = 0
        self.total_wire_bytes = 0
        self.total_raw_bytes = 0
        self.last_tick = None
        self.measured_fps = 0.0
        self.last_diag = None

    def _target_size(self, frame: np.ndarray) -> tuple[int, int]:
        h, w = frame.shape[:2]
        target_w = min(int(self.width_var.get()), w)
        target_h = max(2, int(round(h * target_w / w)))
        target_w -= target_w % 2
        target_h -= target_h % 2
        return max(2, target_w), max(2, target_h)

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.codec_size is None:
            self.codec_size = self._target_size(frame)
        w, h = self.codec_size
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def current_params(self) -> CodecParams:
        return CodecParams(
            rho=self.rho_var.get(),
            qstep=self.qstep_var.get(),
            deadzone=self.deadzone_var.get(),
            keyframe_interval=self.keyint_var.get(),
            zlib_level=5,
        ).validated()

    def _ensure_encoder(self, frame: np.ndarray):
        if self.encoder is None:
            h, w = frame.shape[:2]
            self.encoder = BoundaryEncoder(w, h, self.current_params())
        else:
            self.encoder.set_params(self.current_params())

    def force_keyframe(self):
        if self.encoder is not None:
            self.encoder.force_keyframe()
            self.status_var.set("Next live/video frame will be a keyframe.")

    def reset_codec_state(self):
        if self.source_kind == "thv":
            messagebox.showinfo("Thursday", "For .thv playback, reopen the file to restart the decoder from frame 0.")
            return
        if self.encoder is not None:
            self.encoder.reset()
        self.status_var.set("Boundary state reset. Next frame starts a new memory.")

    def toggle_record(self):
        if self.writer is not None:
            self.stop_recording()
            return
        if self.source_kind not in ("video", "webcam") or self.codec_size is None:
            messagebox.showinfo("Thursday", "Start a video/webcam source first and let at least one frame arrive.")
            return
        path = filedialog.asksaveasfilename(title="Save Thursday video", defaultextension=".thv", filetypes=[("Thursday video", "*.thv")])
        if not path:
            return
        w, h = self.codec_size
        self.writer = THVWriter(path, w, h, self.source_fps, metadata={"source": self.source_name})
        if self.encoder is not None:
            self.encoder.force_keyframe()
        self.record_var.set("Stop recording")
        self.status_var.set(f"Recording to {Path(path).name}; next frame forced to keyframe.")

    def stop_recording(self):
        if self.writer is not None:
            frames = self.writer.frames
            bytes_written = self.writer.bytes_written
            path = self.writer.path
            self.writer.close()
            self.writer = None
            self.record_var.set("Record .thv")
            self.status_var.set(f"Saved {frames} frames, {bytes_written / 1024:.1f} KiB → {path}")

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self._schedule(1)

    def step_once(self):
        if self.source_kind == "none":
            return
        self.playing = False
        self._process_one()

    def _schedule(self, delay_ms: int | None = None):
        if not self.playing:
            return
        if self.pending_after_id is not None:
            return
        if delay_ms is None:
            delay_ms = max(1, int(round(1000.0 / max(1.0, self.source_fps))))
        self.pending_after_id = self.root.after(delay_ms, self._scheduled_tick)

    def _scheduled_tick(self):
        self.pending_after_id = None
        if not self.playing:
            return
        alive = self._process_one()
        if alive and self.playing:
            self._schedule()

    def _process_one(self) -> bool:
        t0 = time.perf_counter()
        try:
            if self.source_kind in ("video", "webcam"):
                if self.capture is None:
                    return False
                ok, frame = self.capture.read()
                if not ok:
                    if self.source_kind == "video":
                        self.playing = False
                        self.stop_recording()
                        self.status_var.set("End of video.")
                    return False
                frame = self._prepare_frame(frame)
                self._ensure_encoder(frame)
                packet, diag = self.encoder.encode(frame)
                if self.writer is not None:
                    self.writer.write(packet)
                self._display_live(diag)
            elif self.source_kind == "thv":
                if self.reader is None or self.reader_decoder is None:
                    return False
                packet = self.reader.read_packet()
                if packet is None:
                    self.playing = False
                    self.status_var.set("End of .thv file.")
                    return False
                diag = self.reader_decoder.decode(packet)
                self._display_thv(diag)
            else:
                return False

            self.frames_processed += 1
            self.total_wire_bytes += diag.packet_bytes
            self.total_raw_bytes += diag.raw_bytes
            self.last_diag = diag
            now = time.perf_counter()
            dt = now - (self.last_tick or now)
            if dt > 0:
                inst = 1.0 / dt
                self.measured_fps = inst if self.measured_fps == 0 else 0.9 * self.measured_fps + 0.1 * inst
            self.last_tick = now
            self._update_status(diag, time.perf_counter() - t0)
            return True
        except Exception as e:
            self.playing = False
            self.stop_recording()
            messagebox.showerror("Thursday codec error", str(e))
            return False

    @staticmethod
    def _innovation_image(residual: np.ndarray) -> np.ndarray:
        r = residual.astype(np.float32)
        mag = np.mean(np.abs(r), axis=2)
        sign = np.mean(r, axis=2)
        scale = max(8.0, float(np.percentile(mag, 99.5)))
        norm = np.clip(mag / scale, 0, 1)
        out = np.zeros((*mag.shape, 3), dtype=np.uint8)
        pos = sign >= 0
        out[..., 1] = (norm * 100).astype(np.uint8)
        out[..., 2][pos] = (norm[pos] * 255).astype(np.uint8)
        out[..., 0][~pos] = (norm[~pos] * 255).astype(np.uint8)
        return out

    @staticmethod
    def _heat_image(gray: np.ndarray) -> np.ndarray:
        g = np.asarray(gray, dtype=np.float32)
        scale = max(1.0, float(np.percentile(g, 99.5)))
        u = np.clip(g / scale * 255.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(u, cv2.COLORMAP_INFERNO)

    def _display_live(self, d):
        self._show("source", d.source)
        self._show("boundary", d.predictor)
        self._show("innovation", self._innovation_image(d.raw_residual))
        self._show("recon", d.reconstruction)
        self._show("error", self._heat_image(np.mean(d.abs_error, axis=2)))
        self._show("transmit", self._innovation_image(d.transmitted_residual))

    def _display_thv(self, d):
        blank = np.zeros_like(d.reconstruction)
        cv2.putText(blank, "SOURCE NOT STORED", (10, max(30, blank.shape[0] // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        self._show("source", blank)
        self._show("boundary", d.predictor)
        self._show("innovation", self._innovation_image(d.raw_residual))
        self._show("recon", d.reconstruction)
        self._show("error", blank)
        self._show("transmit", self._innovation_image(d.transmitted_residual))

    def _show(self, name: str, bgr: np.ndarray | None):
        if bgr is None:
            return
        rgb = cv2.cvtColor(np.asarray(bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        max_w, max_h = 450, 340
        scale = min(max_w / pil.width, max_h / pil.height, 1.0)
        if scale != 1.0:
            pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(pil)
        label = self.panels[name]
        label.configure(image=img)
        label.image = img

    def _update_status(self, d, proc_time: float):
        ratio = self.total_raw_bytes / max(1, self.total_wire_bytes)
        psnr = "n/a" if d.psnr is None else ("inf" if math.isinf(d.psnr) else f"{d.psnr:.1f} dB")
        rec = "REC" if self.writer is not None else ""
        kind = "KEY" if d.keyframe else "delta"
        self.status_var.set(
            f"{rec} {self.source_kind}:{self.source_name} | frame={d.index} {kind} | "
            f"rho={d.rho:.3f} q={d.qstep:.2f} dead={d.deadzone:.1f} | "
            f"innovation sent={100*d.nonzero_fraction:.1f}% | PSNR={psnr} | "
            f"packet={d.packet_bytes/1024:.1f} KiB | cumulative raw/.thv={ratio:.2f}x | "
            f"loop={proc_time*1000:.1f} ms ({self.measured_fps:.1f} fps)"
        )

    def on_close(self):
        self.close_source()
        self.root.destroy()


def main():
    root = tk.Tk()
    BoundaryCodecApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
