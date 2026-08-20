"""Thursday Gate 2: a stateful-boundary video codec and .thv container.

The codec is intentionally small and inspectable.  Its predictor is not merely the
previous frame: it is a slowly relaxing *boundary state* that is updated from the
reconstructed stream.  Encoder and decoder therefore carry exactly the same state.

The file format is streaming and self-contained:

    THV1 | JSON header | FRM1 packet | FRM1 packet | ...

Keyframes are zlib-compressed BGR uint8 frames. Delta frames contain a sparse,
quantised innovation relative to the remembered boundary. Each frame packet stores
its rho/qstep/deadzone parameters so the membrane can be changed while recording.

This is a research/demo codec, not a replacement for H.264/AV1.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import struct
import zlib
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

MAGIC = b"THV1"
FRAME_MAGIC = b"FRM1"
VERSION = 1
HEADER_PREFIX = struct.Struct("<4sI")
FRAME_HEADER = struct.Struct("<4sIBfffII")
FLAG_KEYFRAME = 0x01


@dataclass(frozen=True)
class CodecParams:
    rho: float = 0.94
    qstep: float = 8.0
    deadzone: float = 4.0
    keyframe_interval: int = 120
    zlib_level: int = 5

    def validated(self) -> "CodecParams":
        return CodecParams(
            rho=float(np.clip(self.rho, 0.0, 0.9999)),
            qstep=max(0.25, float(self.qstep)),
            deadzone=max(0.0, float(self.deadzone)),
            keyframe_interval=max(1, int(self.keyframe_interval)),
            zlib_level=int(np.clip(self.zlib_level, 0, 9)),
        )


@dataclass
class FramePacket:
    index: int
    keyframe: bool
    rho: float
    qstep: float
    deadzone: float
    payload: bytes

    @property
    def wire_bytes(self) -> int:
        return FRAME_HEADER.size + len(self.payload)


@dataclass
class FrameDiagnostics:
    index: int
    keyframe: bool
    source: np.ndarray | None
    predictor: np.ndarray
    raw_residual: np.ndarray
    transmitted_residual: np.ndarray
    reconstruction: np.ndarray
    abs_error: np.ndarray | None
    nonzero_fraction: float
    psnr: float | None
    packet_bytes: int
    raw_bytes: int
    rho: float
    qstep: float
    deadzone: float

    @property
    def compression_ratio(self) -> float:
        return self.raw_bytes / max(1, self.packet_bytes)


def _as_u8_bgr(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("frame must have shape HxWx3")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _psnr(source: np.ndarray, recon: np.ndarray) -> float:
    err = source.astype(np.float32) - recon.astype(np.float32)
    mse = float(np.mean(err * err))
    if mse <= 1e-12:
        return math.inf
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def _encode_sparse_quantized(q: np.ndarray, level: int) -> bytes:
    flat = np.asarray(q, dtype="<i2").reshape(-1)
    mask = flat != 0
    packed_mask = np.packbits(mask, bitorder="little")
    vals = flat[mask].astype("<i2", copy=False)
    raw = struct.pack("<II", flat.size, vals.size) + packed_mask.tobytes() + vals.tobytes()
    return zlib.compress(raw, level)


def _decode_sparse_quantized(payload: bytes, shape: tuple[int, int, int]) -> np.ndarray:
    raw = zlib.decompress(payload)
    if len(raw) < 8:
        raise ValueError("truncated sparse residual")
    total, count = struct.unpack_from("<II", raw, 0)
    expected = int(np.prod(shape))
    if total != expected:
        raise ValueError(f"residual size mismatch: packet={total}, expected={expected}")
    mask_len = (total + 7) // 8
    start = 8
    stop = start + mask_len
    if len(raw) < stop + count * 2:
        raise ValueError("truncated sparse residual values")
    mask = np.unpackbits(np.frombuffer(raw[start:stop], dtype=np.uint8), bitorder="little")[:total].astype(bool)
    if int(mask.sum()) != count:
        raise ValueError("sparse residual count mismatch")
    values = np.frombuffer(raw[stop:stop + count * 2], dtype="<i2")
    flat = np.zeros(total, dtype=np.int16)
    flat[mask] = values
    return flat.reshape(shape)


class BoundaryEncoder:
    def __init__(self, width: int, height: int, params: CodecParams | None = None):
        self.width = int(width)
        self.height = int(height)
        self.params = (params or CodecParams()).validated()
        self.boundary: np.ndarray | None = None
        self.frame_index = 0
        self.force_keyframe_next = True

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

    def reset(self) -> None:
        self.boundary = None
        self.frame_index = 0
        self.force_keyframe_next = True

    def force_keyframe(self) -> None:
        self.force_keyframe_next = True

    def set_params(self, params: CodecParams) -> None:
        self.params = params.validated()

    def _update_boundary(self, recon: np.ndarray, rho: float) -> None:
        rf = recon.astype(np.float32)
        if self.boundary is None:
            self.boundary = rf.copy()
        else:
            self.boundary = (np.float32(rho) * self.boundary + np.float32(1.0 - rho) * rf).astype(np.float32)

    def encode(self, frame: np.ndarray) -> tuple[FramePacket, FrameDiagnostics]:
        src = _as_u8_bgr(frame)
        if src.shape != self.shape:
            raise ValueError(f"frame shape {src.shape} does not match codec shape {self.shape}")
        p = self.params.validated()
        idx = self.frame_index
        key = self.force_keyframe_next or self.boundary is None or idx % p.keyframe_interval == 0

        if key:
            payload = zlib.compress(src.tobytes(), p.zlib_level)
            predictor = src.astype(np.float32)
            raw_residual = np.zeros_like(predictor, dtype=np.float32)
            transmitted = np.zeros_like(predictor, dtype=np.float32)
            recon = src.copy()
            nonzero = 0.0
        else:
            predictor = self.boundary.copy()
            raw_residual = src.astype(np.float32) - predictor
            q = np.rint(raw_residual / np.float32(p.qstep)).astype(np.int16)
            q[np.abs(raw_residual) < np.float32(p.deadzone)] = 0
            payload = _encode_sparse_quantized(q, p.zlib_level)
            transmitted = q.astype(np.float32) * np.float32(p.qstep)
            recon_f = np.clip(predictor + transmitted, 0.0, 255.0)
            recon = np.rint(recon_f).astype(np.uint8)
            nonzero = float(np.count_nonzero(q)) / float(q.size)

        packet = FramePacket(idx, key, p.rho, p.qstep, p.deadzone, payload)
        if key:
            self.boundary = recon.astype(np.float32).copy()
        else:
            self._update_boundary(recon, p.rho)
        error = np.abs(src.astype(np.float32) - recon.astype(np.float32))
        diag = FrameDiagnostics(
            index=idx,
            keyframe=key,
            source=src,
            predictor=np.clip(predictor, 0, 255).astype(np.uint8),
            raw_residual=raw_residual,
            transmitted_residual=transmitted,
            reconstruction=recon,
            abs_error=error,
            nonzero_fraction=nonzero,
            psnr=_psnr(src, recon),
            packet_bytes=packet.wire_bytes,
            raw_bytes=src.nbytes,
            rho=p.rho,
            qstep=p.qstep,
            deadzone=p.deadzone,
        )
        self.frame_index += 1
        self.force_keyframe_next = False
        return packet, diag


class BoundaryDecoder:
    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.boundary: np.ndarray | None = None
        self.expected_index = 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

    def reset(self) -> None:
        self.boundary = None
        self.expected_index = 0

    def _update_boundary(self, recon: np.ndarray, rho: float) -> None:
        rf = recon.astype(np.float32)
        if self.boundary is None:
            self.boundary = rf.copy()
        else:
            self.boundary = (np.float32(rho) * self.boundary + np.float32(1.0 - rho) * rf).astype(np.float32)

    def decode(self, packet: FramePacket) -> FrameDiagnostics:
        if self.boundary is None and packet.keyframe:
            self.expected_index = packet.index
        if packet.index != self.expected_index:
            raise ValueError(f"packet index {packet.index}, expected {self.expected_index}")
        if packet.keyframe:
            raw = zlib.decompress(packet.payload)
            expected = int(np.prod(self.shape))
            if len(raw) != expected:
                raise ValueError("keyframe byte count mismatch")
            recon = np.frombuffer(raw, dtype=np.uint8).copy().reshape(self.shape)
            predictor = recon.astype(np.float32)
            raw_residual = np.zeros_like(predictor, dtype=np.float32)
            transmitted = np.zeros_like(predictor, dtype=np.float32)
            nonzero = 0.0
        else:
            if self.boundary is None:
                raise ValueError("delta frame before keyframe")
            predictor = self.boundary.copy()
            q = _decode_sparse_quantized(packet.payload, self.shape)
            transmitted = q.astype(np.float32) * np.float32(packet.qstep)
            raw_residual = transmitted.copy()
            recon = np.rint(np.clip(predictor + transmitted, 0.0, 255.0)).astype(np.uint8)
            nonzero = float(np.count_nonzero(q)) / float(q.size)

        if packet.keyframe:
            self.boundary = recon.astype(np.float32).copy()
        else:
            self._update_boundary(recon, packet.rho)
        diag = FrameDiagnostics(
            index=packet.index,
            keyframe=packet.keyframe,
            source=None,
            predictor=np.clip(predictor, 0, 255).astype(np.uint8),
            raw_residual=raw_residual,
            transmitted_residual=transmitted,
            reconstruction=recon,
            abs_error=None,
            nonzero_fraction=nonzero,
            psnr=None,
            packet_bytes=packet.wire_bytes,
            raw_bytes=recon.nbytes,
            rho=packet.rho,
            qstep=packet.qstep,
            deadzone=packet.deadzone,
        )
        self.expected_index += 1
        return diag


class THVWriter:
    def __init__(self, path: str | Path, width: int, height: int, fps: float, metadata: dict | None = None):
        self.path = Path(path)
        self.fp: BinaryIO = self.path.open("wb")
        self.header = {
            "version": VERSION,
            "width": int(width),
            "height": int(height),
            "fps": float(fps),
            "channels": 3,
            "color": "BGR8",
            "codec": "stateful-boundary-sparse-residual",
        }
        if metadata:
            self.header["user"] = metadata
        blob = json.dumps(self.header, separators=(",", ":"), sort_keys=True).encode("utf-8")
        self.fp.write(HEADER_PREFIX.pack(MAGIC, len(blob)))
        self.fp.write(blob)
        self.frames = 0
        self.bytes_written = HEADER_PREFIX.size + len(blob)
        self.closed = False

    def write(self, packet: FramePacket) -> int:
        if self.closed:
            raise ValueError("writer is closed")
        flags = FLAG_KEYFRAME if packet.keyframe else 0
        crc = zlib.crc32(packet.payload) & 0xFFFFFFFF
        head = FRAME_HEADER.pack(
            FRAME_MAGIC,
            int(packet.index),
            flags,
            float(packet.rho),
            float(packet.qstep),
            float(packet.deadzone),
            len(packet.payload),
            crc,
        )
        self.fp.write(head)
        self.fp.write(packet.payload)
        n = len(head) + len(packet.payload)
        self.bytes_written += n
        self.frames += 1
        return n

    def close(self) -> None:
        if not self.closed:
            self.fp.flush()
            self.fp.close()
            self.closed = True

    def __enter__(self) -> "THVWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class THVReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.fp: BinaryIO = self.path.open("rb")
        prefix = self.fp.read(HEADER_PREFIX.size)
        if len(prefix) != HEADER_PREFIX.size:
            raise ValueError("truncated THV header")
        magic, n = HEADER_PREFIX.unpack(prefix)
        if magic != MAGIC:
            raise ValueError("not a THV1 file")
        blob = self.fp.read(n)
        if len(blob) != n:
            raise ValueError("truncated THV metadata")
        self.header = json.loads(blob.decode("utf-8"))
        if int(self.header.get("version", -1)) != VERSION:
            raise ValueError(f"unsupported THV version {self.header.get('version')}")
        self.closed = False

    @property
    def width(self) -> int:
        return int(self.header["width"])

    @property
    def height(self) -> int:
        return int(self.header["height"])

    @property
    def fps(self) -> float:
        return float(self.header.get("fps", 30.0))

    def read_packet(self) -> FramePacket | None:
        head = self.fp.read(FRAME_HEADER.size)
        if not head:
            return None
        if len(head) != FRAME_HEADER.size:
            raise ValueError("truncated THV frame header")
        magic, idx, flags, rho, qstep, deadzone, n, crc = FRAME_HEADER.unpack(head)
        if magic != FRAME_MAGIC:
            raise ValueError("invalid THV frame marker")
        payload = self.fp.read(n)
        if len(payload) != n:
            raise ValueError("truncated THV frame payload")
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            raise ValueError(f"CRC mismatch in frame {idx}")
        return FramePacket(
            index=int(idx),
            keyframe=bool(flags & FLAG_KEYFRAME),
            rho=float(rho),
            qstep=float(qstep),
            deadzone=float(deadzone),
            payload=payload,
        )

    def packets(self) -> Iterator[FramePacket]:
        while True:
            packet = self.read_packet()
            if packet is None:
                return
            yield packet

    def close(self) -> None:
        if not self.closed:
            self.fp.close()
            self.closed = True

    def __enter__(self) -> "THVReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def encode_frames_to_thv(frames: list[np.ndarray], path: str | Path, fps: float = 30.0, params: CodecParams | None = None) -> list[FrameDiagnostics]:
    if not frames:
        raise ValueError("frames is empty")
    first = _as_u8_bgr(frames[0])
    h, w = first.shape[:2]
    enc = BoundaryEncoder(w, h, params)
    out: list[FrameDiagnostics] = []
    with THVWriter(path, w, h, fps) as writer:
        for frame in frames:
            packet, diag = enc.encode(frame)
            writer.write(packet)
            out.append(diag)
    return out


def decode_thv(path: str | Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with THVReader(path) as reader:
        dec = BoundaryDecoder(reader.width, reader.height)
        for packet in reader.packets():
            frames.append(dec.decode(packet).reconstruction)
    return frames
