from pathlib import Path
import numpy as np

from thv_codec import (
    BoundaryDecoder,
    BoundaryEncoder,
    CodecParams,
    THVReader,
    THVWriter,
    decode_thv,
)


def frames(n=8, h=24, w=32):
    out = []
    for i in range(n):
        x = np.zeros((h, w, 3), dtype=np.uint8)
        x[:] = (20 + i, 30, 40)
        x[4:12, 3 + i:11 + i] = (220, 180, 40)
        out.append(x)
    return out


def test_keyframe_is_exact():
    src = frames(1)[0]
    enc = BoundaryEncoder(src.shape[1], src.shape[0], CodecParams())
    dec = BoundaryDecoder(src.shape[1], src.shape[0])
    packet, ediag = enc.encode(src)
    ddiag = dec.decode(packet)
    assert packet.keyframe
    assert np.array_equal(ediag.reconstruction, src)
    assert np.array_equal(ddiag.reconstruction, src)
    assert np.array_equal(enc.boundary, dec.boundary)


def test_encoder_decoder_boundary_stays_in_sync():
    fs = frames(12)
    params = CodecParams(rho=0.91, qstep=7, deadzone=3, keyframe_interval=50)
    enc = BoundaryEncoder(fs[0].shape[1], fs[0].shape[0], params)
    dec = BoundaryDecoder(fs[0].shape[1], fs[0].shape[0])
    for f in fs:
        packet, ediag = enc.encode(f)
        ddiag = dec.decode(packet)
        assert np.array_equal(ediag.reconstruction, ddiag.reconstruction)
        assert np.array_equal(enc.boundary, dec.boundary)


def test_thv_file_roundtrip(tmp_path: Path):
    fs = frames(10)
    p = tmp_path / "roundtrip.thv"
    params = CodecParams(rho=0.95, qstep=10, deadzone=5, keyframe_interval=99)
    enc = BoundaryEncoder(fs[0].shape[1], fs[0].shape[0], params)
    expected = []
    with THVWriter(p, fs[0].shape[1], fs[0].shape[0], 25.0) as writer:
        for f in fs:
            packet, diag = enc.encode(f)
            writer.write(packet)
            expected.append(diag.reconstruction.copy())
    actual = decode_thv(p)
    assert len(actual) == len(expected)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
    with THVReader(p) as reader:
        assert reader.width == fs[0].shape[1]
        assert reader.height == fs[0].shape[0]
        assert reader.fps == 25.0


def test_boundary_memory_can_create_afterimage_when_innovation_is_suppressed():
    h, w = 24, 32
    gray = np.full((h, w, 3), 80, dtype=np.uint8)
    bright = gray.copy()
    bright[6:18, 8:24] = 230
    params = CodecParams(rho=0.90, qstep=16, deadzone=300, keyframe_interval=999)
    enc = BoundaryEncoder(w, h, params)
    enc.encode(bright)
    packet, diag = enc.encode(gray)
    assert not packet.keyframe
    assert float(diag.reconstruction[10, 12].mean()) > 180
    assert float(gray[10, 12].mean()) == 80


def test_parameter_changes_are_carried_in_packets():
    fs = frames(3)
    enc = BoundaryEncoder(fs[0].shape[1], fs[0].shape[0], CodecParams(rho=0.8, qstep=6, deadzone=2, keyframe_interval=99))
    dec = BoundaryDecoder(fs[0].shape[1], fs[0].shape[0])
    p0, _ = enc.encode(fs[0])
    dec.decode(p0)
    enc.set_params(CodecParams(rho=0.97, qstep=20, deadzone=9, keyframe_interval=99))
    p1, e1 = enc.encode(fs[1])
    d1 = dec.decode(p1)
    assert abs(p1.rho - 0.97) < 1e-6
    assert abs(p1.qstep - 20.0) < 1e-6
    assert np.array_equal(e1.reconstruction, d1.reconstruction)
    assert np.array_equal(enc.boundary, dec.boundary)


def test_decoder_accepts_recording_that_starts_mid_session(tmp_path: Path):
    fs = frames(6)
    enc = BoundaryEncoder(fs[0].shape[1], fs[0].shape[0], CodecParams(keyframe_interval=99))
    for f in fs[:3]:
        enc.encode(f)
    enc.force_keyframe()
    p = tmp_path / "midstream.thv"
    expected = []
    with THVWriter(p, fs[0].shape[1], fs[0].shape[0], 30.0) as writer:
        for f in fs[3:]:
            packet, diag = enc.encode(f)
            writer.write(packet)
            expected.append(diag.reconstruction.copy())
    actual = decode_thv(p)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
