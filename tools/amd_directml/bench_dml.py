"""Benchmark + quality check of the RVC pipeline stages on DirectML.

Runs the same steps VCClient does for one chunk (content vec -> RMVPE pitch ->
RVC synth), times each stage on the GPU, and compares an fp32 model with its
fp16 conversion. Also writes the converted test file so you can listen.

Usage:
    python bench_dml.py <pretrain_dir> <model_fp32.onnx> <model_fp16.onnx> <input.wav> <out_dir> [tune] [gpu]
Needs: pip install onnx onnxruntime-directml numpy
"""
import os
import sys
import time
import wave

import numpy as np
import onnx
import onnxruntime as ort

pretrain, m32, m16, wav_in, out_dir = sys.argv[1:6]
TUNE = int(sys.argv[6]) if len(sys.argv) > 6 else 12
GPU = int(sys.argv[7]) if len(sys.argv) > 7 else 0
os.makedirs(out_dir, exist_ok=True)


def session(path_or_bytes, provider="dml"):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if provider == "dml":
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return ort.InferenceSession(path_or_bytes, so, providers=["DmlExecutionProvider"], provider_options=[{"device_id": GPU}])
    return ort.InferenceSession(path_or_bytes, so, providers=["CPUExecutionProvider"])


def seeded(path):
    # Fix the random-noise seeds so fp32 and fp16 can be compared sample by sample.
    m = onnx.load(path)
    for n in m.graph.node:
        if n.op_type in ("RandomNormalLike", "RandomUniformLike"):
            for a in list(n.attribute):
                if a.name == "seed":
                    n.attribute.remove(a)
            n.attribute.append(onnx.helper.make_attribute("seed", 1.0))
    return m.SerializeToString()


def read_wav(path):
    with wave.open(path) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768
    return x.reshape(-1, ch).mean(1), sr


def resample(x, sr_in, sr_out):
    # Windowed-sinc resampler (good enough for a benchmark).
    from math import gcd

    g = gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    taps = 64 * max(up, down) + 1
    cutoff = 0.5 / max(up, down)
    t = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * cutoff * t) * 2 * cutoff * np.hamming(taps) * up
    y = np.zeros(len(x) * up, dtype=np.float32)
    y[::up] = x
    y = np.convolve(y, h.astype(np.float32), mode="same")
    return y[::down].astype(np.float32)


def write_wav(path, x, sr):
    x = np.clip(x, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def timeit(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


T0 = time.perf_counter()
def step(msg):
    print(f"[{time.perf_counter()-T0:6.1f}s] {msg}", flush=True)

step("loading content vec (gpu)")
cv = session(os.path.join(pretrain, "content_vec_500.onnx"))
step("loading content vec (cpu)")
cv_cpu = session(os.path.join(pretrain, "content_vec_500.onnx"), "cpu")
step("loading rmvpe")
rm = session(os.path.join(pretrain, "rmvpe.onnx"))
step("loading fp32 model")
s32 = session(seeded(m32))
step("loading fp16 model")
s16 = session(seeded(m16))
model_sr = 32000
for p in onnx.load(m32, load_external_data=False).metadata_props:
    if p.key == "metadata":
        import json

        model_sr = json.loads(p.value).get("samplingRate", model_sr)


def stages(audio16):
    a = audio16[np.newaxis, :]
    feats = cv.run(["unit12"], {"audio": a})[0]
    feats = np.repeat(feats, 2, axis=1)  # RVC doubles the frame rate
    f0 = rm.run(["pitchf"], {"waveform": a, "threshold": np.array([0.3], np.float32)})[0].squeeze()
    f0 = f0 * 2 ** (TUNE / 12)
    n = min(feats.shape[1], len(f0))
    feats, f0 = feats[:, :n], f0[:n].astype(np.float32)
    mel = 1127 * np.log(1 + f0 / 700)
    mmin, mmax = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
    mel[mel > 0] = (mel[mel > 0] - mmin) * 254 / (mmax - mmin) + 1
    coarse = np.rint(np.clip(mel, 1, 255)).astype(np.int64)
    return {
        "feats": feats.astype(np.float32),
        "p_len": np.array([n], np.int64),
        "pitch": coarse[np.newaxis, :],
        "pitchf": f0[np.newaxis, :],
        "sid": np.array([0], np.int64),
    }


step("models loaded")
x, sr = read_wav(wav_in)
x16 = resample(x, sr, 16000)

# One chunk the size VCClient uses with chunk=256 + extra=16384 (~1.02 s at 16 kHz).
chunk = x16[16000 : 16000 + 16384]
inputs = stages(chunk)
a = chunk[np.newaxis, :]

print(f"DirectML device {GPU}, one chunk of {len(chunk)/16000*1000:.0f} ms audio, tune {TUNE:+d}")
print(f"  content vec  CPU (like stock app) : {timeit(lambda: cv_cpu.run(['unit12'], {'audio': a}), n=10):7.1f} ms")
print(f"  content vec  GPU (patched)        : {timeit(lambda: cv.run(['unit12'], {'audio': a})):7.1f} ms")
print(f"  rmvpe pitch  GPU                  : {timeit(lambda: rm.run(['pitchf'], {'waveform': a, 'threshold': np.array([0.3], np.float32)})):7.1f} ms")
print(f"  RVC synth    GPU fp32 (current)   : {timeit(lambda: s32.run(['audio'], inputs)):7.1f} ms")
print(f"  RVC synth    GPU fp16 (converted) : {timeit(lambda: s16.run(['audio'], inputs)):7.1f} ms")

step("timing done")
# Quality: whole file through both models, same noise seed.
full = stages(x16)
y32 = np.asarray(s32.run(["audio"], full)[0]).reshape(-1)
y16 = np.asarray(s16.run(["audio"], full)[0]).reshape(-1)
n = min(len(y32), len(y16))
err = y32[:n] - y16[:n]
snr = 10 * np.log10(np.sum(y32[:n] ** 2) / max(np.sum(err**2), 1e-12))
print(f"fp16 vs fp32 difference: SNR {snr:.1f} dB (above ~30 dB = sounds the same)")
print(f"nan/inf in fp16 output: {bool(~np.isfinite(y16).all())}, peak fp32 {np.abs(y32).max():.2f} fp16 {np.abs(y16).max():.2f}")

write_wav(os.path.join(out_dir, "test_input.wav"), x, sr)
write_wav(os.path.join(out_dir, "egirl_fp32.wav"), y32, model_sr)
write_wav(os.path.join(out_dir, "egirl_fp16.wav"), y16, model_sr)
print("wrote", out_dir)
