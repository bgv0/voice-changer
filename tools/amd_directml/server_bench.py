"""Time a running VCClient v1.5.3 server end to end through its /test endpoint.

Sends test audio in real chunk-sized pieces (like the browser client does) and
reports how long each conversion takes, plus the server's own /performance
numbers. Optional: write the converted audio to a wav so you can listen.

Usage:
    python server_bench.py input.wav [chunk_x128=256] [n_chunks=30] [out.wav] [port=18888]
Standard library + numpy only.
"""
import base64
import json
import sys
import time
import urllib.request
import wave

import numpy as np

wav_in = sys.argv[1]
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 256
N = int(sys.argv[3]) if len(sys.argv) > 3 else 30
OUT = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
PORT = int(sys.argv[5]) if len(sys.argv) > 5 else 18888
BASE = f"http://127.0.0.1:{PORT}"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read())


def post_json(path, obj):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


with wave.open(wav_in) as w:
    sr, ch = w.getframerate(), w.getnchannels()
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, ch).mean(1).astype(np.int16)
assert sr == 48000, "input must be 48 kHz"

info = get("/info")
pi = info.get("pipelineInfo", {})
print("embedder :", pi.get("embedder"))
print("model    :", pi.get("inferencer", {}).get("file"), pi.get("inferencer", {}).get("onnxExecutionProvider"))
print("f0       :", info.get("f0Detector"), "| extra", info.get("extraConvertSize"), "| tune", info.get("tran"))

size = CHUNK * 128
times, outs, perfs = [], [], []
pos = 0
for i in range(N + 3):  # first 3 are warm-up
    seg = x[pos : pos + size]
    if len(seg) < size:
        pos = 0
        seg = x[:size]
    pos += size
    t0 = time.perf_counter()
    res = post_json("/test", {"timestamp": 0, "buffer": base64.b64encode(seg.tobytes()).decode()})
    dt = (time.perf_counter() - t0) * 1000
    out = np.frombuffer(base64.b64decode(res["changedVoiceBase64"]), np.int16)
    if i >= 3:
        times.append(dt)
        outs.append(out)
        p = get("/performance")
        perfs.append(p if isinstance(p, list) else p.get("performance", p))

chunk_ms = size / 48
t = np.array(times)
print(f"\nchunk {CHUNK} = {chunk_ms:.0f} ms of audio, {N} chunks")
print(f"conversion time: median {np.median(t):.1f} ms, p90 {np.percentile(t, 90):.1f} ms, max {t.max():.1f} ms")
print(f"real-time headroom: {chunk_ms / np.median(t):.1f}x  (needs > 1x; higher = you can use a smaller chunk)")
try:
    pa = np.array([p for p in perfs if isinstance(p, list) and len(p) >= 4], dtype=float)
    if len(pa):
        print("server /performance median [total, pre, main, post] ms:", np.round(np.median(pa, 0), 1).tolist())
except Exception:
    pass

if OUT:
    y = np.concatenate(outs)
    with wave.open(OUT, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(y.tobytes())
    print("wrote", OUT)
