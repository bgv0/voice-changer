"""Check whether the silence gate cuts off phrase endings.

Builds a test signal of short phrases separated by silence (each phrase end is
a "sentence end"), streams it through a running server's /test endpoint in
real chunk sizes, and counts chunks that contained speech but came back silent.

Usage:
    python gate_test.py input.wav [threshold=0.003] [chunk_x128=192] [port=18888]
"""
import base64
import json
import sys
import urllib.request
import wave

import numpy as np

wav_in = sys.argv[1]
TH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.003
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 192
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 18888
B = f"http://127.0.0.1:{PORT}"


def post_form(path, **fields):
    boundary = "vcpboundary"
    body = b"".join(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()
    ) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(B + path, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    urllib.request.urlopen(req, timeout=30).read()


def convert(seg):
    req = urllib.request.Request(
        B + "/test",
        data=json.dumps({"timestamp": 0, "buffer": base64.b64encode(seg.tobytes()).decode()}).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return np.frombuffer(base64.b64decode(r["changedVoiceBase64"]), np.int16)


with wave.open(wav_in) as w:
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels()).mean(1)
sr = 48000
# phrases of 0.9-1.7 s with 1.2 s silence between them
rng = np.random.default_rng(0)
parts, pos = [], 0
while pos < len(x) - sr:
    n = int(sr * rng.uniform(0.9, 1.7))
    parts += [x[pos : pos + n], np.zeros(int(sr * 1.2))]
    pos += n
sig = np.concatenate(parts * 2).astype(np.int16)

post_form("/update_settings", key="silentThreshold", val=TH)
size = CHUNK * 128
outs = []
for i in range(0, len(sig) - size, size):
    outs.append((sig[i : i + size], convert(sig[i : i + size])))
outs = outs[3:]  # warm-up

if "-v" in sys.argv:
    for k, (inp, out) in enumerate(outs):
        ri = np.sqrt(np.mean((inp.astype(np.float32) / 32768) ** 2))
        ro = np.sqrt(np.mean((out.astype(np.float32) / 32768) ** 2)) if out.size else 0
        print(f"{k:3d} in_rms {ri:.4f}  out_rms {ro:.4f}  {'MUTED' if out.size and np.abs(out).max() == 0 else ''}")
sound_chunks = muted = 0
for inp, out in outs:
    rms = np.sqrt(np.mean((inp.astype(np.float32) / 32768) ** 2))
    if rms > 0.0005:  # chunk has some sound in it
        sound_chunks += 1
        if out.size and np.abs(out).max() == 0:
            muted += 1
print(f"threshold {TH}, chunk {CHUNK}: {muted} of {sound_chunks} chunks with sound were muted")
