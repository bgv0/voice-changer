# Post-import patch for voice_changer.RVC.RVCr2 (applied after the bundled
# module loads; see pathlib_hook.py).
#
# Better silence gate. The stock gate decides from the RMS of a slightly older
# slice of audio (it skips the newest crossfade part) and mutes whole chunks:
#   * start of speech: the chunk that holds the first sound of a word ("h" of
#     "hello") is judged on the silence before it and muted, so the word
#     onset is lost and comes out slurred.
#   * end of speech: a chunk with the last word plus silence averages under the
#     threshold and is muted, cutting off sentence endings.
# Here the gate is decided on the *new* audio instead: it opens as soon as the
# new chunk has sound in it, and stays open for HOLD_SEC after the last chunk
# that did. Real silence is still gated.
import math

import numpy as np

HOLD_SEC = 0.8


def apply(mod):
    cls = getattr(mod, "RVCr2", None)
    if cls is None or getattr(cls.inference, "_vcp_gate_hold", False):
        return
    orig = cls.inference

    def run_ungated(self, th, receivedData, args, kwargs):
        self.settings.silentThreshold = 0.0
        try:
            return orig(self, receivedData, *args, **kwargs)
        finally:
            self.settings.silentThreshold = th

    def inference(self, receivedData, *args, **kwargs):
        try:
            th = float(self.settings.silentThreshold)
            x = np.asarray(receivedData, dtype=np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
            sr = getattr(self, "inputSampleRate", 48000) or 48000
            hold_chunks = max(1, math.ceil(HOLD_SEC * sr / max(1, x.size)))
        except Exception:
            return orig(self, receivedData, *args, **kwargs)

        if th <= 0:
            return orig(self, receivedData, *args, **kwargs)

        if rms >= th:
            # Sound in the new audio: open now (keeps word onsets).
            self._vcp_hold = hold_chunks
            return run_ungated(self, th, receivedData, args, kwargs)

        hold = getattr(self, "_vcp_hold", 0)
        if hold > 0:
            # Quiet, but speech was recent: stay open (keeps word endings).
            self._vcp_hold = hold - 1
            return run_ungated(self, th, receivedData, args, kwargs)

        return orig(self, receivedData, *args, **kwargs)

    inference._vcp_gate_hold = True
    cls.inference = inference
    globals().get("log", print)(f"RVCr2 gate: opens on new audio, holds {HOLD_SEC}s")
