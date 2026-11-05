# Post-import patch for voice_changer.RVC.RVCr2 (applied after the bundled
# module loads; see pathlib_hook.py).
#
# Gate hold: the stock silence gate looks at the RMS of each whole chunk. When
# a chunk holds the last word of a sentence followed by silence, its average
# drops under the threshold and the *entire* chunk is muted - the end of the
# sentence gets cut off. Here, once the input has been above the threshold, the
# gate stays open for HOLD_SEC afterwards, so word endings always come through
# while real silence is still gated.
import math

import numpy as np

HOLD_SEC = 0.8


def apply(mod):
    cls = getattr(mod, "RVCr2", None)
    if cls is None or getattr(cls.inference, "_vcp_gate_hold", False):
        return
    orig = cls.inference

    def inference(self, receivedData, *args, **kwargs):
        try:
            th = float(self.settings.silentThreshold)
            x = np.asarray(receivedData, dtype=np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
            sr = getattr(self, "inputSampleRate", 48000) or 48000
            hold_chunks = max(1, math.ceil(HOLD_SEC * sr / max(1, x.size)))
        except Exception:
            return orig(self, receivedData, *args, **kwargs)

        if th <= 0 or rms >= th:
            self._vcp_hold = hold_chunks
            return orig(self, receivedData, *args, **kwargs)

        hold = getattr(self, "_vcp_hold", 0)
        if hold > 0:
            # Below threshold but still inside the hold window: bypass the gate.
            self._vcp_hold = hold - 1
            self.settings.silentThreshold = 0.0
            try:
                return orig(self, receivedData, *args, **kwargs)
            finally:
                self.settings.silentThreshold = th
        return orig(self, receivedData, *args, **kwargs)

    inference._vcp_gate_hold = True
    cls.inference = inference
    globals().get("log", print)(f"RVCr2 gate hold enabled ({HOLD_SEC}s)")
