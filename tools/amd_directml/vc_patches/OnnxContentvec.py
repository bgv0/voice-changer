# Patched replacement for voice_changer.RVC.embedder.OnnxContentvec.
#
# The stock 1.5.3 build ships content_vec_500.onnx but this class just raises
# "Not implemented", so the app falls back to the PyTorch HuBERT model. On AMD
# (DirectML) PyTorch has no GPU, so that model runs on the CPU every chunk.
# This version runs the ONNX model on the DirectML GPU instead.
#
# Any problem while loading -> raise, and the app falls back to its original
# PyTorch path, so this patch can never make things worse than stock.
import sys

import numpy as np
import onnxruntime
import torch
from torch import device

from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.embedder.Embedder import Embedder

# (embOutputLayer, useFinalProj) -> ONNX output name
_OUTPUTS = {
    (12, False): "unit12",  # RVC v2 models
    (9, True): "units9",  # RVC v1 models
}


def _find_gpu_index() -> int:
    # EmbedderManager doesn't pass the GPU number down, so read it from the
    # createPipeline() call that is loading us.
    f = sys._getframe(1)
    while f is not None:
        if f.f_code.co_name == "createPipeline" and "gpu" in f.f_locals:
            return int(f.f_locals["gpu"])
        f = f.f_back
    return -1


class OnnxContentvec(Embedder):
    def loadModel(self, file: str, dev: device) -> Embedder:
        gpu = _find_gpu_index()
        if gpu < 0 or "DmlExecutionProvider" not in onnxruntime.get_available_providers():
            raise Exception("[vc_patches] DirectML GPU not selected, using stock embedder")

        providers, provider_options = DeviceManager.get_instance().getOnnxExecutionProvider(gpu)
        if providers[0] != "DmlExecutionProvider":
            raise Exception("[vc_patches] not a DirectML device, using stock embedder")

        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        # Settings DirectML requires.
        so.enable_mem_pattern = False
        so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.onnx_session = onnxruntime.InferenceSession(
            file, sess_options=so, providers=providers, provider_options=provider_options
        )
        self.gpu = gpu
        self._fallback = None
        super().setProps("hubert_base", file, dev, False)
        print(f"[vc_patches] content vec embedder running on DirectML gpu{gpu}")
        return self

    def getEmbedderInfo(self):
        info = super().getEmbedderInfo()
        info["devType"] = f"dml:{self.gpu}"
        info["onnxExecutionProvider"] = self.onnx_session.get_providers()
        return info

    def _torch_fallback(self):
        # Only used for unusual model types this ONNX file has no output for.
        if self._fallback is None:
            from voice_changer.RVC.embedder.EmbedderManager import EmbedderManager
            from voice_changer.RVC.embedder.FairseqHubert import FairseqHubert

            print("[vc_patches] model needs a layer the ONNX embedder lacks, using torch embedder")
            self._fallback = FairseqHubert().loadModel(EmbedderManager.params.hubert_base, self.dev, False)
        return self._fallback

    def extractFeatures(self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True) -> torch.Tensor:
        out_name = _OUTPUTS.get((embOutputLayer, useFinalProj))
        if out_name is None:
            return self._torch_fallback().extractFeatures(feats, embOutputLayer, useFinalProj)

        audio = feats.detach().cpu().numpy().astype(np.float32)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        res = self.onnx_session.run([out_name], {"audio": audio})[0]
        return torch.from_numpy(res).to(self.dev)
