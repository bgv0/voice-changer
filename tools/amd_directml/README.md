# AMD / DirectML speed-ups for VCClient v1.5.3

Tested on an RX 9070 XT + Ryzen 7 7700, `MMVCServerSIO_win_onnxdirectML-cuda_v.1.5.3.18a`.

## The problem

On the DirectML build, PyTorch has no GPU, so anything the app runs through
PyTorch's device (`DeviceManager.getDevice()` -> `cpu`) runs on the CPU. The
content-vec / HuBERT embedder is one of those: it runs on the CPU for every
chunk, even though `pretrain/content_vec_500.onnx` is right there.

## The fix

`vc_patches/OnnxContentvec.py` runs `content_vec_500.onnx` on the selected
DirectML GPU (outputs `unit12` for v2 models, `units9` for v1). Anything it
can't handle falls back to the stock path, so it can't make things worse.

The release build is a PyInstaller exe, so the module can't just be edited.
`install_patch.py` adds a tiny import hook (`vc_patches/pathlib_hook.py`) to
`base_library.zip` that loads listed modules from the `vc_patches` folder next
to the exe. The original `base_library.zip` is backed up.

```
python install_patch.py "<...>\MMVCServerSIO"              # install
python install_patch.py "<...>\MMVCServerSIO" --uninstall  # restore
```

Check it worked: `http://127.0.0.1:18888/info` -> `pipelineInfo.embedder.devType`
should be `dml:0` instead of `cpu`.

## Results (chunk 256, extra 16384, Egirl v2 32k model, rmvpe_onnx)

| | per-chunk conversion | real-time headroom |
|---|---|---|
| stock | 206 ms | 3.3x |
| patched | 84-92 ms | 7.4x |

Output quality: stock-vs-patched difference is the same as patched-vs-patched
(RVC adds random noise every run), i.e. no change.

With the patch, chunk 128 (341 ms) runs at ~4x headroom on a 48k model with an
index. Chunk 96 was much slower than expected (likely DirectML re-compiling for
changing input sizes), so 128 is the practical floor here.

## Other tools

* `add_model.sh` - add a `.pth` (+ `.index`) to a running server as an ONNX slot
  (export + reload), since `.pth` models also run on the CPU under DirectML.
* `server_bench.py` - time a running server through its `/test` endpoint.
* `bench_dml.py` - time each pipeline stage on DirectML, compare two models.
* `convert_rvc_onnx_fp16.py` - fp16 conversion of an RVC ONNX model.
  **Not recommended:** fully fp16 was ~30% faster but measurably noisier; keeping
  the decoder fp32 fixes quality but leaves only ~1.5 ms gain.
