"""Convert an fp32 RVC .onnx model (as exported by VCClient on AMD/DirectML) to fp16.

VCClient only exports fp16 models when an NVIDIA GPU is present, so AMD users
always get the slower fp32 export. This converts it afterwards.

* Inputs/outputs stay fp32, so VCClient loads it with no code changes.
* The NSF sine generator (/dec/m_source) and the random-noise ops stay fp32:
  in fp16 the running phase sum loses precision and the pitch drifts/buzzes.

Usage:
    python convert_rvc_onnx_fp16.py model_simple.onnx [out.onnx] [extra/prefix/to/keep/fp32 ...]
Needs: pip install onnx onnxconverter-common
"""
import json
import sys

import onnx
from onnxconverter_common import float16

KEEP_FP32_PREFIXES = ("/dec/m_source/",)
KEEP_FP32_OPS = ["RandomNormalLike", "RandomUniformLike", "CumSum", "Sin", "Cos", "Range"]


def _fix_casts(model):
    # onnxconverter-common bug: it retypes a Cast's output to fp16 in value_info
    # but leaves the Cast's own "to=float". Make the Cast match its output.
    fp16_values = {
        v.name for v in model.graph.value_info if v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    }
    fixed = 0
    for n in model.graph.node:
        if n.op_type == "Cast" and n.output[0] in fp16_values:
            for a in n.attribute:
                if a.name == "to" and a.i == onnx.TensorProto.FLOAT:
                    a.i = onnx.TensorProto.FLOAT16
                    fixed += 1
    if fixed:
        print(f"fixed {fixed} Cast node(s)")


def _fix_mixed_constants(model):
    # Another converter gap: some nodes (e.g. the attention-mask Equal) end up
    # comparing an fp16 value with an fp32 constant. Give those nodes an fp16
    # copy of the constant.
    import numpy as np
    from onnx import numpy_helper

    g = model.graph
    fp16_values = {v.name for v in g.value_info if v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16}
    fp16_values |= {i.name for i in g.initializer if i.data_type == onnx.TensorProto.FLOAT16}
    inits = {i.name: i for i in g.initializer}
    made = {}
    for n in g.node:
        if n.op_type == "Cast" or not any(i in fp16_values for i in n.input):
            continue
        for k, name in enumerate(n.input):
            init = inits.get(name)
            if init is None or init.data_type != onnx.TensorProto.FLOAT:
                continue
            if name not in made:
                arr = numpy_helper.to_array(init).astype(np.float16)
                t = numpy_helper.from_array(arr, name + "_fp16")
                g.initializer.append(t)
                made[name] = t.name
            n.input[k] = made[name]
    if made:
        print(f"made fp16 copies of {len(made)} constant(s)")


def convert(src: str, dst: str, extra_keep=()):
    model = onnx.load(src)
    keep = KEEP_FP32_PREFIXES + tuple(extra_keep)
    block_nodes = [n.name for n in model.graph.node if n.name.startswith(keep)]
    op_block = list(float16.DEFAULT_OP_BLOCK_LIST) + KEEP_FP32_OPS

    model16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        node_block_list=block_nodes,
        op_block_list=op_block,
    )

    _fix_casts(model16)
    _fix_mixed_constants(model16)

    # Keep VCClient's metadata, and mark the file so it's recognisable.
    for p in model16.metadata_props:
        if p.key == "metadata":
            try:
                meta = json.loads(p.value)
                meta["fp16_converted"] = True
                p.value = json.dumps(meta)
            except ValueError:
                pass

    onnx.save(model16, dst)
    print(f"saved {dst}  (kept {len(block_nodes)} nodes in fp32)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".onnx", "_fp16.onnx")
    convert(src, dst, sys.argv[3:])
