"""Remove ONNX-only font regression ScatterND before converting with pnnx.

The Rust postprocessor already applies sigmoid to the final ten regression
values, so exporting the fully-connected layer output is equivalent.
"""

from pathlib import Path

import onnx
from onnx import helper


root = Path(__file__).resolve().parents[1] / "temp" / "vulkan-pilot"
source = root / "font-detector.onnx"
target = root / "font-detector-raw.onnx"
model = onnx.load(source)
model.graph.ClearField("output")
model.graph.output.append(
    helper.make_tensor_value_info(
        "/model/model/fc/Gemm_output_0",
        onnx.TensorProto.FLOAT,
        ["batch_size", 6162],
    )
)
onnx.save(model, target)
print(target)
