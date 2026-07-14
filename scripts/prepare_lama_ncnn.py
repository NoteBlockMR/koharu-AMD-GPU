"""Create a fixed low-memory LaMa ONNX graph for pnnx conversion."""

import os
from pathlib import Path

import onnx


size = 256
source = Path(os.environ["KOHARU_LAMA_ONNX"])
target = Path(__file__).resolve().parents[1] / "temp" / "vulkan-pilot" / "lama-256.onnx"
model = onnx.load(source)
for value in model.graph.input:
    shape = value.type.tensor_type.shape.dim
    shape[0].dim_value = 1
    shape[2].dim_value = size
    shape[3].dim_value = size
for value in model.graph.output:
    shape = value.type.tensor_type.shape.dim
    shape[0].dim_value = 1
    shape[1].dim_value = 3
    shape[2].dim_value = size
    shape[3].dim_value = size
onnx.save(model, target)
print(target)
