#!/usr/bin/env python3
"""Smoke-test a converted comic text detector with ncnn Vulkan."""

import argparse
import time
from pathlib import Path

import ncnn
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    net = ncnn.Net()
    net.opt.use_vulkan_compute = True
    net.opt.use_fp16_storage = True
    net.opt.use_fp16_packed = True
    net.set_vulkan_device(ncnn.get_default_gpu_index())

    assert net.load_param(str(model_dir / "comictextdetector.ncnn.param")) == 0
    assert net.load_model(str(model_dir / "comictextdetector.ncnn.bin")) == 0

    source = np.zeros((3, 1024, 1024), dtype=np.float32)
    source_mat = ncnn.Mat(source).clone()
    extractor = net.create_extractor()
    assert extractor.input("in0", source_mat) == 0

    started = time.perf_counter()
    for name in ("out0", "out1", "out2"):
        status, output = extractor.extract(name)
        assert status == 0
        print(f"{name}: {np.asarray(output).shape}")
    print(f"forward: {(time.perf_counter() - started) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
