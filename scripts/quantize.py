#!/usr/bin/env python3
"""Post-training dynamic quantization: converts an fp32 ONNX model to
int8, trading a small amount of accuracy for reduced model size and
(usually) faster CPU inference — exactly the trade-off an edge deployment
needs to measure honestly rather than assume.

Unlike scripts/train_and_export.py, this one genuinely runs anywhere with
onnxruntime installed (no PyTorch needed) — including this project's own
environment, so its output can be directly fed into edgeguard_bench for a
real, measured fp32-vs-int8 comparison, not an assumed percentage.

Usage:
    pip install onnxruntime
    python scripts/quantize.py --model models/ppe.onnx --out models/ppe.int8.onnx
"""

import argparse

from onnxruntime.quantization import quantize_dynamic, QuantType


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to the fp32 ONNX model")
    parser.add_argument("--out", required=True, help="path to write the quantized int8 model")
    args = parser.parse_args()

    quantize_dynamic(
        model_input=args.model,
        model_output=args.out,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model written to {args.out}")
    print("Compare with:")
    print(f"  ./build/edgeguard_bench --model={args.model} --image=<sample>")
    print(f"  ./build/edgeguard_bench --model={args.out} --image=<sample>")
    print(f"  ./build/edgeguard_eval  --model={args.out} --data=data/valid   # confirm accuracy didn't collapse")


if __name__ == "__main__":
    main()
