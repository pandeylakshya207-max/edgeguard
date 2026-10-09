#!/usr/bin/env python3
"""Post-training dynamic quantization: converts an fp32 ONNX model to
int8, trading a small amount of accuracy for reduced model size and
(usually) faster CPU inference — exactly the trade-off an edge deployment
needs to measure honestly rather than assume.

It needs only onnxruntime (no PyTorch), and the `measure` workflow runs it.

Known limitation: OpenCV 4.10 cannot load the model this produces. Dynamic
quantization inserts DynamicQuantizeLinear nodes, which OpenCV's ONNX
importer does not implement, so edgeguard cannot run the output yet. On the
YOLOv8s measured in the README the file shrinks from 44.7 MB to 11.5 MB,
and that is all that can be said about it for now.

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
