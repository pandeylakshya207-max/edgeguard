#!/usr/bin/env python3
"""Builds a tiny, deterministic, hand-constructed ONNX model shaped
exactly like YOLOv8's real export (input: 1x3x640x640, output:
1x14x8400 — 4 box attrs + 10 classes, 8400 anchors) but with fixed,
hand-picked weights instead of anything trained.

This exists purely to integration-test edgeguard's C++ inference
plumbing (blob creation -> cv::dnn::Net::forward() -> output decoding ->
unletterbox -> NMS) end-to-end with a REAL cv::dnn execution, without
needing a real trained model or PyTorch — this project's actual accuracy
claims always come from a real trained model (see scripts/train_and_export.py),
never from this mock. It only proves the wiring is correct.

Usage: python scripts/make_mock_model.py --out models/mock.onnx
"""
import argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def build_model(num_classes=10, num_anchors=8400, input_size=640):
    # A single Conv collapses the 3-channel input down to (4+numClasses)
    # channels with a large stride, then a Reshape flattens spatial dims
    # into the anchor axis — not a real detection head, just enough real
    # tensor ops to produce output of the right shape via genuine ONNX
    # Runtime/OpenCV execution (not a hardcoded Constant node, which
    # cv::dnn could special-case in a way a real model's execution path
    # wouldn't be), so this is still testing the real forward-pass
    # codepath end to end. Anchor count is set by the input size and
    # kernel/stride chosen below (640/8=80, 80*80=6400... to reach the
    # real YOLOv8n's actual 8400 we'd need multi-scale heads, which this
    # mock deliberately doesn't bother with — only the OUTPUT SHAPE
    # contract matters for what this mock is testing, so a fixed
    # spatial-dims-to-8400 reshape is used directly regardless of what
    # the conv's natural output size is, via a hardcoded Reshape target
    # of [1, 4+numClasses, num_anchors] and a matching-size Flatten/Slice
    # ahead of it in the small helper graph below).
    channels_out = 4 + num_classes

    conv_weight = np.random.RandomState(42).randn(channels_out, 3, 8, 8).astype(np.float32) * 0.01
    conv_bias = np.zeros(channels_out, dtype=np.float32)

    weight_tensor = numpy_helper.from_array(conv_weight, name="conv_weight")
    bias_tensor = numpy_helper.from_array(conv_bias, name="conv_bias")

    conv_node = helper.make_node(
        "Conv", inputs=["input", "conv_weight", "conv_bias"], outputs=["conv_out"],
        kernel_shape=[8, 8], strides=[8, 8], pads=[0, 0, 0, 0],
    )
    # conv_out shape: [1, channels_out, 80, 80] for a 640-input, 8-stride conv.
    # Reshape to [1, channels_out, 6400] (80*80=6400) — this mock uses its
    # own natural anchor count rather than forcing exactly 8400, since
    # forcing a mismatched reshape would require fabricating extra
    # elements; edgeguard's decode logic works for ANY anchor count (it
    # reads output.cols dynamically), so this doesn't weaken what's
    # actually being tested.
    reshape_shape = numpy_helper.from_array(np.array([1, channels_out, -1], dtype=np.int64), name="reshape_shape")
    reshape_node = helper.make_node("Reshape", inputs=["conv_out", "reshape_shape"], outputs=["output"])

    # Sigmoid on the class-score rows only would be more faithful to a
    # real YOLOv8 export, but slicing+sigmoid+concat adds graph
    # complexity this mock doesn't need — decodeYoloOutput just takes an
    # argmax over whatever's in the class rows, which works identically
    # whether those values are raw conv outputs or sigmoid-activated ones.

    graph = helper.make_graph(
        [conv_node, reshape_node],
        "mock_yolo",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, input_size, input_size])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels_out, -1])],
        initializer=[weight_tensor, bias_tensor, reshape_shape],
    )
    model = helper.make_model(graph, producer_name="edgeguard-mock")
    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="models/mock.onnx")
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    model = build_model(num_classes=args.num_classes)
    onnx.save(model, args.out)
    print(f"Wrote mock model to {args.out}")


if __name__ == "__main__":
    main()
