#!/usr/bin/env python3
"""Train a YOLOv8n model on the Construction Site Safety dataset and
export it to ONNX for edgeguard (the C++ inference/deployment side of
this project) to consume.

Run this on a machine with a working PyTorch install (a GPU speeds up
training a lot but isn't required) — NOT expected to run in this
project's own CI, which deliberately has no PyTorch/GPU at all, since
edgeguard's whole point is proving out training-free, Python-free,
GPU-free deployment. Training and deployment are different jobs with
different infrastructure needs; this script documents the first one
even though it doesn't execute inside this repo's own environment.

Usage:
    pip install ultralytics
    python scripts/train_and_export.py --data data/data.yaml --epochs 100

Produces runs/detect/train/weights/best.pt, then exports it to
models/ppe.onnx in the layout edgeguard::Detector expects (YOLOv8's
native ONNX export format: output shape [1, 4+numClasses, numAnchors],
see include/edgeguard/detector.hpp's decodeYoloOutput doc comment).
"""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/data.yaml",
                         help="path to the dataset's YOLO data.yaml (Roboflow's export includes one)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640,
                         help="must match --size passed to edgeguard/edgeguard_bench at inference time")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="starting checkpoint — nano variant, chosen deliberately: this project's "
                              "whole point is realistic CPU-only edge deployment, and the larger YOLOv8 "
                              "variants (s/m/l/x) trade meaningfully higher accuracy for latency/size this "
                              "project's benchmarks are specifically about measuring the cost of")
    parser.add_argument("--out", default="models/ppe.onnx")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)

    exported_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported_path, out_path)
    print(f"Exported ONNX model to {out_path}")
    print(f"Run inference with: ./build/edgeguard --model={out_path} --image=<your image>")


if __name__ == "__main__":
    main()
