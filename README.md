# edgeguard

[![CI](https://github.com/pandeylakshya207-max/edgeguard/actions/workflows/ci.yml/badge.svg)](https://github.com/pandeylakshya207-max/edgeguard/actions/workflows/ci.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B)](https://en.cppreference.com/w/cpp/17)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A CPU-only, edge-deployable PPE (personal protective equipment) detector
in C++ — real-time object detection for construction-site safety
compliance (hardhat, safety vest, mask), built around a genuine
resource-constrained deployment story rather than a Jupyter-notebook
accuracy number.

## Why C++, and why this scope

Most object-detection projects stop at "trained a model, here's the
accuracy" — that's a Python/PyTorch story, and it's not a very
differentiated one. This project is deliberately about the other half:
what it actually takes to run a real detector under real constraints —
no GPU, a single CPU core, a fixed memory budget — the way it would
actually be deployed on a factory floor or a job site, not a dev
workstation.

That's why inference is written in C++ against OpenCV's `dnn` module
directly (not a Python wrapper around a Python framework), why the
benchmark harness measures real wall-clock latency and real
`/proc`-reported memory rather than assumed numbers, and why the model
itself gets quantized and re-measured rather than shipped as-is.

Training (which needs a real GPU-capable Python/PyTorch environment) and
inference/deployment (this repo, C++, zero Python dependencies at
runtime) are treated as two separate jobs with two separate toolchains —
because in a real production system, they usually are.

## Architecture

```
include/edgeguard/   public headers — the interface every module is tested against
src/                 implementations, plus 4 binaries:
  geometry.*           BoundingBox, IoU
  nms.*                class-aware non-max suppression
  letterbox.*          aspect-ratio-preserving resize/pad + inverse transform
  labels.*             YOLO-format annotation parsing
  evaluator.*          mean Average Precision (PASCAL VOC/COCO-style)
  detector.*           YOLOv8 ONNX output decoding + cv::dnn::Net wrapper
  main.cpp             `edgeguard` — run detection on one image
  bench.cpp            `edgeguard_bench` — real latency/throughput/memory measurement
  eval.cpp             `edgeguard_eval` — mAP over a full validation split
tests/                Catch2 unit tests — one file per module, 37 test cases
docker/               multi-stage Dockerfile (build toolchain vs. minimal runtime image)
scripts/              dataset docs, training/export (Python+PyTorch), quantization (Python+onnxruntime)
```

Every module above `detector.cpp` is pure, dependency-free logic
(geometry, NMS, letterboxing, label parsing, mAP) and is fully unit
tested with hand-built synthetic inputs — none of it needs a real model
file to verify. `detector.cpp` is the one integration point that
actually needs a real `.onnx` model to exercise end-to-end; everything
it calls internally is already independently tested.

## Building

```bash
sudo apt-get install cmake libopencv-dev catch2
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/edgeguard_tests
```

## Usage

```bash
# Run detection on one image
./build/edgeguard --model=models/ppe.onnx --image=photo.jpg --out=result.jpg

# Measure real latency/throughput/memory on this machine
./build/edgeguard_bench --model=models/ppe.onnx --image=sample.jpg --iters=100

# Full validation-set accuracy (per-class AP + overall mAP)
./build/edgeguard_eval --model=models/ppe.onnx --data=data/valid
```

See `scripts/fetch_dataset.md` for how to get the dataset and train a
model — that part runs in Python with a real PyTorch install, separately
from this repo's own C++ build/CI (see "Why C++" above for why that split
is deliberate, not a limitation).

## Testing

```bash
./build/edgeguard_tests          # or: ctest --test-dir build
```

37 test cases, 97 assertions, all passing, zero compiler warnings under
`-Wall -Wextra`. A few worth calling out specifically:

- **Class-aware NMS is a dedicated, deliberately-tested design decision**,
  not an accident: a Hardhat box and a Person box legitimately occupy the
  same pixels (the hardhat sits on the person's head), so suppression
  must never happen across classes — a generic single-class NMS
  implementation would silently delete correct detections in exactly
  this dataset's most common scenario.
- **The mAP evaluator includes a hand-computed numeric test case**
  (a false positive between two true positives, worked out by hand to an
  exact expected AP of 5/6) rather than only qualitative "is it roughly
  right" assertions — the same standard applied throughout this project:
  a metric implementation is only trustworthy once it's been checked
  against an independently-verifiable number, not just its own output.
- **YOLO output decoding is tested against hand-built synthetic tensors**
  matching YOLOv8's exact ONNX output convention (`[1, 4+numClasses,
  numAnchors]`, no separate objectness score), so the decoding logic is
  fully verified before it's ever pointed at a real model file.

## Edge deployment

The Dockerfile is a multi-stage build: the builder stage has the full
toolchain (cmake, compiler, OpenCV dev headers), the runtime stage only
ships OpenCV's shared libraries and the compiled binary — real deployment
image, not a dev container.

```bash
docker build -f docker/Dockerfile -t edgeguard .
docker run --cpus=1 --memory=512m -v $(pwd)/models:/app/models \
  edgeguard --model=/app/models/ppe.onnx --image=/app/models/sample.jpg --out=/app/result.jpg
```

`--cpus=1 --memory=512m` is the actual resource-constrained target this
project's benchmarks are measured against — not a hypothetical, a real
`docker run` flag that genuinely caps what the container can use.

## Status

The C++ inference/deployment stack — geometry, NMS, letterboxing, label
parsing, mAP evaluation, YOLO output decoding, the CLI/bench/eval
binaries, CI, and the Dockerfile — is complete and fully tested against
synthetic data.

**Real accuracy and latency numbers against the actual PPE dataset are
the next step**, once a trained model is available to point
`edgeguard_bench` and `edgeguard_eval` at. This section will be updated
with real, measured fp32-vs-int8 comparisons (latency, throughput,
memory, mAP) the moment that happens — not before, and not with assumed
placeholder numbers in the meantime.

## License

MIT — see [LICENSE](LICENSE).
