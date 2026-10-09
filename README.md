# edgeguard

[![CI](https://github.com/pandeylakshya207-max/edgeguard/actions/workflows/ci.yml/badge.svg)](https://github.com/pandeylakshya207-max/edgeguard/actions/workflows/ci.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B)](https://en.cppreference.com/w/cpp/17)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The deployment half of an object detector, in C++: it takes a trained
YOLOv8 model for construction-site safety equipment (hardhats, vests,
masks) and runs it on a CPU, with no GPU and no Python at runtime.

The repository contains the inference pipeline, an accuracy evaluator and a
benchmark harness, and it reports what they measure on a real model inside
a container limited to one CPU core and 512 MB of memory.

It does not train models. The model measured below is a public checkpoint
trained by someone else; see [The model](#the-model).

## Results

Measured by the [`measure`](.github/workflows/measure.yml) workflow on
GitHub-hosted runners (4 cores), inside the Docker image this repository
builds. The workflow was run twice; the accuracy figures were identical in
both runs.

### Accuracy

mAP at IoU 0.5 on the Construction Site Safety dataset, 10 classes.

| Split | edgeguard, as deployed | edgeguard, validator protocol | Ultralytics validator |
| --- | ---: | ---: | ---: |
| Validation (114 images, 697 boxes) | 0.853 | 0.865 | 0.863 |
| Test (82 images, 760 boxes) | 0.816 | 0.815 | 0.815 |

- **As deployed** is what the detector does in use: one class per anchor
  (the best one), NMS at 0.45.
- **Validator protocol** scores every class of every anchor and uses NMS at
  0.7, which is how the Ultralytics validator works. `edgeguard_eval
  --multi-label --nms=0.7` selects it.
- **Ultralytics validator** is `yolo val` run on the same ONNX file and the
  same images. It is an independent implementation of the whole pipeline
  (preprocessing, decoding, NMS and the mAP calculation).

Under the same protocol the two implementations agree to within 0.002 mAP
on both splits, and to within about 0.005 on every one of the ten classes:

| Class | edgeguard | Ultralytics |
| --- | ---: | ---: |
| Hardhat | 0.888 | 0.888 |
| Mask | 0.952 | 0.955 |
| NO-Hardhat | 0.793 | 0.789 |
| NO-Mask | 0.768 | 0.768 |
| NO-Safety Vest | 0.833 | 0.828 |
| Person | 0.913 | 0.910 |
| Safety Cone | 0.899 | 0.897 |
| Safety Vest | 0.926 | 0.926 |
| machinery | 0.975 | 0.970 |
| vehicle | 0.703 | 0.702 |

(Validation split, AP at IoU 0.5.)

### Latency and memory

One 640x640 image, 200 timed runs after 10 warm-up runs, each run covering
the full pipeline (letterbox, forward pass, decode, NMS).

| Run | Container limits | Threads | Mean | p95 | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 1 CPU, 512 MB | 1 | 440 ms | 446 ms | 2.3 FPS |
| 2 | 1 CPU, 512 MB | 1 | 325 ms | 340 ms | 3.1 FPS |
| 1 | 4 CPUs, 512 MB | 4 | 179 ms | 188 ms | 5.6 FPS |
| 2 | 4 CPUs, 512 MB | 4 | 129 ms | 132 ms | 7.7 FPS |

The two runs differ by about a quarter. GitHub does not fix the hardware
behind a hosted runner (the first run reported an AMD EPYC 9V74), so treat
these as a range for a cloud CPU core, not as one number.

Resident memory after inference was 360 MiB in both runs, inside the
512 MiB limit. The model file is 44.7 MB.

### What the numbers say

- The C++ pipeline is correct: an independent implementation gives the same
  accuracy on the same model.
- Keeping only the best class per anchor, which is what a deployed detector
  wants, costs about one point of mAP on the validation split compared with
  the validator's protocol, and nothing on the test split.
- On one core this model is not real-time: two to three frames per second. It
  is a YOLOv8s. The nano variant is several times smaller and would be
  faster, but no public nano checkpoint for this dataset was available, so
  that is not measured here.
- Four cores give about 2.5 times the single-core throughput in both runs,
  not 4 times.

### What did not work: int8 quantization

`scripts/quantize.py` applies ONNX Runtime's dynamic quantization, which
shrinks the model from 44.7 MB to 11.5 MB. OpenCV 4.10 cannot load the
result: its importer has no `DynamicQuantizeLinear` layer. There are
therefore no int8 latency or accuracy figures. Static quantization, which
produces a different set of operators, is the next thing to try.

## The model

The measured checkpoint is the YOLOv8s published in
[VoxDroid/Construction-Site-Safety-PPE-Detection](https://github.com/VoxDroid/Construction-Site-Safety-PPE-Detection),
trained for 200 epochs on Roboflow's
[Construction Site Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
dataset. That repository also carries the validation and test splits used
above.

- The `measure` workflow downloads the checkpoint and the two splits at a
  pinned commit each time it runs.
- The checkpoint is **not** stored in this repository. Models trained with
  Ultralytics are AGPL-3.0 and this repository is MIT.
- Its author reports 0.877 mAP@0.5, taken from the last training epoch.
  The exported ONNX file scores 0.863 with the Ultralytics validator, which
  is the figure edgeguard is compared against.

## What testing on a real model found

Until the measurements above, this project had only been run on synthetic
tensors and a small hand-built mock model. Running a real one found two
problems:

- **OpenCV 4.6 cannot load a YOLOv8 model.** The CI and the Docker image
  used Ubuntu 24.04, which packages OpenCV 4.6. Its ONNX importer fails on
  a Reshape node in the detection head, so no real model would have loaded.
  The image and CI now use Debian 13 (OpenCV 4.10), and the build refuses
  older versions.
- **The benchmark was not single-threaded.** It printed "single-threaded"
  but never limited OpenCV's thread pool, so it used every core. It now
  takes `--threads` (default 1) and prints the count it used.

## How it works

```
image -> letterbox to 640x640 -> cv::dnn forward pass -> decode -> undo letterbox -> NMS -> boxes
```

```
include/edgeguard/   public headers
src/
  geometry.*           bounding boxes and IoU
  nms.*                class-aware non-maximum suppression
  letterbox.*          aspect-preserving resize and pad, and its inverse
  labels.*             YOLO-format annotation parsing, the class list
  evaluator.*          average precision per class and mAP
  detector.*           YOLOv8 output decoding and the cv::dnn wrapper
  main.cpp             edgeguard        detect objects in one image
  bench.cpp            edgeguard_bench  latency, throughput and memory
  eval.cpp             edgeguard_eval   mAP over a labelled split
tests/               Catch2 tests, one file per module
docker/Dockerfile    two-stage build: toolchain and tests, then a runtime image
scripts/             measurement, quantization, training and dataset notes
```

Two decisions worth knowing about:

- **NMS is class-aware.** A hardhat sits on a person's head, so a Hardhat
  box and a Person box overlap heavily and both are correct. Suppression
  only happens between boxes of the same class.
- **Decoding is separate from the network.** `decodeYoloOutput` takes a
  plain matrix, so it is tested against hand-built tensors and never needs
  a model file. The same holds for letterboxing, NMS, label parsing and the
  mAP calculation.

## Building

The Docker image is the supported build. It compiles everything and runs
the test suite as part of the build, and it is what CI uses:

```bash
docker build -f docker/Dockerfile -t edgeguard .
```

To build directly you need CMake, Catch2 3 and **OpenCV 4.10 or newer**.
Debian 13 packages all three; Ubuntu 24.04's OpenCV 4.6 is too old (see
above) and the build refuses it.

```bash
sudo apt-get install cmake build-essential libopencv-dev catch2
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/edgeguard_tests
```

## Usage

With a model at `work/ppe.onnx`:

```bash
# detect objects in one image, under the deployment limits
docker run --rm --cpus=1 --memory=512m -v "$PWD/work":/work -w /work \
  edgeguard --model=ppe.onnx --image=photo.jpg --out=result.jpg

# latency, throughput and memory
docker run --rm --cpus=1 --memory=512m -v "$PWD/work":/work -w /work \
  --entrypoint /app/edgeguard_bench edgeguard \
  --model=ppe.onnx --image=photo.jpg --iters=200 --threads=1

# per-class AP and mAP over a split with images/ and labels/ (YOLO format)
docker run --rm -v "$PWD/work":/work -w /work \
  --entrypoint /app/edgeguard_eval edgeguard --model=ppe.onnx --data=valid
```

From a direct build the same tools are `./build/edgeguard`,
`./build/edgeguard_bench` and `./build/edgeguard_eval`.

## Reproducing the measurements

```bash
gh workflow run measure.yml        # or start it from the Actions tab
```

The workflow builds the image, downloads the checkpoint and the two splits,
runs [`scripts/measure.sh`](scripts/measure.sh), then runs the Ultralytics
validator and the quantization experiment. The results are written to the
run summary and uploaded as an artifact. It takes about ten minutes.

## Testing

```bash
./build/edgeguard_tests          # or: ctest --test-dir build
```

43 test cases and 117 assertions, with no compiler warnings under
`-Wall -Wextra`. CI runs them inside the Docker image on every push.

- **Geometry, NMS, letterboxing, label parsing and decoding** are tested
  against hand-built inputs with known answers.
- **The mAP evaluator** includes a case worked out by hand: a false
  positive between two true positives, with an expected AP of exactly 5/6.
- **The full pipeline** runs end to end on a small mock ONNX model with the
  YOLOv8 output shape. That test checks the wiring, not accuracy; the mock
  has random weights.

The unit tests did not catch the OpenCV 4.6 problem, because the mock model
has none of the layers a real detection head has. The `measure` workflow is
the test that covers a real model.

## Limits

- The measured model is a third-party YOLOv8s, not a nano model and not one
  trained here.
- The splits are small (114 and 82 images), so per-class AP is noisy. Safety
  Cone scores 0.50 on the test split, where it appears in 8 images.
- Latency was measured on shared cloud VMs, and two runs differed by about
  25%. Expect it to vary between runs and machines.
- There is no int8 result (see above).
- The tools process single images. There is no video or camera pipeline.
- `scripts/train_and_export.py` documents how to train and export a model
  with Ultralytics; it has not been run as part of this project.

## License

MIT for the code in this repository; see [LICENSE](LICENSE). The model and
dataset used for the measurements belong to their authors and are not
redistributed here.
