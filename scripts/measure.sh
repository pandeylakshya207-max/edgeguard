#!/usr/bin/env bash
# Measures a real model with the edgeguard tools and prints the results:
# accuracy on the validation and test splits, then latency, throughput and
# memory. Every figure comes from the deployment image, run under the
# limits the README states (1 CPU and 512 MB for the single-core benchmark).
#
# Usage: scripts/measure.sh [work-dir]
#
# The work directory must contain:
#   ppe.onnx                      the model
#   valid/images, valid/labels    a split in YOLO format
#   test/images,  test/labels     a second split
#   sample.jpg                    the image used for the latency benchmark
#
# .github/workflows/measure.yml prepares that directory from a public
# checkpoint and dataset and then calls this script. To run the tools from
# a local build instead of the image, set EDGEGUARD_NATIVE to the build
# directory (the CPU and memory limits are then not applied).
set -euo pipefail

WORK=${1:-work}
ITERS=${ITERS:-200}
IMAGE=${EDGEGUARD_IMAGE:-edgeguard}
MEMORY=512m

for needed in ppe.onnx sample.jpg valid/images valid/labels test/images test/labels; do
    if [ ! -e "$WORK/$needed" ]; then
        echo "measure.sh: $WORK/$needed is missing" >&2
        exit 1
    fi
done

# tool <cpus> <binary> <args...>: runs one edgeguard binary inside the work directory.
tool() {
    local cpus=$1 binary=$2
    shift 2
    if [ -n "${EDGEGUARD_NATIVE:-}" ]; then
        (cd "$WORK" && "$EDGEGUARD_NATIVE/$binary" "$@")
    else
        docker run --rm --cpus="$cpus" --memory="$MEMORY" \
            -v "$(cd "$WORK" && pwd)":/work -w /work \
            --entrypoint "/app/$binary" "$IMAGE" "$@"
    fi
}

cores=$(nproc)

echo "== machine =="
echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')"
echo "cores: $cores"
echo "model file: $(du -h "$WORK/ppe.onnx" | cut -f1)"
echo "validation images: $(find "$WORK/valid/images" -type f | wc -l), test images: $(find "$WORK/test/images" -type f | wc -l)"

echo
echo "== accuracy: validation split, best class per anchor (the deployed behaviour) =="
tool "$cores" edgeguard_eval --model=ppe.onnx --data=valid

echo
echo "== accuracy: test split, best class per anchor =="
tool "$cores" edgeguard_eval --model=ppe.onnx --data=test

echo
echo "== accuracy: validation split, Ultralytics validation protocol (every class per anchor, NMS 0.7) =="
tool "$cores" edgeguard_eval --model=ppe.onnx --data=valid --multi-label --nms=0.7

echo
echo "== accuracy: test split, Ultralytics validation protocol =="
tool "$cores" edgeguard_eval --model=ppe.onnx --data=test --multi-label --nms=0.7

echo
echo "== latency: 1 thread, container limited to 1 CPU and $MEMORY =="
tool 1 edgeguard_bench --model=ppe.onnx --image=sample.jpg --iters="$ITERS" --threads=1

echo
echo "== latency: $cores threads, container limited to $cores CPUs and $MEMORY =="
tool "$cores" edgeguard_bench --model=ppe.onnx --image=sample.jpg --iters="$ITERS" --threads="$cores"
