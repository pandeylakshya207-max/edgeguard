# Two stages: the builder has the full toolchain and runs the test suite as
# part of the build; the runtime stage carries only the OpenCV shared
# libraries and the three binaries.
#
# Debian 13 (trixie) is used because it packages OpenCV 4.10. The OpenCV 4.6
# in Ubuntu 24.04 cannot import a YOLOv8 ONNX export.
#
#   docker build -f docker/Dockerfile -t edgeguard .
#   docker run --rm --cpus=1 --memory=512m -v "$PWD/work":/work -w /work \
#       edgeguard --model=ppe.onnx --image=photo.jpg --out=result.jpg
#
# The other tools are selected with --entrypoint:
#   docker run --rm --entrypoint /app/edgeguard_bench edgeguard --help
#   docker run --rm --entrypoint /app/edgeguard_eval  edgeguard --help

FROM debian:trixie-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential libopencv-dev catch2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt .
COPY include include
COPY src src
COPY tests tests
COPY models models
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" \
    && ./build/edgeguard_tests

FROM debian:trixie-slim AS runtime
# The runtime library packages carry the OpenCV version in their names
# (libopencv-core410 and so on), so they are looked up rather than spelled out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       $(apt-cache pkgnames | grep -E '^libopencv-(core|imgproc|imgcodecs|dnn|highgui)4[0-9]+(t64)?$') \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /src/build/edgeguard /src/build/edgeguard_bench /src/build/edgeguard_eval /app/
COPY models /app/models
# Fails the build here, not at first use, if a shared library is missing.
RUN /app/edgeguard --help > /dev/null && /app/edgeguard_bench --help > /dev/null && /app/edgeguard_eval --help > /dev/null
ENTRYPOINT ["/app/edgeguard"]
