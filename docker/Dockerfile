# Multi-stage build: the builder stage has the full toolchain (cmake,
# compiler, OpenCV dev headers); the runtime stage only has what's
# actually needed to RUN the compiled binary (OpenCV's shared libraries,
# not its headers/static libs/build tools) — this is what keeps the
# final deployed image small, which is a real, measurable part of the
# "edge deployment" story alongside latency and model size.

FROM ubuntu:24.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential libopencv-dev catch2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt .
COPY include include
COPY src src
COPY tests tests

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" \
    && ./build/edgeguard_tests

FROM ubuntu:24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core406t64 libopencv-imgproc406t64 libopencv-imgcodecs406t64 \
    libopencv-dnn406t64 libopencv-highgui406t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /src/build/edgeguard /app/edgeguard
COPY --from=builder /src/build/edgeguard_bench /app/edgeguard_bench
COPY models /app/models

# No default CMD — this image is meant to be run with explicit
# --model/--image arguments (see README), the same way the CLI works
# outside a container. Resource limits are applied at `docker run` time
# (--cpus, --memory), which is where this project's benchmarked
# constrained-CPU numbers actually come from — see README's "Edge
# deployment" section for the exact command and measured results.
ENTRYPOINT ["/app/edgeguard"]
