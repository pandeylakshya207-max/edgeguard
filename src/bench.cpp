// edgeguard_bench: measures real inference latency/throughput/memory for
// a given model on this machine — the actual evidence behind any
// "runs at Xms on CPU" claim, not an assumed number.
//
// Usage: edgeguard_bench --model=models/ppe.onnx --image=sample.jpg --iters=100

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>

#include "edgeguard/detector.hpp"

namespace {

// Reads current process resident memory (RSS) in kilobytes from
// /proc/self/status — the real, OS-reported figure for this process
// right now, not an estimate derived from model file size (which
// systematically understates true runtime memory: activation buffers,
// OpenCV's internal working memory, and the DNN engine's own allocations
// are not visible from a model file's size on disk alone).
long currentRssKb() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0;
            std::sscanf(line.c_str(), "VmRSS: %ld kB", &kb);
            return kb;
        }
    }
    return -1;  // /proc unavailable (non-Linux) — report "unknown" rather than a fabricated number
}

}  // namespace

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv,
        "{model | models/ppe.onnx | path to ONNX model}"
        "{image |                 | path to a sample image (required)}"
        "{iters | 100             | number of inference iterations to average over}"
        "{warmup| 10              | warmup iterations, excluded from timing}"
        "{size  | 640             | model input size}"
        "{help h|                 | show this help message}");

    if (parser.has("help") || !parser.has("image")) {
        parser.printMessage();
        return parser.has("help") ? 0 : 2;
    }

    std::string modelPath = parser.get<std::string>("model");
    std::string imagePath = parser.get<std::string>("image");
    int iters = parser.get<int>("iters");
    int warmup = parser.get<int>("warmup");
    int size = parser.get<int>("size");

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "edgeguard_bench: could not read image: " << imagePath << "\n";
        return 1;
    }

    long rssBeforeLoad = currentRssKb();
    edgeguard::Detector detector(modelPath, size, /*confThreshold=*/0.25f, /*nmsThreshold=*/0.45f);
    long rssAfterLoad = currentRssKb();

    for (int i = 0; i < warmup; ++i) {
        detector.detect(image);
    }

    std::vector<double> latenciesMs;
    latenciesMs.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto start = std::chrono::steady_clock::now();
        detector.detect(image);
        auto end = std::chrono::steady_clock::now();
        latenciesMs.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    long rssAfterInference = currentRssKb();

    std::sort(latenciesMs.begin(), latenciesMs.end());
    double sum = std::accumulate(latenciesMs.begin(), latenciesMs.end(), 0.0);
    double mean = sum / latenciesMs.size();
    double p50 = latenciesMs[latenciesMs.size() / 2];
    double p95 = latenciesMs[static_cast<size_t>(latenciesMs.size() * 0.95)];
    double p99 = latenciesMs[static_cast<size_t>(latenciesMs.size() * 0.99)];
    double fps = 1000.0 / mean;

    std::cout << "edgeguard_bench — model: " << modelPath << "\n";
    std::cout << "  image: " << imagePath << " (" << image.cols << "x" << image.rows << ")\n";
    std::cout << "  input size: " << size << "x" << size << "\n";
    std::cout << "  iterations: " << iters << " (+" << warmup << " warmup, excluded)\n";
    std::cout << "  --- latency ---\n";
    std::cout << "  mean: " << mean << " ms\n";
    std::cout << "  p50:  " << p50 << " ms\n";
    std::cout << "  p95:  " << p95 << " ms\n";
    std::cout << "  p99:  " << p99 << " ms\n";
    std::cout << "  throughput: " << fps << " FPS (single-threaded)\n";
    std::cout << "  --- memory (RSS, /proc/self/status) ---\n";
    if (rssBeforeLoad >= 0) {
        std::cout << "  before model load: " << rssBeforeLoad << " kB\n";
        std::cout << "  after model load:  " << rssAfterLoad << " kB  (+" << (rssAfterLoad - rssBeforeLoad) << " kB)\n";
        std::cout << "  after inference:   " << rssAfterInference << " kB  (+" << (rssAfterInference - rssAfterLoad) << " kB)\n";
    } else {
        std::cout << "  unavailable on this platform\n";
    }
    return 0;
}
