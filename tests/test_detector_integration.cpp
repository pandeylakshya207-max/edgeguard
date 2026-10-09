#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <opencv2/imgproc.hpp>

#include "edgeguard/detector.hpp"
#include "edgeguard/test_config.hpp"

// These tests exercise the FULL Detector::detect() pipeline — letterbox,
// blob conversion, a real cv::dnn::Net::forward() pass, output decoding,
// unletterboxing, and NMS — end to end, using a real (if untrained) ONNX
// model. Every module Detector calls is already unit tested in isolation
// with synthetic inputs (see the other test files); what's uniquely
// verified HERE is that they're wired together correctly and that a real
// cv::dnn execution round-trips through all of them without crashing,
// producing correctly-shaped, sane output. It deliberately does NOT test
// detection accuracy — mock.onnx has random, untrained weights, so its
// output classes/boxes are meaningless; only the plumbing is under test.
//
// See scripts/make_mock_model.py for how models/mock.onnx was built —
// small enough (11KB) to commit directly as a test fixture, unlike a
// real trained model.

namespace fs = std::filesystem;

namespace {
std::string mockModelPath() {
    fs::path path = fs::path(EDGEGUARD_SOURCE_DIR) / "models" / "mock.onnx";
    if (fs::exists(path)) return path.string();
    return "";
}
}  // namespace

TEST_CASE("Detector loads a real ONNX model without throwing", "[detector][integration]") {
    std::string path = mockModelPath();
    if (path.empty()) {
        WARN("models/mock.onnx not found — skipping (run scripts/make_mock_model.py first)");
        return;
    }
    REQUIRE_NOTHROW(edgeguard::Detector(path, /*inputSize=*/640, 0.01f, 0.45f));
}

TEST_CASE("Detector::detect runs the full pipeline end-to-end on a real image without crashing",
          "[detector][integration]") {
    std::string path = mockModelPath();
    if (path.empty()) {
        WARN("models/mock.onnx not found — skipping (run scripts/make_mock_model.py first)");
        return;
    }

    // A real (if synthetic) non-square, non-640 image — exercises the
    // letterbox path with actual resizing/padding math, not just a
    // pre-sized 640x640 input that would trivially skip it.
    cv::Mat image(480, 720, CV_8UC3, cv::Scalar(120, 130, 140));
    cv::rectangle(image, cv::Point(100, 100), cv::Point(300, 400), cv::Scalar(200, 50, 50), -1);

    // Very low confidence threshold: mock.onnx's random untrained
    // weights won't produce realistic confident detections, but the
    // pipeline must still run to completion and return SOME
    // well-formed (if meaningless) result rather than crashing.
    edgeguard::Detector detector(path, 640, /*confThreshold=*/0.001f, /*nmsThreshold=*/0.45f);
    std::vector<edgeguard::BoundingBox> boxes;
    REQUIRE_NOTHROW(boxes = detector.detect(image));

    // Whatever came back must be well-formed: valid class IDs, and boxes
    // that make geometric sense (even if the model itself is untrained
    // and their content is meaningless).
    for (const auto& box : boxes) {
        REQUIRE(box.classId >= 0);
        REQUIRE(box.classId < 10);
        REQUIRE(box.confidence >= 0.001f);
        REQUIRE(box.x2 >= box.x1);
        REQUIRE(box.y2 >= box.y1);
        // detections are clamped to the picture they came from
        REQUIRE(box.x1 >= 0.0f);
        REQUIRE(box.y1 >= 0.0f);
        REQUIRE(box.x2 <= 720.0f);
        REQUIRE(box.y2 <= 480.0f);
    }
}

TEST_CASE("Detector in multi-label mode returns at least as many boxes as in best-class mode",
          "[detector][integration]") {
    std::string path = mockModelPath();
    if (path.empty()) {
        WARN("models/mock.onnx not found — skipping (run scripts/make_mock_model.py first)");
        return;
    }
    cv::Mat image(480, 720, CV_8UC3, cv::Scalar(120, 130, 140));
    cv::rectangle(image, cv::Point(100, 100), cv::Point(300, 400), cv::Scalar(200, 50, 50), -1);

    // NMS threshold 1.0 suppresses nothing, so the two counts compare the decoders alone.
    edgeguard::Detector bestClass(path, 640, 0.001f, 1.0f);
    edgeguard::Detector everyClass(path, 640, 0.001f, 1.0f, /*multiLabel=*/true);
    REQUIRE(everyClass.detect(image).size() >= bestClass.detect(image).size());
}
