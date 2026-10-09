#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <opencv2/core.hpp>

#include "edgeguard/detector.hpp"

using edgeguard::decodeYoloOutput;
using Catch::Approx;

// Builds a synthetic YOLOv8-shaped output tensor: (4+numClasses) rows x
// numAnchors columns, so decodeYoloOutput can be tested against known,
// hand-picked values without ever needing a real model file.
static cv::Mat makeOutput(int numClasses, int numAnchors) {
    return cv::Mat::zeros(4 + numClasses, numAnchors, CV_32F);
}

TEST_CASE("decodeYoloOutput extracts a single confident detection correctly", "[detector]") {
    cv::Mat out = makeOutput(/*numClasses=*/2, /*numAnchors=*/1);
    out.at<float>(0, 0) = 100.0f;  // cx
    out.at<float>(1, 0) = 100.0f;  // cy
    out.at<float>(2, 0) = 50.0f;   // w
    out.at<float>(3, 0) = 80.0f;   // h
    out.at<float>(4, 0) = 0.1f;    // class 0 score
    out.at<float>(5, 0) = 0.9f;    // class 1 score — the winner

    auto boxes = decodeYoloOutput(out, /*confThreshold=*/0.5f);
    REQUIRE(boxes.size() == 1);
    REQUIRE(boxes[0].classId == 1);
    REQUIRE(boxes[0].confidence == Approx(0.9f));
    REQUIRE(boxes[0].x1 == Approx(75.0f));   // 100 - 50/2
    REQUIRE(boxes[0].y1 == Approx(60.0f));   // 100 - 80/2
    REQUIRE(boxes[0].x2 == Approx(125.0f));  // 100 + 50/2
    REQUIRE(boxes[0].y2 == Approx(140.0f));  // 100 + 80/2
}

TEST_CASE("decodeYoloOutput filters out anchors below the confidence threshold", "[detector]") {
    cv::Mat out = makeOutput(2, 1);
    out.at<float>(0, 0) = 100.0f;
    out.at<float>(1, 0) = 100.0f;
    out.at<float>(2, 0) = 10.0f;
    out.at<float>(3, 0) = 10.0f;
    out.at<float>(4, 0) = 0.3f;
    out.at<float>(5, 0) = 0.2f;  // best class score is 0.3, below threshold

    auto boxes = decodeYoloOutput(out, 0.5f);
    REQUIRE(boxes.empty());
}

TEST_CASE("decodeYoloOutput picks the correct argmax class and preserves anchor order", "[detector]") {
    cv::Mat out = makeOutput(2, 3);
    // Anchor 0: kept, class 1 wins (0.9 > 0.1)
    out.at<float>(0, 0) = 100; out.at<float>(1, 0) = 100; out.at<float>(2, 0) = 50; out.at<float>(3, 0) = 80;
    out.at<float>(4, 0) = 0.1f; out.at<float>(5, 0) = 0.9f;
    // Anchor 1: filtered (best score 0.3 < threshold 0.5)
    out.at<float>(0, 1) = 200; out.at<float>(1, 1) = 200; out.at<float>(2, 1) = 20; out.at<float>(3, 1) = 20;
    out.at<float>(4, 1) = 0.3f; out.at<float>(5, 1) = 0.2f;
    // Anchor 2: kept, class 0 wins (0.6 > 0.55) — close scores, verifies strict argmax, not a tie-break bug
    out.at<float>(0, 2) = 300; out.at<float>(1, 2) = 300; out.at<float>(2, 2) = 40; out.at<float>(3, 2) = 40;
    out.at<float>(4, 2) = 0.6f; out.at<float>(5, 2) = 0.55f;

    auto boxes = decodeYoloOutput(out, 0.5f);
    REQUIRE(boxes.size() == 2);
    REQUIRE(boxes[0].classId == 1);
    REQUIRE(boxes[0].confidence == Approx(0.9f));
    REQUIRE(boxes[1].classId == 0);
    REQUIRE(boxes[1].confidence == Approx(0.6f));
}

TEST_CASE("decodeYoloOutput rejects a tensor with too few rows to be valid", "[detector]") {
    cv::Mat bad = cv::Mat::zeros(3, 10, CV_32F);  // only 3 rows — can't even hold 4 box coords
    REQUIRE_THROWS_AS(decodeYoloOutput(bad, 0.5f), std::invalid_argument);
}

TEST_CASE("decodeYoloOutput skips an anchor with non-positive width or height instead of emitting an inverted box", "[detector]") {
    // Regression test: an untrained/corrupted/adversarial model can
    // legitimately produce a negative "width" or "height" value at some
    // anchor (nothing in the tensor format itself prevents it) — the raw
    // formula x2 = cx + w/2 would then produce x2 < x1, a geometrically
    // inverted box, if decodeYoloOutput trusted the value blindly. Found
    // via test_detector_integration.cpp's full-pipeline test against a
    // real (if untrained) ONNX model, which is exactly the kind of thing
    // a hand-picked "sensible" unit test wouldn't have exposed.
    cv::Mat out = makeOutput(2, 2);
    // Anchor 0: negative width — must be skipped.
    out.at<float>(0, 0) = 100; out.at<float>(1, 0) = 100;
    out.at<float>(2, 0) = -20.0f; out.at<float>(3, 0) = 30.0f;
    out.at<float>(4, 0) = 0.9f; out.at<float>(5, 0) = 0.1f;
    // Anchor 1: negative height — must be skipped.
    out.at<float>(0, 1) = 200; out.at<float>(1, 1) = 200;
    out.at<float>(2, 1) = 30.0f; out.at<float>(3, 1) = -20.0f;
    out.at<float>(4, 1) = 0.9f; out.at<float>(5, 1) = 0.1f;

    auto boxes = decodeYoloOutput(out, 0.5f);
    REQUIRE(boxes.empty());

    // Every box decodeYoloOutput ever returns must be geometrically
    // valid, as a general property — checked here across a wider mix of
    // valid and invalid anchors, not just the all-invalid case above.
    cv::Mat mixed = makeOutput(2, 3);
    mixed.at<float>(0, 0) = 50; mixed.at<float>(1, 0) = 50; mixed.at<float>(2, 0) = 10; mixed.at<float>(3, 0) = 10;
    mixed.at<float>(4, 0) = 0.9f; mixed.at<float>(5, 0) = 0.0f;  // valid
    mixed.at<float>(0, 1) = 60; mixed.at<float>(1, 1) = 60; mixed.at<float>(2, 1) = -5; mixed.at<float>(3, 1) = 10;
    mixed.at<float>(4, 1) = 0.9f; mixed.at<float>(5, 1) = 0.0f;  // invalid: negative width
    mixed.at<float>(0, 2) = 70; mixed.at<float>(1, 2) = 70; mixed.at<float>(2, 2) = 10; mixed.at<float>(3, 2) = 10;
    mixed.at<float>(4, 2) = 0.9f; mixed.at<float>(5, 2) = 0.0f;  // valid
    auto mixedBoxes = decodeYoloOutput(mixed, 0.5f);
    REQUIRE(mixedBoxes.size() == 2);
    for (const auto& b : mixedBoxes) {
        REQUIRE(b.x2 >= b.x1);
        REQUIRE(b.y2 >= b.y1);
    }
}

TEST_CASE("decodeYoloOutput on an all-zero tensor with a nonzero threshold returns nothing", "[detector]") {
    cv::Mat out = makeOutput(10, 100);  // realistic scale: 10 classes, 100 anchors, everything zero
    auto boxes = decodeYoloOutput(out, 0.25f);
    REQUIRE(boxes.empty());
}

TEST_CASE("decodeYoloOutput in multi-label mode emits one box per class above the threshold", "[detector]") {
    cv::Mat out = makeOutput(/*numClasses=*/3, /*numAnchors=*/2);
    // Anchor 0: classes 0 and 2 clear the threshold, class 1 does not.
    out.at<float>(0, 0) = 100; out.at<float>(1, 0) = 100; out.at<float>(2, 0) = 50; out.at<float>(3, 0) = 80;
    out.at<float>(4, 0) = 0.7f; out.at<float>(5, 0) = 0.2f; out.at<float>(6, 0) = 0.6f;
    // Anchor 1: nothing clears it.
    out.at<float>(0, 1) = 200; out.at<float>(1, 1) = 200; out.at<float>(2, 1) = 20; out.at<float>(3, 1) = 20;
    out.at<float>(4, 1) = 0.1f; out.at<float>(5, 1) = 0.2f; out.at<float>(6, 1) = 0.3f;

    auto boxes = decodeYoloOutput(out, 0.5f, /*multiLabel=*/true);
    REQUIRE(boxes.size() == 2);
    REQUIRE(boxes[0].classId == 0);
    REQUIRE(boxes[0].confidence == Approx(0.7f));
    REQUIRE(boxes[1].classId == 2);
    REQUIRE(boxes[1].confidence == Approx(0.6f));
    // both boxes share the anchor's geometry
    REQUIRE(boxes[0].x1 == Approx(75.0f));
    REQUIRE(boxes[1].x1 == Approx(75.0f));
    REQUIRE(boxes[1].y2 == Approx(140.0f));

    // the default mode keeps only the best class of the same anchor
    auto best = decodeYoloOutput(out, 0.5f);
    REQUIRE(best.size() == 1);
    REQUIRE(best[0].classId == 0);
}

TEST_CASE("decodeYoloOutput in multi-label mode still rejects degenerate boxes and zero scores", "[detector]") {
    cv::Mat out = makeOutput(2, 2);
    // Anchor 0: non-positive width.
    out.at<float>(0, 0) = 100; out.at<float>(1, 0) = 100; out.at<float>(2, 0) = 0; out.at<float>(3, 0) = 80;
    out.at<float>(4, 0) = 0.9f; out.at<float>(5, 0) = 0.9f;
    // Anchor 1: valid box, all-zero scores, threshold zero.
    out.at<float>(0, 1) = 200; out.at<float>(1, 1) = 200; out.at<float>(2, 1) = 20; out.at<float>(3, 1) = 20;

    REQUIRE(decodeYoloOutput(out, 0.0f, /*multiLabel=*/true).empty());
}
