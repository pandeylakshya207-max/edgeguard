#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "edgeguard/evaluator.hpp"

using edgeguard::BoundingBox;
using edgeguard::Detection;
using edgeguard::GroundTruth;
using edgeguard::averagePrecision;
using edgeguard::meanAveragePrecision;
using Catch::Approx;

TEST_CASE("a perfect detector (every GT matched exactly, no false positives) has AP of 1.0", "[evaluator]") {
    std::vector<GroundTruth> gt = {
        {{0, 0, 10, 10, 0, 1.0f}, /*imageId=*/0},
        {{20, 20, 30, 30, 0, 1.0f}, /*imageId=*/0},
    };
    std::vector<Detection> preds = {
        {{0, 0, 10, 10, 0, 0.9f}, 0},
        {{20, 20, 30, 30, 0, 0.8f}, 0},
    };
    auto ap = averagePrecision(preds, gt, /*classId=*/0, /*iouThreshold=*/0.5f);
    REQUIRE(ap.has_value());
    REQUIRE(*ap == Approx(1.0f));
}

TEST_CASE("a class with ground truth but zero predictions has AP of 0.0", "[evaluator]") {
    std::vector<GroundTruth> gt = {{{0, 0, 10, 10, 0, 1.0f}, 0}};
    std::vector<Detection> preds = {};  // detector never fires on this class
    auto ap = averagePrecision(preds, gt, 0, 0.5f);
    REQUIRE(ap.has_value());
    REQUIRE(*ap == Approx(0.0f));
}

TEST_CASE("a class absent from ground truth entirely returns nullopt, not 0", "[evaluator]") {
    std::vector<GroundTruth> gt = {{{0, 0, 10, 10, /*classId=*/1, 1.0f}, 0}};  // only class 1 present
    std::vector<Detection> preds = {{{0, 0, 10, 10, 0, 0.9f}, 0}};
    auto ap = averagePrecision(preds, gt, /*classId=*/0, 0.5f);
    REQUIRE_FALSE(ap.has_value());
}

TEST_CASE("a false positive between two true positives produces the hand-computed AP of 5/6", "[evaluator]") {
    // 2 ground-truth boxes; 3 predictions in confidence order: TP, FP, TP.
    // Worked by hand in this module's design notes:
    //   rank1 (TP): precision=1.0,   recall=0.5
    //   rank2 (FP): precision=0.5,   recall=0.5
    //   rank3 (TP): precision=0.667, recall=1.0
    // After all-point interpolation: precision = [1.0, 0.667, 0.667]
    // AP = 0.5*1.0 + 0*0.667 + 0.5*0.667 = 0.8333... = 5/6
    std::vector<GroundTruth> gt = {
        {{0, 0, 10, 10, 0, 1.0f}, 0},
        {{100, 100, 110, 110, 0, 1.0f}, 0},
    };
    std::vector<Detection> preds = {
        {{0, 0, 10, 10, 0, 0.9f}, 0},           // TP: matches GT1
        {{500, 500, 510, 510, 0, 0.85f}, 0},    // FP: matches nothing
        {{100, 100, 110, 110, 0, 0.8f}, 0},     // TP: matches GT2
    };
    auto ap = averagePrecision(preds, gt, 0, 0.5f);
    REQUIRE(ap.has_value());
    REQUIRE(*ap == Approx(5.0f / 6.0f).margin(0.001));
}

TEST_CASE("a ground-truth box can only be matched once — a duplicate prediction is a false positive", "[evaluator]") {
    // Single ground-truth box, two predictions both landing on it. The
    // second (lower-confidence) one must NOT count as a second true
    // positive — exactly the redundant-detection problem NMS exists to
    // clean up before evaluation, verified independently here at the
    // evaluator level too (defense in depth: correct evaluation must not
    // depend on the detector's NMS having already run correctly).
    std::vector<GroundTruth> gt = {{{0, 0, 10, 10, 0, 1.0f}, 0}};
    std::vector<Detection> preds = {
        {{0, 0, 10, 10, 0, 0.9f}, 0},
        {{1, 1, 11, 11, 0, 0.8f}, 0},  // overlaps the same GT box, but it's already claimed
    };
    auto ap = averagePrecision(preds, gt, 0, 0.5f);
    REQUIRE(ap.has_value());
    // Recall saturates at 1.0 on the first (correct) prediction with
    // precision 1.0; the redundant second prediction doesn't change AP
    // because interpolation takes the max precision at or after each
    // recall level, and recall never advances past the first TP. This is
    // the standard, correct behavior of the metric — not a bug — and is
    // exactly why relying on AP alone doesn't excuse skipping NMS in a
    // real pipeline (precision at LOWER recall levels than this one, in
    // a less trivial example, absolutely would be hurt by the same kind
    // of redundant false positive).
    REQUIRE(*ap == Approx(1.0f));
}

TEST_CASE("a prediction on an image with no ground truth of that class is a false positive", "[evaluator]") {
    std::vector<GroundTruth> gt = {{{0, 0, 10, 10, 0, 1.0f}, /*imageId=*/1}};  // only on image 1
    std::vector<Detection> preds = {{{0, 0, 10, 10, 0, 0.9f}, /*imageId=*/2}};  // fires on image 2 instead
    auto ap = averagePrecision(preds, gt, 0, 0.5f);
    REQUIRE(ap.has_value());
    REQUIRE(*ap == Approx(0.0f));
}

TEST_CASE("meanAveragePrecision averages only over classes present in ground truth", "[evaluator]") {
    // Class 0: perfect detector (AP=1.0). Class 1: absent from ground
    // truth entirely (must be excluded, not treated as AP=0). Class 2:
    // present but never predicted (AP=0.0). mAP should be the average of
    // ONLY class 0 and class 2: (1.0 + 0.0) / 2 = 0.5 — if class 1 were
    // incorrectly included as a 0, mAP would be 1/3 instead.
    std::vector<GroundTruth> gt = {
        {{0, 0, 10, 10, 0, 1.0f}, 0},
        {{0, 0, 10, 10, 2, 1.0f}, 0},
    };
    std::vector<Detection> preds = {{{0, 0, 10, 10, 0, 0.9f}, 0}};

    auto result = meanAveragePrecision(preds, gt, /*numClasses=*/3, 0.5f);
    REQUIRE(result.perClassAP[0].has_value());
    REQUIRE(*result.perClassAP[0] == Approx(1.0f));
    REQUIRE_FALSE(result.perClassAP[1].has_value());
    REQUIRE(result.perClassAP[2].has_value());
    REQUIRE(*result.perClassAP[2] == Approx(0.0f));
    REQUIRE(result.mAP == Approx(0.5f));
}
