#include <catch2/catch_test_macros.hpp>

#include "edgeguard/nms.hpp"

using edgeguard::BoundingBox;
using edgeguard::nonMaxSuppression;

TEST_CASE("NMS on empty input returns empty", "[nms]") {
    REQUIRE(nonMaxSuppression({}, 0.5f).empty());
}

TEST_CASE("NMS keeps a single box unchanged", "[nms]") {
    std::vector<BoundingBox> boxes = {{0, 0, 10, 10, /*classId=*/0, /*confidence=*/0.9f}};
    auto kept = nonMaxSuppression(boxes, 0.5f);
    REQUIRE(kept.size() == 1);
}

TEST_CASE("NMS suppresses a lower-confidence duplicate of the same class", "[nms]") {
    std::vector<BoundingBox> boxes = {
        {0, 0, 10, 10, /*classId=*/0, /*confidence=*/0.95f},   // best
        {1, 1, 11, 11, /*classId=*/0, /*confidence=*/0.80f},   // heavily overlapping, same class -> suppressed
    };
    auto kept = nonMaxSuppression(boxes, 0.5f);
    REQUIRE(kept.size() == 1);
    REQUIRE(kept[0].confidence == 0.95f);
}

TEST_CASE("NMS keeps non-overlapping boxes of the same class", "[nms]") {
    std::vector<BoundingBox> boxes = {
        {0, 0, 10, 10, 0, 0.9f},
        {100, 100, 110, 110, 0, 0.8f},
    };
    auto kept = nonMaxSuppression(boxes, 0.5f);
    REQUIRE(kept.size() == 2);
}

TEST_CASE("NMS never suppresses across different classes, even with full overlap", "[nms]") {
    // This is the design property the module exists to get right for
    // this dataset: a Hardhat box and a Person box legitimately occupy
    // the same pixels (the hardhat sits on the person's head). A
    // class-agnostic NMS would incorrectly delete one of these.
    std::vector<BoundingBox> boxes = {
        {0, 0, 10, 10, /*classId=*/0 /* Person */, 0.9f},
        {0, 0, 10, 10, /*classId=*/1 /* Hardhat */, 0.85f},  // identical box, different class
    };
    auto kept = nonMaxSuppression(boxes, 0.5f);
    REQUIRE(kept.size() == 2);
}

TEST_CASE("NMS threshold boundary: IoU exactly at threshold is NOT suppressed", "[nms]") {
    // Suppression condition is strictly greater-than the threshold, so a
    // box at exactly the threshold survives — verifies the boundary isn't
    // silently off-by-one in either direction.
    BoundingBox a{0, 0, 10, 10, 0, 0.9f};
    BoundingBox b{5, 0, 15, 10, 0, 0.8f};  // IoU = 1/3, per the geometry tests
    auto kept = nonMaxSuppression({a, b}, 1.0f / 3.0f);
    REQUIRE(kept.size() == 2);

    auto keptStrict = nonMaxSuppression({a, b}, 1.0f / 3.0f - 0.0001f);
    REQUIRE(keptStrict.size() == 1);
}

TEST_CASE("NMS output is confidence-sorted", "[nms]") {
    std::vector<BoundingBox> boxes = {
        {0, 0, 5, 5, 0, 0.3f},
        {100, 100, 105, 105, 0, 0.9f},
        {200, 200, 205, 205, 0, 0.6f},
    };
    auto kept = nonMaxSuppression(boxes, 0.5f);
    REQUIRE(kept.size() == 3);
    REQUIRE(kept[0].confidence == 0.9f);
    REQUIRE(kept[1].confidence == 0.6f);
    REQUIRE(kept[2].confidence == 0.3f);
}
