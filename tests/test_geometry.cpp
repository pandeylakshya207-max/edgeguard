#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "edgeguard/geometry.hpp"

using edgeguard::BoundingBox;
using edgeguard::iou;
using edgeguard::intersectionArea;
using Catch::Approx;

TEST_CASE("identical boxes have IoU of 1.0", "[geometry]") {
    BoundingBox a{0, 0, 10, 10};
    BoundingBox b{0, 0, 10, 10};
    REQUIRE(iou(a, b) == Approx(1.0f));
}

TEST_CASE("non-overlapping boxes have IoU of 0.0", "[geometry]") {
    BoundingBox a{0, 0, 10, 10};
    BoundingBox b{20, 20, 30, 30};
    REQUIRE(iou(a, b) == Approx(0.0f));
    REQUIRE(intersectionArea(a, b) == Approx(0.0f));
}

TEST_CASE("boxes touching at an edge (zero-width overlap) have IoU of 0.0", "[geometry]") {
    // Adjacent, not overlapping — the shared edge has zero area, so this
    // must NOT be treated as any overlap at all. A naive implementation
    // using >= instead of > for the overlap check would get this wrong.
    BoundingBox a{0, 0, 10, 10};
    BoundingBox b{10, 0, 20, 10};
    REQUIRE(iou(a, b) == Approx(0.0f));
}

TEST_CASE("half-overlapping boxes produce the expected IoU", "[geometry]") {
    // a: [0,0]-[10,10] area 100. b: [5,0]-[15,10] area 100.
    // Intersection: [5,0]-[10,10] = 5*10 = 50. Union = 100+100-50 = 150.
    // IoU = 50/150 = 1/3.
    BoundingBox a{0, 0, 10, 10};
    BoundingBox b{5, 0, 15, 10};
    REQUIRE(iou(a, b) == Approx(1.0f / 3.0f));
}

TEST_CASE("one box fully inside another", "[geometry]") {
    // a: 10x10=100. b (inside a): 4x4=16. Intersection = 16 (all of b).
    // Union = 100 + 16 - 16 = 100. IoU = 16/100 = 0.16.
    BoundingBox a{0, 0, 10, 10};
    BoundingBox b{3, 3, 7, 7};
    REQUIRE(iou(a, b) == Approx(0.16f));
}

TEST_CASE("degenerate (zero or negative area) boxes never produce a positive IoU", "[geometry]") {
    // A malformed box (e.g. from a corrupt annotation or a decoding bug)
    // must not silently produce a misleadingly "valid" IoU — this is
    // exactly the kind of input-validation edge case that's easy to skip
    // and hard to notice later, since it only manifests on bad data.
    BoundingBox degenerate{5, 5, 5, 5};  // zero width and height
    BoundingBox normal{0, 0, 10, 10};
    REQUIRE(iou(degenerate, normal) == Approx(0.0f));

    BoundingBox inverted{10, 10, 0, 0};  // x2<x1, y2<y1 — corrupt
    REQUIRE(iou(inverted, normal) == Approx(0.0f));
}

TEST_CASE("width/height/area helpers", "[geometry]") {
    BoundingBox b{2.0f, 3.0f, 12.0f, 9.0f};
    REQUIRE(b.width() == Approx(10.0f));
    REQUIRE(b.height() == Approx(6.0f));
    REQUIRE(b.area() == Approx(60.0f));
}
