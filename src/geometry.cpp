#include "edgeguard/geometry.hpp"

#include <algorithm>

namespace edgeguard {

float intersectionArea(const BoundingBox& a, const BoundingBox& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);

    float iw = ix2 - ix1;
    float ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) {
        return 0.0f;  // no overlap
    }
    return iw * ih;
}

float iou(const BoundingBox& a, const BoundingBox& b) {
    float inter = intersectionArea(a, b);
    if (inter <= 0.0f) return 0.0f;

    float unionArea = a.area() + b.area() - inter;
    if (unionArea <= 0.0f) return 0.0f;  // both boxes degenerate

    return inter / unionArea;
}

}  // namespace edgeguard
