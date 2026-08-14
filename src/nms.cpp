#include "edgeguard/nms.hpp"

#include <algorithm>

namespace edgeguard {

std::vector<BoundingBox> nonMaxSuppression(std::vector<BoundingBox> boxes,
                                            float iouThreshold) {
    // Highest confidence first: greedy NMS always keeps the current best
    // remaining box and suppresses everything that overlaps it too much,
    // which is only correct if we process boxes in descending confidence
    // order.
    std::sort(boxes.begin(), boxes.end(),
              [](const BoundingBox& a, const BoundingBox& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<BoundingBox> kept;
    kept.reserve(boxes.size());

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(boxes[i]);

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            // Class-aware: only compare and possibly suppress boxes of
            // the same class — see the header's doc comment for why this
            // matters for this specific dataset.
            if (boxes[j].classId != boxes[i].classId) continue;

            if (iou(boxes[i], boxes[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return kept;
}

}  // namespace edgeguard
