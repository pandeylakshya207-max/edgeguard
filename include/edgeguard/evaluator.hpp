#pragma once

#include <optional>
#include <vector>

#include "edgeguard/geometry.hpp"

namespace edgeguard {

// Detection/GroundTruth tag a BoundingBox with which image it came from —
// essential because matching a prediction to a ground-truth box must
// never cross image boundaries (a prediction on image 7 can only match a
// ground-truth box that's also on image 7, no matter how similar the
// coordinates happen to look).
struct Detection {
    BoundingBox box;
    int imageId = -1;
};

struct GroundTruth {
    BoundingBox box;
    int imageId = -1;
};

// averagePrecision computes AP for a single class using the standard
// all-point interpolation (PASCAL VOC 2010+ / COCO convention): precision
// at each recall level is interpolated to the maximum precision achieved
// at any recall >= that level (which makes the precision-recall curve
// monotonically non-increasing before integrating), then AP is the area
// under that interpolated curve.
//
// Matching rule: predictions are processed in descending confidence
// order; each is matched against the highest-IoU not-yet-matched
// ground-truth box of the same class in the same image. A match counts
// as a true positive only if IoU >= iouThreshold AND that ground-truth
// box hasn't already been claimed by a higher-confidence prediction —
// this "each ground-truth box can only be matched once" rule is what
// stops a detector from earning credit for firing five overlapping boxes
// at one real object.
//
// Returns std::nullopt if there are zero ground-truth instances of this
// class in the dataset — AP is undefined in that case (not 0), and
// meanAveragePrecision correctly excludes such classes from the average
// rather than penalizing the model for a class that was never present.
std::optional<float> averagePrecision(const std::vector<Detection>& predictions,
                                       const std::vector<GroundTruth>& groundTruths,
                                       int classId, float iouThreshold);

struct MeanAveragePrecisionResult {
    std::vector<std::optional<float>> perClassAP;  // indexed by classId; nullopt = class absent from ground truth
    float mAP = 0.0f;  // mean over classes that had at least one ground-truth instance
};

MeanAveragePrecisionResult meanAveragePrecision(const std::vector<Detection>& predictions,
                                                 const std::vector<GroundTruth>& groundTruths,
                                                 int numClasses, float iouThreshold);

}  // namespace edgeguard
