#pragma once

#include <vector>

#include "edgeguard/geometry.hpp"

namespace edgeguard {

// nonMaxSuppression removes redundant overlapping detections of the same
// object. A detector like YOLO typically fires multiple overlapping boxes
// for a single real object; NMS keeps only the highest-confidence one per
// cluster of overlapping boxes and discards the rest.
//
// This is class-aware: suppression only happens between boxes of the SAME
// classId. Two boxes of different classes (e.g. a "Person" box and a
// "Hardhat" box) are expected to genuinely overlap — a hardhat sits on a
// person's head — so they must never suppress each other regardless of
// IoU. A naive class-agnostic NMS would silently delete correct
// detections in exactly this kind of multi-class, spatially-nested
// scenario, which is exactly the failure mode this dataset's classes
// (Person + Hardhat + Safety Vest, all overlapping the same region) would
// expose.
//
// boxes is taken by value (not reference) because the algorithm sorts
// and partitions its working copy — callers keep their own original list
// untouched.
std::vector<BoundingBox> nonMaxSuppression(std::vector<BoundingBox> boxes,
                                            float iouThreshold);

}  // namespace edgeguard
