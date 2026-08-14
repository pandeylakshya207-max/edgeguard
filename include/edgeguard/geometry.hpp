#pragma once

namespace edgeguard {

// BoundingBox uses absolute pixel coordinates (x1,y1) top-left and
// (x2,y2) bottom-right — not center/width/height (YOLO's native output
// format) and not normalized [0,1] coordinates (the format annotation
// files use). Every producer of a BoundingBox (label parsing, model
// output decoding) is responsible for converting into this single
// canonical representation, so every consumer (IoU, NMS, evaluation)
// only ever has to handle one format.
struct BoundingBox {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    int classId = -1;
    float confidence = 0.0f;

    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const {
        float w = width();
        float h = height();
        // A degenerate box (from malformed input) must contribute zero
        // area, not a negative one that would corrupt downstream IoU
        // math — clamp rather than trust the input.
        if (w <= 0.0f || h <= 0.0f) return 0.0f;
        return w * h;
    }
};

// intersectionArea returns the area of overlap between two boxes, 0 if
// they don't overlap at all.
float intersectionArea(const BoundingBox& a, const BoundingBox& b);

// iou (Intersection over Union) is the standard measure of how well two
// boxes overlap: 1.0 for identical boxes, 0.0 for no overlap. Used both
// by NMS (suppressing duplicate detections of the same object) and by
// evaluation (deciding whether a prediction "matches" a ground-truth
// box, conventionally at IoU >= 0.5).
float iou(const BoundingBox& a, const BoundingBox& b);

}  // namespace edgeguard
