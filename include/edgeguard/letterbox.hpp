#pragma once

#include <opencv2/core.hpp>

#include "edgeguard/geometry.hpp"

namespace edgeguard {

// LetterboxResult captures both the transformed image and everything
// needed to undo the transform on detection results afterward. Losing
// track of scale/padding is a classic, easy-to-miss YOLO deployment bug:
// forget to unletterbox and every box is silently misplaced on the
// original image, in a way that's obvious in a demo (boxes look "close
// but off") but easy to not notice in an automated pipeline that never
// visually inspects output.
struct LetterboxResult {
    cv::Mat image;   // targetSize x targetSize, aspect ratio preserved, padded with gray
    float scale = 1.0f;  // factor the original image was scaled by
    int padLeft = 0;
    int padTop = 0;
};

// letterbox resizes src to fit within a targetSize x targetSize square
// while preserving aspect ratio, padding the remaining space (centered)
// with a neutral gray — the standard preprocessing YOLO-family models
// expect, since naively stretching to a square would distort object
// shapes and hurt accuracy.
LetterboxResult letterbox(const cv::Mat& src, int targetSize);

// unletterboxBox maps a box's coordinates from the letterboxed image
// space back to the original image's coordinate space, using the scale
// and padding recorded in `result`. Must be applied to every detection
// before it means anything relative to the original image (e.g. before
// drawing it, or before comparing it against ground-truth boxes which
// are always in original-image coordinates).
BoundingBox unletterboxBox(const BoundingBox& box, const LetterboxResult& result);

}  // namespace edgeguard
