#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

#include "edgeguard/geometry.hpp"

namespace edgeguard {

// decodeYoloOutput parses a YOLOv8-style ONNX output tensor into
// BoundingBoxes. YOLOv8's export convention (unlike YOLOv5's) has no
// separate objectness score — output shape is [1, 4+numClasses,
// numAnchors]: 4 rows of box coordinates (center-x, center-y, width,
// height, in pixel coordinates relative to the model's input size)
// followed by numClasses rows of per-class confidence, one column per
// anchor. This function takes the already-squeezed (4+numClasses) x
// numAnchors matrix (batch dimension removed) — squeezing the raw
// network output is the caller's job (Detector::detect does it), so this
// function can be unit tested against a hand-built matrix without ever
// touching cv::dnn::Net or a real model file.
//
// A box is kept only if its best class score exceeds confThreshold.
// classId is the argmax over the class-score rows; confidence is that
// class's score. Returned boxes are in the SAME pixel coordinate space
// as the input tensor (the letterboxed model-input space) — the caller
// is responsible for unletterboxing them back to original-image
// coordinates, and for running NMS (this function deliberately does
// neither, so each concern stays independently testable).
//
// With multiLabel = true, an anchor yields one box for EVERY class whose
// score reaches confThreshold instead of only its best class. A deployed
// detector wants the single best class; the multi-label form exists so
// that edgeguard_eval can follow the same protocol as the Ultralytics
// validator, which scores every class of every anchor.
std::vector<BoundingBox> decodeYoloOutput(const cv::Mat& output, float confThreshold,
                                           bool multiLabel = false);

// Detector wraps a cv::dnn::Net loaded from an ONNX (or Darknet) model
// file and runs the full detect pipeline: letterbox preprocess -> forward
// pass -> decode -> unletterbox -> NMS. This is the integration point
// that actually needs a real model file to exercise end-to-end; the
// pieces it calls (letterbox, decodeYoloOutput, nonMaxSuppression) are
// all independently unit tested without one.
class Detector {
 public:
    Detector(const std::string& modelPath, int inputSize, float confThreshold,
             float nmsThreshold, bool multiLabel = false);

    std::vector<BoundingBox> detect(const cv::Mat& image);

 private:
    cv::dnn::Net net_;
    int inputSize_;
    float confThreshold_;
    float nmsThreshold_;
    bool multiLabel_;
};

}  // namespace edgeguard
