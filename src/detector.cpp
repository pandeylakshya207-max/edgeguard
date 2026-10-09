#include "edgeguard/detector.hpp"

#include <algorithm>
#include <stdexcept>

#include "edgeguard/letterbox.hpp"
#include "edgeguard/nms.hpp"

namespace edgeguard {

std::vector<BoundingBox> decodeYoloOutput(const cv::Mat& output, float confThreshold,
                                           bool multiLabel) {
    // output is (4 + numClasses) rows x numAnchors columns.
    CV_Assert(output.dims == 2);
    int numAttrs = output.rows;
    int numAnchors = output.cols;
    int numClasses = numAttrs - 4;
    if (numClasses <= 0) {
        throw std::invalid_argument("decodeYoloOutput: output has fewer than 5 rows, not a valid YOLOv8 tensor");
    }

    std::vector<BoundingBox> boxes;
    for (int a = 0; a < numAnchors; ++a) {
        float cx = output.at<float>(0, a);
        float cy = output.at<float>(1, a);
        float w = output.at<float>(2, a);
        float h = output.at<float>(3, a);

        // A real trained YOLOv8 model's box-regression head keeps w/h
        // non-negative by construction, but decodeYoloOutput has no way
        // to verify that assumption holds for whatever model file it's
        // actually handed — an untrained, corrupted, or simply
        // maliciously-crafted model could emit a negative width/height,
        // which would silently produce a geometrically inverted box
        // (x2 < x1) downstream. Skip it rather than trust it; this is
        // the same "validate rather than assume" principle
        // BoundingBox::area() already applies to degenerate boxes after
        // construction, applied here at the point of construction
        // instead.
        if (w <= 0.0f || h <= 0.0f) continue;

        BoundingBox box;
        box.x1 = cx - w / 2.0f;
        box.y1 = cy - h / 2.0f;
        box.x2 = cx + w / 2.0f;
        box.y2 = cy + h / 2.0f;

        if (multiLabel) {
            // One box per class that clears the threshold.
            for (int c = 0; c < numClasses; ++c) {
                float score = output.at<float>(4 + c, a);
                if (score <= 0.0f || score < confThreshold) continue;
                box.classId = c;
                box.confidence = score;
                boxes.push_back(box);
            }
            continue;
        }

        int bestClass = -1;
        float bestScore = 0.0f;
        for (int c = 0; c < numClasses; ++c) {
            float score = output.at<float>(4 + c, a);
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }

        if (bestClass < 0 || bestScore < confThreshold) continue;

        box.classId = bestClass;
        box.confidence = bestScore;
        boxes.push_back(box);
    }
    return boxes;
}

Detector::Detector(const std::string& modelPath, int inputSize, float confThreshold,
                    float nmsThreshold, bool multiLabel)
    : inputSize_(inputSize), confThreshold_(confThreshold), nmsThreshold_(nmsThreshold),
      multiLabel_(multiLabel) {
    net_ = cv::dnn::readNet(modelPath);
    if (net_.empty()) {
        throw std::runtime_error("Detector: failed to load model from " + modelPath);
    }
    // CPU-only backend/target — this project's entire premise is
    // realistic edge-deployment latency, so it always runs the same
    // codepath a resource-constrained device without a GPU would use,
    // never silently taking a faster GPU path in a dev environment that
    // then wouldn't represent real deployed performance.
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<BoundingBox> Detector::detect(const cv::Mat& image) {
    LetterboxResult lb = letterbox(image, inputSize_);

    cv::Mat blob;
    cv::dnn::blobFromImage(lb.image, blob, 1.0 / 255.0, cv::Size(inputSize_, inputSize_),
                            cv::Scalar(), /*swapRB=*/true, /*crop=*/false);
    net_.setInput(blob);

    cv::Mat rawOutput = net_.forward();
    // Network output is [1, 4+numClasses, numAnchors]; squeeze the batch
    // dimension down to the 2D matrix decodeYoloOutput expects.
    cv::Mat squeezed(rawOutput.size[1], rawOutput.size[2], CV_32F, rawOutput.ptr<float>());

    std::vector<BoundingBox> boxes = decodeYoloOutput(squeezed, confThreshold_, multiLabel_);
    const float maxX = static_cast<float>(image.cols);
    const float maxY = static_cast<float>(image.rows);
    for (auto& box : boxes) {
        box = unletterboxBox(box, lb);
        // The network can place a box edge slightly outside the picture.
        // Nothing exists out there, so clamp to the image.
        box.x1 = std::clamp(box.x1, 0.0f, maxX);
        box.y1 = std::clamp(box.y1, 0.0f, maxY);
        box.x2 = std::clamp(box.x2, 0.0f, maxX);
        box.y2 = std::clamp(box.y2, 0.0f, maxY);
    }
    return nonMaxSuppression(std::move(boxes), nmsThreshold_);
}

}  // namespace edgeguard
