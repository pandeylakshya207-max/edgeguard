// edgeguard_eval: runs the detector over a full validation split and
// reports per-class and overall mean Average Precision — the actual
// accuracy evidence for this project, not an assumed number.
//
// Expects a directory with images/*.jpg and labels/*.txt (YOLO format,
// matching filenames minus extension) — exactly what a Roboflow
// YOLOv8-format export's valid/ or test/ split looks like.
//
// Usage: edgeguard_eval --model=models/ppe.onnx --data=data/valid --iou=0.5

#include <filesystem>
#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>

#include "edgeguard/detector.hpp"
#include "edgeguard/evaluator.hpp"
#include "edgeguard/labels.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv,
        "{model | models/ppe.onnx | path to ONNX model}"
        "{data  |                 | path to a split directory containing images/ and labels/ (required)}"
        "{conf  | 0.001           | confidence threshold — deliberately low for evaluation, so the "
                                    "precision-recall curve isn't truncated before it's built; --conf on "
                                    "the CLI/bench tools is a real serving threshold, this is not}"
        "{nms   | 0.45            | NMS IoU threshold}"
        "{iou   | 0.5             | IoU threshold for counting a detection as a true positive}"
        "{size  | 640             | model input size}"
        "{threads | 0             | CPU threads OpenCV may use; 0 leaves OpenCV's default (all cores)}"
        "{multi-label |           | score every class of every anchor, as the Ultralytics validator does, "
                                    "instead of only the best class per anchor}"
        "{help h|                 | show this help message}");

    if (parser.has("help") || !parser.has("data")) {
        parser.printMessage();
        return parser.has("help") ? 0 : 2;
    }

    std::string modelPath = parser.get<std::string>("model");
    fs::path dataDir = parser.get<std::string>("data");
    float conf = parser.get<float>("conf");
    float nms = parser.get<float>("nms");
    float iouThresh = parser.get<float>("iou");
    int size = parser.get<int>("size");
    bool multiLabel = parser.has("multi-label");
    int threads = parser.get<int>("threads");
    if (threads > 0) cv::setNumThreads(threads);

    fs::path imagesDir = dataDir / "images";
    fs::path labelsDir = dataDir / "labels";
    if (!fs::exists(imagesDir) || !fs::exists(labelsDir)) {
        std::cerr << "edgeguard_eval: expected " << imagesDir << " and " << labelsDir << " to exist\n";
        return 1;
    }

    edgeguard::Detector detector(modelPath, size, conf, nms, multiLabel);

    std::vector<edgeguard::Detection> allPredictions;
    std::vector<edgeguard::GroundTruth> allGroundTruth;

    int imageId = 0;
    int numImages = 0;
    for (const auto& entry : fs::directory_iterator(imagesDir)) {
        if (!entry.is_regular_file()) continue;
        fs::path imgPath = entry.path();
        std::string ext = imgPath.extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        cv::Mat image = cv::imread(imgPath.string());
        if (image.empty()) {
            std::cerr << "  warning: could not read " << imgPath << ", skipping\n";
            continue;
        }

        fs::path labelPath = labelsDir / (imgPath.stem().string() + ".txt");
        auto gtBoxes = edgeguard::parseYoloLabelFile(labelPath.string(), image.cols, image.rows);
        for (const auto& box : gtBoxes) {
            allGroundTruth.push_back({box, imageId});
        }

        auto detections = detector.detect(image);
        for (const auto& box : detections) {
            allPredictions.push_back({box, imageId});
        }

        ++imageId;
        ++numImages;
        if (numImages % 100 == 0) {
            std::cout << "  processed " << numImages << " images...\n";
        }
    }

    std::cout << "\nedgeguard_eval — " << numImages << " images, "
              << allGroundTruth.size() << " ground-truth boxes, "
              << allPredictions.size() << " predictions (before per-class AP matching)\n";
    std::cout << "IoU threshold: " << iouThresh << ", NMS threshold: " << nms << ", confidence >= " << conf
              << ", " << (multiLabel ? "every class per anchor" : "best class per anchor") << "\n\n";

    auto result = edgeguard::meanAveragePrecision(allPredictions, allGroundTruth,
                                                    static_cast<int>(edgeguard::classNames().size()),
                                                    iouThresh);

    const auto& names = edgeguard::classNames();
    std::cout << "Per-class AP:\n";
    for (size_t c = 0; c < names.size(); ++c) {
        std::cout << "  " << names[c] << ": ";
        if (result.perClassAP[c].has_value()) {
            std::cout << *result.perClassAP[c];
        } else {
            std::cout << "(no ground truth for this class)";
        }
        std::cout << "\n";
    }
    std::cout << "\nmAP@" << iouThresh << ": " << result.mAP << "\n";
    return 0;
}
