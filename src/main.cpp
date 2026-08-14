// edgeguard CLI: run PPE detection on a single image and either display
// or save the annotated result.
//
// Usage:
//   edgeguard --model=models/ppe.onnx --image=photo.jpg --out=result.jpg
//   edgeguard --model=models/ppe.onnx --image=photo.jpg   (opens a window instead)

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "edgeguard/detector.hpp"
#include "edgeguard/labels.hpp"

namespace {

// A fixed, readable color per class rather than random colors every run
// — makes output images comparable across runs when eyeballing results,
// which matters when you're visually sanity-checking a model's mistakes.
cv::Scalar colorForClass(int classId) {
    static const std::vector<cv::Scalar> palette = {
        {0, 255, 0},    {255, 0, 0},    {0, 0, 255},    {255, 255, 0},
        {255, 0, 255},  {0, 255, 255},  {128, 0, 255},  {255, 128, 0},
        {0, 128, 255},  {128, 255, 0},
    };
    return palette[classId % palette.size()];
}

void drawDetections(cv::Mat& image, const std::vector<edgeguard::BoundingBox>& boxes) {
    const auto& names = edgeguard::classNames();
    for (const auto& box : boxes) {
        cv::Scalar color = colorForClass(box.classId);
        cv::rectangle(image, cv::Point(static_cast<int>(box.x1), static_cast<int>(box.y1)),
                      cv::Point(static_cast<int>(box.x2), static_cast<int>(box.y2)), color, 2);

        std::string label = (box.classId >= 0 && box.classId < static_cast<int>(names.size()))
                                 ? names[box.classId]
                                 : "class" + std::to_string(box.classId);
        char confStr[16];
        std::snprintf(confStr, sizeof(confStr), " %.0f%%", box.confidence * 100.0f);
        label += confStr;

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(image,
                      cv::Point(static_cast<int>(box.x1), static_cast<int>(box.y1) - textSize.height - 4),
                      cv::Point(static_cast<int>(box.x1) + textSize.width, static_cast<int>(box.y1)),
                      color, cv::FILLED);
        cv::putText(image, label, cv::Point(static_cast<int>(box.x1), static_cast<int>(box.y1) - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv,
        "{model    |models/ppe.onnx | path to ONNX model}"
        "{image    |               | path to input image (required)}"
        "{out      |               | path to save annotated output (optional; opens a window if omitted)}"
        "{conf     |0.25           | confidence threshold}"
        "{nms      |0.45           | NMS IoU threshold}"
        "{size     |640            | model input size}"
        "{help h   |               | show this help message}");

    if (parser.has("help") || !parser.has("image")) {
        parser.printMessage();
        return parser.has("help") ? 0 : 2;
    }

    std::string modelPath = parser.get<std::string>("model");
    std::string imagePath = parser.get<std::string>("image");
    std::string outPath = parser.get<std::string>("out");
    float conf = parser.get<float>("conf");
    float nms = parser.get<float>("nms");
    int size = parser.get<int>("size");

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "edgeguard: could not read image: " << imagePath << "\n";
        return 1;
    }

    edgeguard::Detector detector(modelPath, size, conf, nms);
    std::vector<edgeguard::BoundingBox> boxes = detector.detect(image);

    std::cout << "edgeguard: " << boxes.size() << " detection(s)\n";
    const auto& names = edgeguard::classNames();
    for (const auto& box : boxes) {
        std::string name = (box.classId >= 0 && box.classId < static_cast<int>(names.size()))
                                ? names[box.classId]
                                : "?";
        std::cout << "  " << name << "  conf=" << box.confidence << "  box=(" << box.x1 << ","
                  << box.y1 << ")-(" << box.x2 << "," << box.y2 << ")\n";
    }

    drawDetections(image, boxes);

    if (!outPath.empty()) {
        cv::imwrite(outPath, image);
        std::cout << "edgeguard: wrote annotated image to " << outPath << "\n";
    } else {
        cv::imshow("edgeguard", image);
        cv::waitKey(0);
    }
    return 0;
}
