#include "edgeguard/letterbox.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace edgeguard {

LetterboxResult letterbox(const cv::Mat& src, int targetSize) {
    LetterboxResult result;

    int srcW = src.cols;
    int srcH = src.rows;

    // Scale by the SMALLER ratio so the larger dimension fits exactly
    // within targetSize and the smaller dimension has room to be padded
    // — this is what "preserve aspect ratio, fit within bounds" means.
    float scaleW = static_cast<float>(targetSize) / static_cast<float>(srcW);
    float scaleH = static_cast<float>(targetSize) / static_cast<float>(srcH);
    result.scale = std::min(scaleW, scaleH);

    int newW = static_cast<int>(std::round(srcW * result.scale));
    int newH = static_cast<int>(std::round(srcH * result.scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    // Center the resized image within the target square. When the total
    // padding is odd (targetSize - newW is odd), floor-dividing for both
    // sides would drop a pixel; the right/bottom edge absorbs that extra
    // pixel via copyMakeBorder's explicit per-side arguments below.
    result.padLeft = (targetSize - newW) / 2;
    result.padTop = (targetSize - newH) / 2;
    int padRight = targetSize - newW - result.padLeft;
    int padBottom = targetSize - newH - result.padTop;

    cv::copyMakeBorder(resized, result.image, result.padTop, padBottom,
                        result.padLeft, padRight, cv::BORDER_CONSTANT,
                        cv::Scalar(114, 114, 114));  // neutral gray, standard YOLO convention

    return result;
}

BoundingBox unletterboxBox(const BoundingBox& box, const LetterboxResult& result) {
    BoundingBox out = box;
    out.x1 = (box.x1 - result.padLeft) / result.scale;
    out.y1 = (box.y1 - result.padTop) / result.scale;
    out.x2 = (box.x2 - result.padLeft) / result.scale;
    out.y2 = (box.y2 - result.padTop) / result.scale;
    return out;
}

}  // namespace edgeguard
