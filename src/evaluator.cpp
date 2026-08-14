#include "edgeguard/evaluator.hpp"

#include <algorithm>
#include <map>

namespace edgeguard {

std::optional<float> averagePrecision(const std::vector<Detection>& predictions,
                                       const std::vector<GroundTruth>& groundTruths,
                                       int classId, float iouThreshold) {
    // Filter to this class only, and group ground truths by image so
    // matching never crosses image boundaries.
    std::map<int, std::vector<GroundTruth>> gtByImage;
    int numGroundTruth = 0;
    for (const auto& gt : groundTruths) {
        if (gt.box.classId != classId) continue;
        gtByImage[gt.imageId].push_back(gt);
        ++numGroundTruth;
    }
    if (numGroundTruth == 0) {
        return std::nullopt;  // class not present in this dataset — AP undefined
    }

    std::vector<Detection> preds;
    for (const auto& d : predictions) {
        if (d.box.classId == classId) preds.push_back(d);
    }
    std::sort(preds.begin(), preds.end(), [](const Detection& a, const Detection& b) {
        return a.box.confidence > b.box.confidence;
    });

    // matched[imageId][i] tracks whether ground-truth box i in that image
    // has already been claimed by an earlier (higher-confidence)
    // prediction — each ground-truth box may only produce one TP.
    std::map<int, std::vector<bool>> matched;
    for (const auto& [imgId, boxes] : gtByImage) {
        matched[imgId] = std::vector<bool>(boxes.size(), false);
    }

    std::vector<int> tp(preds.size(), 0);
    std::vector<int> fp(preds.size(), 0);

    for (size_t i = 0; i < preds.size(); ++i) {
        const auto& pred = preds[i];
        auto it = gtByImage.find(pred.imageId);
        if (it == gtByImage.end()) {
            fp[i] = 1;  // prediction on an image with no ground truth of this class at all
            continue;
        }

        const auto& candidates = it->second;
        auto& matchedFlags = matched[pred.imageId];

        float bestIou = 0.0f;
        int bestIdx = -1;
        for (size_t g = 0; g < candidates.size(); ++g) {
            if (matchedFlags[g]) continue;  // already claimed
            float overlap = iou(pred.box, candidates[g].box);
            if (overlap > bestIou) {
                bestIou = overlap;
                bestIdx = static_cast<int>(g);
            }
        }

        if (bestIdx >= 0 && bestIou >= iouThreshold) {
            tp[i] = 1;
            matchedFlags[bestIdx] = true;
        } else {
            fp[i] = 1;
        }
    }

    // Cumulative TP/FP -> precision/recall at each prediction rank.
    std::vector<float> precision(preds.size());
    std::vector<float> recall(preds.size());
    int cumTp = 0, cumFp = 0;
    for (size_t i = 0; i < preds.size(); ++i) {
        cumTp += tp[i];
        cumFp += fp[i];
        precision[i] = static_cast<float>(cumTp) / static_cast<float>(cumTp + cumFp);
        recall[i] = static_cast<float>(cumTp) / static_cast<float>(numGroundTruth);
    }

    // All-point interpolation: precision at recall r is replaced by the
    // max precision achieved at any recall >= r. A single backward pass
    // (from highest rank to lowest) computing a running max does this in
    // one linear sweep.
    for (int i = static_cast<int>(precision.size()) - 2; i >= 0; --i) {
        precision[i] = std::max(precision[i], precision[i + 1]);
    }

    // Integrate: AP = sum over prediction ranks of (recall[i] - recall[i-1]) * precision[i],
    // i.e. the area under the (now monotonically non-increasing) interpolated
    // precision-vs-recall step function.
    float ap = 0.0f;
    float prevRecall = 0.0f;
    for (size_t i = 0; i < preds.size(); ++i) {
        ap += (recall[i] - prevRecall) * precision[i];
        prevRecall = recall[i];
    }
    return ap;
}

MeanAveragePrecisionResult meanAveragePrecision(const std::vector<Detection>& predictions,
                                                 const std::vector<GroundTruth>& groundTruths,
                                                 int numClasses, float iouThreshold) {
    MeanAveragePrecisionResult result;
    result.perClassAP.resize(numClasses);

    float sum = 0.0f;
    int count = 0;
    for (int c = 0; c < numClasses; ++c) {
        auto ap = averagePrecision(predictions, groundTruths, c, iouThreshold);
        result.perClassAP[c] = ap;
        if (ap.has_value()) {
            sum += *ap;
            ++count;
        }
    }
    result.mAP = (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
    return result;
}

}  // namespace edgeguard
