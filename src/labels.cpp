#include "edgeguard/labels.hpp"

#include <fstream>
#include <sstream>

namespace edgeguard {

std::optional<BoundingBox> parseYoloLabelLine(const std::string& line, int imageWidth,
                                               int imageHeight) {
    std::istringstream iss(line);
    int classId;
    float cx, cy, w, h;
    if (!(iss >> classId >> cx >> cy >> w >> h)) {
        return std::nullopt;  // wrong field count or unparsable number
    }
    // A well-formed line has exactly these five fields and nothing more
    // meaningful after them (trailing whitespace is fine; trailing junk
    // suggests a corrupt line we shouldn't silently half-parse).
    std::string extra;
    if (iss >> extra) {
        return std::nullopt;
    }

    BoundingBox box;
    box.classId = classId;
    box.confidence = 1.0f;  // ground-truth boxes are certain by definition
    box.x1 = (cx - w / 2.0f) * imageWidth;
    box.y1 = (cy - h / 2.0f) * imageHeight;
    box.x2 = (cx + w / 2.0f) * imageWidth;
    box.y2 = (cy + h / 2.0f) * imageHeight;
    return box;
}

std::vector<BoundingBox> parseYoloLabelFile(const std::string& path, int imageWidth,
                                             int imageHeight) {
    std::vector<BoundingBox> boxes;
    std::ifstream file(path);
    if (!file.is_open()) {
        return boxes;  // missing label file == no annotated objects, not an error
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;  // blank line
        }
        auto box = parseYoloLabelLine(line, imageWidth, imageHeight);
        if (box.has_value()) {
            boxes.push_back(*box);
        }
        // Malformed lines are silently skipped rather than aborting the
        // whole file — a training/eval run over thousands of label files
        // shouldn't crash because of one bad line, but see labels_test
        // for a case verifying skip vs partial-parse doesn't corrupt the
        // rest of the boxes in that file.
    }
    return boxes;
}

}  // namespace edgeguard
