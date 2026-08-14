#pragma once

#include <optional>
#include <string>
#include <vector>

#include "edgeguard/geometry.hpp"

namespace edgeguard {

// The 10 classes of the Construction Site Safety dataset this project
// targets, in the exact index order the dataset's YOLO-format labels use
// — classId in a label file is an index into this list, not a name.
inline const std::vector<std::string>& classNames() {
    static const std::vector<std::string> names = {
        "Hardhat",   "Mask",          "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
        "Person",    "Safety Cone",   "Safety Vest", "machinery", "vehicle",
    };
    return names;
}

// parseYoloLabelLine parses one line of a YOLO-format annotation:
//   "<classId> <centerX> <centerY> <width> <height>"
// where all four numeric fields are normalized to [0,1] relative to
// image dimensions (YOLO's native label format — NOT the same
// convention as our canonical BoundingBox, which is absolute pixels;
// this function does that conversion using the supplied image size).
//
// Returns std::nullopt for a malformed line (wrong field count,
// unparsable numbers) rather than throwing or silently returning a
// zeroed box — a caller reading a real annotation file, which may have
// a stray blank line or a hand-edited typo, needs to be able to tell
// "no box here" apart from "a legitimate box at the origin".
std::optional<BoundingBox> parseYoloLabelLine(const std::string& line, int imageWidth,
                                               int imageHeight);

// parseYoloLabelFile reads every line of a YOLO-format .txt annotation
// file and returns the successfully-parsed boxes, silently skipping
// blank lines and lines that fail to parse (see parseYoloLabelLine).
// Returns an empty vector (not an error) for a missing file — a missing
// label file for an image conventionally means "no objects annotated in
// this image", which is different from a parse failure but the same
// practical result for a caller building up a ground-truth set.
std::vector<BoundingBox> parseYoloLabelFile(const std::string& path, int imageWidth,
                                             int imageHeight);

}  // namespace edgeguard
