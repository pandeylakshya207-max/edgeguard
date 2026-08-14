#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <fstream>
#include <cstdio>

#include "edgeguard/labels.hpp"

using edgeguard::BoundingBox;
using edgeguard::parseYoloLabelLine;
using edgeguard::parseYoloLabelFile;
using edgeguard::classNames;
using Catch::Approx;

TEST_CASE("classNames has exactly the 10 dataset classes in the correct index order", "[labels]") {
    const auto& names = classNames();
    REQUIRE(names.size() == 10);
    REQUIRE(names[0] == "Hardhat");
    REQUIRE(names[5] == "Person");
    REQUIRE(names[9] == "vehicle");
}

TEST_CASE("parseYoloLabelLine converts normalized center-format to absolute corner-format", "[labels]") {
    // classId=5 (Person), center at (0.5, 0.5), width=0.2, height=0.4,
    // in a 1000x1000 image -> box centered at (500,500), 200 wide, 400 tall
    // -> corners (400,300)-(600,700).
    auto box = parseYoloLabelLine("5 0.5 0.5 0.2 0.4", 1000, 1000);
    REQUIRE(box.has_value());
    REQUIRE(box->classId == 5);
    REQUIRE(box->x1 == Approx(400.0f));
    REQUIRE(box->y1 == Approx(300.0f));
    REQUIRE(box->x2 == Approx(600.0f));
    REQUIRE(box->y2 == Approx(700.0f));
    REQUIRE(box->confidence == Approx(1.0f));
}

TEST_CASE("parseYoloLabelLine rejects malformed lines instead of guessing", "[labels]") {
    REQUIRE_FALSE(parseYoloLabelLine("", 100, 100).has_value());
    REQUIRE_FALSE(parseYoloLabelLine("not a valid line", 100, 100).has_value());
    REQUIRE_FALSE(parseYoloLabelLine("5 0.5 0.5 0.2", 100, 100).has_value());  // missing field
    REQUIRE_FALSE(parseYoloLabelLine("5 0.5 0.5 0.2 0.4 extra", 100, 100).has_value());  // trailing junk
}

TEST_CASE("parseYoloLabelFile on a missing file returns empty, not an error", "[labels]") {
    auto boxes = parseYoloLabelFile("/tmp/definitely-does-not-exist-edgeguard.txt", 640, 640);
    REQUIRE(boxes.empty());
}

TEST_CASE("parseYoloLabelFile parses multiple lines and skips a malformed one without corrupting the rest", "[labels]") {
    const std::string path = "/tmp/edgeguard_test_labels.txt";
    {
        std::ofstream f(path);
        f << "0 0.1 0.1 0.1 0.1\n";
        f << "\n";  // blank line — must be skipped silently
        f << "this line is garbage\n";  // malformed — must be skipped, not abort the file
        f << "1 0.9 0.9 0.1 0.1\n";
    }
    auto boxes = parseYoloLabelFile(path, 100, 100);
    std::remove(path.c_str());

    REQUIRE(boxes.size() == 2);  // exactly the two well-formed lines
    REQUIRE(boxes[0].classId == 0);
    REQUIRE(boxes[1].classId == 1);
}
