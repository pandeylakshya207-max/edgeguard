#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <opencv2/core.hpp>

#include "edgeguard/letterbox.hpp"

using edgeguard::BoundingBox;
using edgeguard::letterbox;
using edgeguard::unletterboxBox;
using Catch::Approx;

TEST_CASE("letterbox output is always exactly targetSize x targetSize", "[letterbox]") {
    cv::Mat wide(200, 400, CV_8UC3, cv::Scalar(0, 0, 0));   // landscape
    cv::Mat tall(400, 200, CV_8UC3, cv::Scalar(0, 0, 0));   // portrait
    cv::Mat square(300, 300, CV_8UC3, cv::Scalar(0, 0, 0)); // already square

    for (const cv::Mat& img : {wide, tall, square}) {
        auto result = letterbox(img, 640);
        REQUIRE(result.image.cols == 640);
        REQUIRE(result.image.rows == 640);
    }
}

TEST_CASE("letterbox on a square image has zero padding", "[letterbox]") {
    cv::Mat square(300, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    auto result = letterbox(square, 640);
    REQUIRE(result.padLeft == 0);
    REQUIRE(result.padTop == 0);
    REQUIRE(result.scale == Approx(640.0f / 300.0f));
}

TEST_CASE("letterbox on a wide image pads top/bottom, not left/right", "[letterbox]") {
    cv::Mat wide(200, 400, CV_8UC3, cv::Scalar(0, 0, 0));  // 2:1 aspect
    auto result = letterbox(wide, 640);
    REQUIRE(result.padLeft == 0);   // width fits exactly, no horizontal padding
    REQUIRE(result.padTop > 0);     // height has room, gets padded
}

TEST_CASE("letterbox on a tall image pads left/right, not top/bottom", "[letterbox]") {
    cv::Mat tall(400, 200, CV_8UC3, cv::Scalar(0, 0, 0));  // 1:2 aspect
    auto result = letterbox(tall, 640);
    REQUIRE(result.padTop == 0);
    REQUIRE(result.padLeft > 0);
}

TEST_CASE("unletterboxBox correctly inverts a known transform", "[letterbox]") {
    // 400x200 image (2:1) into a 640 target: scale = 640/400 = 1.6,
    // newW=640, newH=320, padTop=(640-320)/2=160, padLeft=0.
    cv::Mat wide(200, 400, CV_8UC3, cv::Scalar(0, 0, 0));
    auto result = letterbox(wide, 640);
    REQUIRE(result.scale == Approx(1.6f));
    REQUIRE(result.padTop == 160);

    // A box at (160,160)-(320,320) in the letterboxed image should map
    // back to (100,0)-(200,100) in the original 400x200 image.
    BoundingBox letterboxed{160, 160, 320, 320, 0, 0.9f};
    BoundingBox original = unletterboxBox(letterboxed, result);
    REQUIRE(original.x1 == Approx(100.0f));
    REQUIRE(original.y1 == Approx(0.0f));
    REQUIRE(original.x2 == Approx(200.0f));
    REQUIRE(original.y2 == Approx(100.0f));
}

TEST_CASE("letterbox then unletterbox round-trips a full-image box back to the original bounds", "[letterbox]") {
    cv::Mat img(150, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    auto result = letterbox(img, 640);

    BoundingBox fullLetterboxed{
        static_cast<float>(result.padLeft), static_cast<float>(result.padTop),
        static_cast<float>(result.padLeft + img.cols * result.scale),
        static_cast<float>(result.padTop + img.rows * result.scale),
        0, 1.0f};
    BoundingBox back = unletterboxBox(fullLetterboxed, result);
    REQUIRE(back.x1 == Approx(0.0f).margin(0.5));
    REQUIRE(back.y1 == Approx(0.0f).margin(0.5));
    REQUIRE(back.x2 == Approx(300.0f).margin(0.5));
    REQUIRE(back.y2 == Approx(150.0f).margin(0.5));
}
