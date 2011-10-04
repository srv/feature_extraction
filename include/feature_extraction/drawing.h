#ifndef DRAWING_H
#define DRAWING_H

#include <vector>

namespace cv
{
    class Mat;
}

namespace feature_extraction
{
    class KeyPoint;
    class StereoFeature;

    void drawKeyPoint(cv::Mat& image, const KeyPoint& key_point);

    void drawKeyPoints(cv::Mat& image, const std::vector<KeyPoint>& key_points);

    void drawStereoFeatures(cv::Mat& canvas, const cv::Mat& left_image,
            const cv::Mat& right_image,
            const std::vector<StereoFeature>& stereo_features);
}


#endif
