#ifndef DRAWING_H
#define DRAWING_H

#include <vector>

namespace cv
{
    class Mat;
}

namespace stereo_feature_extraction
{
    class KeyPoint;
    class StereoFeature;

    void drawKeyPoint(cv::Mat& image, const KeyPoint& key_point);

    void drawStereoFeatures(cv::Mat& canvas, const cv::Mat& left_image,
            const cv::Mat& right_image,
            const std::vector<StereoFeature>& stereo_features);
}


#endif
