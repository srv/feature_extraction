#ifndef STEREO_FEATURE_H
#define STEREO_FEATURE_H

#include <iostream>
#include <opencv2/core/core.hpp>

#include <feature_extraction/key_point.h>

namespace stereo_feature_extraction
{

/**
* Struct to hold the output of the stereo feature extractor
*/
struct StereoFeature
{   
    cv::Point3d world_point;
    feature_extraction::KeyPoint key_point_left;
    feature_extraction::KeyPoint key_point_right;
    cv::Mat descriptor;
    cv::Vec3b color;
};


}

std::ostream& operator<<(std::ostream& ostream,
        const stereo_feature_extraction::StereoFeature& feature);

#endif


