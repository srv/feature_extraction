
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace feature_extraction
{

namespace features_io
{

void saveStereoFeatures(const std::string& filename,
    const std::vector<cv::KeyPoint>& key_points_left,
    const std::vector<cv::KeyPoint>& key_points_right,
    const cv::Mat& descriptors,
    const std::vector<cv::Point3d>& points3d);

void loadStereoFeatures(const std::string& filename, 
    std::vector<cv::KeyPoint>& key_points_left,
    std::vector<cv::KeyPoint>& key_points_right,
    cv::Mat& descriptors,
    std::vector<cv::Point3d>& points3d);

}

}

