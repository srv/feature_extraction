
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace feature_extraction
{

namespace features_io
{

void saveStereoFeatures(const std::string& filename,
    const std::vector<cv::KeyPoint>& key_points,
    const cv::Mat& descriptors,
    const std::vector<cv::Point3d>& world_points);

void loadStereoFeatures(const std::string& filename, 
    std::vector<cv::KeyPoint>& key_points,
    cv::Mat& descriptors,
    std::vector<cv::Point3d>& world_points);

}

}

