#include <stdexcept>
#include <cassert>
#include "feature_extraction/features_io.h"


void feature_extraction::features_io::saveStereoFeatures(
    const std::string& filename,
    const std::vector<cv::KeyPoint>& key_points,
    const cv::Mat& descriptors,
    const std::vector<cv::Point3d>& world_points)
{
  assert(key_points.size() == world_points.size());
  assert(static_cast<int>(key_points.size()) == descriptors.rows);
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  fs << "version" << 1;
  cv::write(fs, "key_points", key_points);
  fs << "descriptors" << descriptors;
  fs << "world_points" << "[:" << world_points << "]";
}

void feature_extraction::features_io::loadStereoFeatures(
    const std::string& filename,
    std::vector<cv::KeyPoint>& key_points,
    cv::Mat& descriptors,
    std::vector<cv::Point3d>& world_points)
{
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  int version;
  fs["version"] >> version;
  if (version == 1)
  {
    cv::read(fs["key_points"], key_points);
    fs["descriptors"] >> descriptors;
    fs["world_points"] >> world_points;
  }
  else
  {
    throw std::runtime_error("unsupported file format");
  }
}

