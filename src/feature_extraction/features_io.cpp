#include <stdexcept>
#include <cassert>
#include "feature_extraction/features_io.h"


void feature_extraction::features_io::saveStereoFeatures(
    const std::string& filename,
    const std::vector<cv::KeyPoint>& key_points_left,
    const std::vector<cv::KeyPoint>& key_points_right,
    const cv::Mat& descriptors,
    const std::vector<cv::Point3d>& points3d)
{
  assert(key_points_left.size() == key_points_right.size());
  assert(key_points_left.size() == points3d.size());
  assert(static_cast<int>(key_points_left.size()) == descriptors.rows);
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  fs << "version" << 1;
  cv::write(fs, "key_points_left", key_points_left);
  cv::write(fs, "key_points_right", key_points_right);
  fs << "descriptors" << descriptors;
  fs << "points3d" << "[:" << points3d << "]";
}

void feature_extraction::features_io::loadStereoFeatures(
    const std::string& filename,
    std::vector<cv::KeyPoint>& key_points_left,
    std::vector<cv::KeyPoint>& key_points_right,
    cv::Mat& descriptors,
    std::vector<cv::Point3d>& points3d)
{
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  int version;
  fs["version"] >> version;
  if (version == 1)
  {
    cv::read(fs["key_points_left"], key_points_left);
    cv::read(fs["key_points_right"], key_points_right);
    fs["descriptors"] >> descriptors;
    fs["points3d"] >> points3d;
  }
  else
  {
    throw std::runtime_error("unsupported file format");
  }
}

