#include <camera_calibration_parsers/parse.h>
#include "feature_matching/stereo_depth_estimator.h"

void feature_matching::StereoDepthEstimator::setCameraInfo(
    const sensor_msgs::CameraInfo& l_info,
    const sensor_msgs::CameraInfo& r_info)
{
  stereo_camera_model_.fromCameraInfo(l_info, r_info);
}

void feature_matching::StereoDepthEstimator::loadCameraInfo(
    const std::string& left_calibration_file,
    const std::string& right_calibration_file)
{
  std::string left_camera_name, right_camera_name;
  sensor_msgs::CameraInfo l_info, r_info;
  camera_calibration_parsers::readCalibration(left_calibration_file, left_camera_name, l_info);
  camera_calibration_parsers::readCalibration(right_calibration_file, right_camera_name, r_info);
  setCameraInfo(l_info, r_info);
}


void feature_matching::StereoDepthEstimator::calculate3DPoint(
    const cv::Point2d& left_point, const cv::Point2d& right_point, 
    cv::Point3d& world_point)
{
  double disparity = left_point.x - right_point.x;
  stereo_camera_model_.projectDisparityTo3d(left_point, disparity, world_point);
}


