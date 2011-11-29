#include "feature_matching/stereo_depth_estimator.h"

void feature_matching::StereoDepthEstimator::setCameraInfo(
    const sensor_msgs::CameraInfo& l_info,
    const sensor_msgs::CameraInfo& r_info)
{
  stereo_camera_model_.fromCameraInfo(l_info, r_info);
}

void feature_matching::StereoDepthEstimator::calculate3DPoint(
    const cv::Point2d& left_point, const cv::Point2d& right_point, 
    cv::Point3d& world_point)
{
  double disparity = left_point.x - right_point.x;
  stereo_camera_model_.projectDisparityTo3d(left_point, disparity, world_point);
}


