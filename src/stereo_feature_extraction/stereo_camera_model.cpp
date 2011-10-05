
#include <camera_calibration_parsers/parse.h>
#include <sensor_msgs/CameraInfo.h>

#include "stereo_feature_extraction/stereo_camera_model.h"

using namespace stereo_feature_extraction;

StereoCameraModel::StereoCameraModel()
{}

void StereoCameraModel::fromCalibrationFiles(
        const std::string& calibration_file_left,
        const std::string& calibration_file_right)
{
    // read calibration data to fill stereo camera model
    sensor_msgs::CameraInfo camera_info_left;
    sensor_msgs::CameraInfo camera_info_right;
    std::string left_name;
    std::string right_name;
    camera_calibration_parsers::readCalibration(
            calibration_file_left, left_name, camera_info_left);
    camera_calibration_parsers::readCalibration(
            calibration_file_right, right_name, camera_info_right);

    model_.fromCameraInfo(camera_info_left, camera_info_right);
}


void StereoCameraModel::fromCameraInfo(
        const sensor_msgs::CameraInfo& camera_info_left,
        const sensor_msgs::CameraInfo& camera_info_right)
{
    model_.fromCameraInfo(camera_info_left, camera_info_right);
}

cv::Point3d StereoCameraModel::computeWorldPoint(
        const cv::Point2d& left_uv_rect,
        const cv::Point2d& right_uv_rect) const
{
    cv::Point3d point;
    double disparity = left_uv_rect.x - right_uv_rect.x;
    model_.projectDisparityTo3d(left_uv_rect, disparity, point);
    return point;
}

