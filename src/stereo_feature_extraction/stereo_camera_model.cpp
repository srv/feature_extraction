
#include "stereo_camera_model.h"

using namespace stereo_feature_extraction;

StereoCameraModel::StereoCameraModel()
{}

cv::Point3d StereoCameraModel::computeWorldPoint(
        const cv::Point2d& left_uv_rect,
        const cv::Point2d& right_uv_rect) const
{
    cv::Point3d point(0, 0, 0);
    return point;
}

