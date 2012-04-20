#ifndef STEREO_DEPTH_ESTIMATOR_H
#define STEREO_DEPTH_ESTIMATOR_H

#include <opencv2/features2d/features2d.hpp>

#include <image_geometry/stereo_camera_model.h>

namespace feature_matching
{
  /**
  * \brief Depth estimator for matched stereo key points
  */
  class StereoDepthEstimator
  {
    public:

      StereoDepthEstimator() {}

      void setCameraInfo(const sensor_msgs::CameraInfo& l_info,
                         const sensor_msgs::CameraInfo& r_info);

      /** \brief loads info from disk **/
      void loadCameraInfo(const std::string& left_calibration_file,
                          const std::string& right_calibration_file);

      void calculate3DPoint(const cv::Point2d& left_point, 
                            const cv::Point2d& right_point,
                            cv::Point3d& world_point);

    private:

      image_geometry::StereoCameraModel stereo_camera_model_;

  };

}

#endif

