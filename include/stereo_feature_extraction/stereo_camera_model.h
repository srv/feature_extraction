#ifndef STEREO_CAMERA_MODEL_H
#define STEREO_CAMERA_MODEL_H

#include <boost/shared_ptr.hpp>
#include <image_geometry/stereo_camera_model.h>

namespace sensor_msgs
{
    template <class ContainerAllocator> struct CameraInfo_;
    typedef  ::sensor_msgs::CameraInfo_<std::allocator<void> > CameraInfo;
}

namespace stereo_feature_extraction
{

/**
* \class StereoCameraModel
* \brief Model for a pair of cameras, wrapper for ROS class
*/
class StereoCameraModel
{

  public:

    /**
    * Creates an empty stereo camera model
    */
    StereoCameraModel();

    /**
    * Reads calibration info from given files to initialize the model.
    * The files must be in ini or yaml format.
    * \param calibration_file_left calibration file for left camera
    * \param calibration_file_right calibration file for right camera
    */
    void fromCalibrationFiles(const std::string& calibration_file_left,
            const std::string& calibration_file_right);

    /**
    * Reads camera info messages to initialize the model.
    * \param camera_info_left left camera info
    * \param camera_info_right right camera info
    */
    void fromCameraInfo(const sensor_msgs::CameraInfo& camera_info_left,
            const sensor_msgs::CameraInfo& camera_info_right);

    /**
    * Computes the 3d position of a matching point pair,
    * using the camera model and the disparity of given points
    * \param left_uv_rect point in the left rectified image
    * \param right_uv_rect point in the right rectified image
    * \return the 3d position of the corresponding world point
    */
    cv::Point3d computeWorldPoint(const cv::Point2d& left_uv_rect,
            const cv::Point2d& right_uv_rect) const;

    /**
    * Defines the pointer type
    */
    typedef boost::shared_ptr<StereoCameraModel> Ptr;

  protected:

    image_geometry::StereoCameraModel model_;

};

}

#endif

