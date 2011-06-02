#ifndef STEREO_CAMERA_MODEL_H
#define STEREO_CAMERA_MODEL_H

#include <boost/shared_ptr.hpp>
#include <image_geometry/stereo_camera_model.h>

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

