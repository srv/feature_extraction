#include <ros/ros.h>
#include <opencv2/core/core.hpp>

#include <boost/function.hpp>

#include "feature_extraction/SetRegionOfInterest.h"


namespace feature_extraction_ros
{

/**
* \brief class that offers a region of interest server with
*        a getter for the actual value
*/
class RegionOfInterestServer
{

public:

  typedef boost::function<void (const cv::Rect&)> RoiSetterCallback;

  /**
  * Advertizes the service
  */
  RegionOfInterestServer(ros::NodeHandle& nh, RoiSetterCallback cb);

private:

  /**
  * Service callback
  */
  bool setRegionOfInterest(feature_extraction::SetRegionOfInterest::Request& req,
                           feature_extraction::SetRegionOfInterest::Response& res);

  RoiSetterCallback cb_;

  ros::ServiceServer service_;

};

}

