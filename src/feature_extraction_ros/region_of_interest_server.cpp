#include "feature_extraction_ros/region_of_interest_server.h"

feature_extraction_ros::RegionOfInterestServer::RegionOfInterestServer(ros::NodeHandle& nh,
    RoiSetterCallback cb) : cb_(cb)
{
  service_ = nh.advertiseService("set_region_of_interest", 
      &RegionOfInterestServer::setRegionOfInterest, this);
}

bool feature_extraction_ros::RegionOfInterestServer::setRegionOfInterest(
    feature_extraction::SetRegionOfInterest::Request& req, 
    feature_extraction::SetRegionOfInterest::Response& res)
{
  cv::Rect roi;
  roi.x = req.region_of_interest.x_offset;
  roi.y = req.region_of_interest.y_offset;
  roi.width = req.region_of_interest.width;
  roi.height = req.region_of_interest.height;
  // call registered callback
  cb_(roi);
  return true;
}

