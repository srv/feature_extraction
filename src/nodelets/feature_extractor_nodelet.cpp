#include <nodelet/nodelet.h>
#include <ros/ros.h>

#include <image_transport/image_transport.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/RegionOfInterest.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vision_msgs/Features.h>

#include "feature_extraction/key_point_detector_factory.h"
#include "feature_extraction/descriptor_extractor_factory.h"
#include "feature_extraction/key_points_filter.h"

#include "feature_extraction_ros/conversions.h"

namespace feature_extraction_ros
{
class FeatureExtractorNodelet : public nodelet::Nodelet
{
public:
  FeatureExtractorNodelet() : show_debug_image_(false)
  {
  }

private:
  virtual void onInit()
  {
    ros::NodeHandle nh = getNodeHandle();
    ros::NodeHandle& private_nh = getPrivateNodeHandle();

    it_.reset(new image_transport::ImageTransport(getNodeHandle()));

    // TODO subscribe / unsubscribe on demand?
    sub_image_ = it_->subscribe("image", 1, &FeatureExtractorNodelet::imageCb, this);
    // TODO change this to service/dynamic reconfigure
    sub_region_of_interest_ = private_nh.subscribe("region_of_interest", 10, &FeatureExtractorNodelet::setRegionOfInterest, this);

    pub_features_ = private_nh.advertise<vision_msgs::Features>("features", 1);

    private_nh.param("show_image", show_debug_image_, false);

    private_nh.param("max_num_key_points", max_num_key_points_, 5000);

    std::string key_point_detector_name;
    private_nh.param("key_point_detector", key_point_detector_name,
                     std::string("SmartSURF"));
    key_point_detector_ =
      feature_extraction::KeyPointDetectorFactory::create(
        key_point_detector_name);

    std::string descriptor_extractor_name;
    private_nh.param("descriptor_extractor", descriptor_extractor_name,
                     std::string("SmartSURF"));
    descriptor_extractor_ =
      feature_extraction::DescriptorExtractorFactory::create(
        descriptor_extractor_name);

    NODELET_INFO_STREAM("Parameters: \n"
                        " key point detector   = " << key_point_detector_name << "\n"
                        " descriptor extractor = " << descriptor_extractor_name << "\n"
                        " max num key points   = " << max_num_key_points_ << "\n"
                        " show image           = " << (show_debug_image_ ? "yes" : "no"));

    window_name_ = key_point_detector_name + "/" +
                   descriptor_extractor_name + " features for " + nh.resolveName("image");

    cv::namedWindow(window_name_, 0);
  }


  void imageCb(const sensor_msgs::ImageConstPtr& image_msg)
  {
    if (region_of_interest_.area() == 0)
    {
      // invalid roi, set to image size
      region_of_interest_ =
        cv::Rect(0, 0, image_msg->width, image_msg->height);
    }
    try
    {
      if (sensor_msgs::image_encodings::isBayer(image_msg->encoding))
      {
        NODELET_ERROR("Feature extraction called with bayer encoded image, skipping!");
        return;
      }
      // bridge to opencv
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(image_msg);

      // select roi
      cv::Mat roi(cv_ptr->image(region_of_interest_));

      // Calculate features
      std::vector<cv::KeyPoint> key_points;
      key_point_detector_->detect(roi, key_points);
      size_t num_before_filter = key_points.size();
      feature_extraction::KeyPointsFilter::filterBest(key_points, 
                                                      max_num_key_points_);
      size_t num_after_filter = key_points.size();
      cv::Mat descriptors;
      descriptor_extractor_->extract(roi, key_points, descriptors);

      NODELET_INFO("%zu key points detected, %zu survived filter, "
                   "%zu descriptors extracted.", num_before_filter,
                   num_after_filter, key_points.size());

      vision_msgs::Features::Ptr features_msg(new vision_msgs::Features());
      features_msg->header = image_msg->header;
      feature_extraction_ros::toMsg(key_points, descriptors, *features_msg);
      pub_features_.publish(features_msg);

      if (show_debug_image_)
      {
        cv::Mat canvas;
        if (cv_ptr->image.channels() == 1)
        {
          cv::cvtColor(cv_ptr->image, canvas, CV_GRAY2BGR);
        }
        else
        {
          canvas = cv_ptr->image.clone();
        }
        cv::drawKeypoints(canvas, key_points, canvas, cv::Scalar::all(-1), 
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS | 
                          cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
        cv::rectangle(canvas, region_of_interest_.tl(),
                      region_of_interest_.br(), cv::Scalar(0, 0, 255), 3);
        cv::imshow(window_name_, canvas);
        cv::waitKey(50);
      }
    }
    catch (cv_bridge::Exception& e)
    {
      NODELET_ERROR("cv_bridge exception: %s", e.what());
    }
  }

  void setRegionOfInterest(const sensor_msgs::RegionOfInterestConstPtr& roi_msg)
  {
    region_of_interest_.x = roi_msg->x_offset;
    region_of_interest_.y = roi_msg->y_offset;
    region_of_interest_.width = roi_msg->width;
    region_of_interest_.height = roi_msg->height;
    NODELET_INFO("Region of interest set to: (%i, %i), %ix%i",
                 region_of_interest_.x,
                 region_of_interest_.y,
                 region_of_interest_.width,
                 region_of_interest_.height);
  }

  ros::Subscriber sub_region_of_interest_;
  boost::shared_ptr<image_transport::ImageTransport> it_;
  image_transport::Subscriber sub_image_;

  ros::Publisher pub_features_;

  feature_extraction::KeyPointDetector::Ptr key_point_detector_;
  feature_extraction::DescriptorExtractor::Ptr descriptor_extractor_;
  cv::Rect region_of_interest_;
  int max_num_key_points_;
  bool show_debug_image_;
  std::string window_name_;

};

} // end of namespace

#include <pluginlib/class_list_macros.h>
PLUGINLIB_DECLARE_CLASS(feature_extraction,
                        FeatureExtractor,
                        feature_extraction_ros::FeatureExtractorNodelet, nodelet::Nodelet);

