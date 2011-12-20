#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <image_geometry/stereo_camera_model.h>
#include <sensor_msgs/image_encodings.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/Features3D.h>

#include "feature_matching/stereo_feature_matcher.h"
#include "feature_matching/stereo_depth_estimator.h"
#include "feature_extraction/key_point_detector_factory.h"
#include "feature_extraction/descriptor_extractor_factory.h"

#include "feature_extraction_ros/conversions.h"
#include "feature_extraction_ros/region_of_interest_server.h"

#include <opencv2/highgui/highgui.hpp>

namespace feature_extraction_ros
{
class StereoFeatureExtractorNodelet : public nodelet::Nodelet
{
public:

  StereoFeatureExtractorNodelet() {
  }

  typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

  virtual void onInit()
  {
    ros::NodeHandle nh = getNodeHandle();
    ros::NodeHandle& nh_private = getPrivateNodeHandle();
    subscribed_ = false;     // no subscription yet

    ros::SubscriberStatusCallback connect_cb =
      boost::bind(&StereoFeatureExtractorNodelet::connectCb, this);

    pub_point_cloud_ = nh_private.advertise<PointCloud>("point_cloud", 1,
                                                        connect_cb, connect_cb);

    pub_features_3d_ = nh_private.advertise<vision_msgs::Features3D>("features_3d", 1,
                                                                     connect_cb, connect_cb);
    region_of_interest_server_.reset(new RegionOfInterestServer(
          nh_private, boost::bind(
            &StereoFeatureExtractorNodelet::setRegionOfInterest, this, _1)));

    nh_private.param("show_image", show_debug_image_, false);

    int queue_size;
    nh_private.param("queue_size", queue_size, 5);

    nh_private.param("max_y_diff", max_y_diff_, 2.0);
    nh_private.param("max_angle_diff", max_angle_diff_, 2.0);
    nh_private.param("max_size_diff", max_size_diff_, 2);
    nh_private.param("matching_threshold", matching_threshold_, 0.8);

    std::string key_point_detector_name;
    nh_private.param("key_point_detector", key_point_detector_name,
                     std::string("SmartSURF"));
    key_point_detector_ =
      feature_extraction::KeyPointDetectorFactory::create(
        key_point_detector_name);

    std::string descriptor_extractor_name;
    nh_private.param("descriptor_extractor", descriptor_extractor_name,
                     std::string("SmartSURF"));
    descriptor_extractor_ =
      feature_extraction::DescriptorExtractorFactory::create(
        descriptor_extractor_name);

    NODELET_INFO_STREAM(
      "Parameters: \n"
      " key point detector   = " << key_point_detector_name << "\n"
      " descriptor extractor = " << descriptor_extractor_name << "\n"
      " max_y_diff           = " << max_y_diff_  << "\n"
      " max_angle_diff       = " << max_angle_diff_ << "\n"
      " max_size_diff        = " << max_size_diff_ << "\n"
      " matching_threshold   = " << matching_threshold_ << "\n"
      " show image           = " << (show_debug_image_ ? "yes" : "no"));

    // Synchronize inputs. Topic subscriptions happen on demand in the
    // connection callback.
    exact_sync_.reset(new ExactSync(ExactPolicy(queue_size), sub_l_image_, sub_r_image_, sub_l_info_, sub_r_info_));
    exact_sync_->registerCallback(boost::bind(&StereoFeatureExtractorNodelet::imageCb, this, _1, _2, _3, _4));

    window_name_ = key_point_detector_name + "/" +
                   descriptor_extractor_name + " stereo features for " + 
                   nh.resolveName("image");

    cv::namedWindow(window_name_, 0);

    NODELET_INFO("Waiting for client subscriptions.");
  }

  // Handles (un)subscribing when clients (un)subscribe
  void connectCb()
  {
    if (pub_features_3d_.getNumSubscribers() == 0 &&
        pub_point_cloud_.getNumSubscribers() == 0)
    {
      NODELET_INFO("No more clients connected, unsubscribing from inputs.");
      sub_l_image_.unsubscribe();
      sub_r_image_.unsubscribe();
      sub_l_info_.unsubscribe();
      sub_r_info_.unsubscribe();
      subscribed_ = false;
    }
    else if (!subscribed_)
    {
      NODELET_INFO("Client connected, subscribing to inputs.");
      ros::NodeHandle nh = getNodeHandle();
      // Queue size 1 should be OK; the one that matters is the synchronizer queue size.
      sub_l_image_.subscribe(nh, "image_left", 1);
      sub_r_image_.subscribe(nh, "image_right", 1);
      sub_l_info_.subscribe(nh, "camera_info_left", 1);
      sub_r_info_.subscribe(nh, "camera_info_right", 1);
      subscribed_ = true;
    }
  }

  void imageCb(const sensor_msgs::ImageConstPtr& l_image_msg,
               const sensor_msgs::ImageConstPtr& r_image_msg,
               const sensor_msgs::CameraInfoConstPtr& l_info_msg,
               const sensor_msgs::CameraInfoConstPtr& r_info_msg)
  {
    ros::WallTime start_time = ros::WallTime::now();

    if (region_of_interest_.area() == 0)
    {
      // invalid roi, set to image size
      region_of_interest_ =
        cv::Rect(0, 0, l_image_msg->width, l_image_msg->height);
    }
    try
    {
      if (sensor_msgs::image_encodings::isBayer(l_image_msg->encoding) ||
          sensor_msgs::image_encodings::isBayer(r_image_msg->encoding))
      {
        NODELET_ERROR("Stereo feature extraction called with bayer encoded image, skipping!");
        return;
      }
      // bridge to opencv
      cv_bridge::CvImageConstPtr l_cv_ptr = cv_bridge::toCvShare(l_image_msg);
      cv_bridge::CvImageConstPtr r_cv_ptr = cv_bridge::toCvShare(r_image_msg);

      // compute roi in right image
      cv::Rect region_of_interest_right(0, region_of_interest_.y, 
                         region_of_interest_.x + region_of_interest_.width,
                         region_of_interest_.height);
      // select rois
      cv::Mat image_left(l_cv_ptr->image(region_of_interest_));
      cv::Mat image_right(r_cv_ptr->image(region_of_interest_right));

      // extract key points and descriptors
      std::vector<cv::KeyPoint> key_points_left;
      key_point_detector_->detect(image_left, key_points_left);
      cv::Mat descriptors_left;
      descriptor_extractor_->extract(image_left, key_points_left, descriptors_left);

      std::vector<cv::KeyPoint> key_points_right;
      key_point_detector_->detect(image_right, key_points_right);
      cv::Mat descriptors_right;
      descriptor_extractor_->extract(image_right, key_points_right, descriptors_right);

      // shift key point positions according to roi
      if (region_of_interest_.x != 0 || region_of_interest_.y != 0)
      {
        for (size_t i = 0; i < key_points_left.size(); ++i)
        {
          key_points_left[i].pt.x += region_of_interest_.x;
          key_points_left[i].pt.y += region_of_interest_.y;
        }
        for (size_t i = 0; i < key_points_right.size(); ++i)
        {
          key_points_right[i].pt.x += region_of_interest_right.x;
          key_points_right[i].pt.y += region_of_interest_right.y;
        }
      }

      // configure and perform matching
      feature_matching::StereoFeatureMatcher::Params params;
      params.max_y_diff = max_y_diff_;
      params.max_angle_diff = max_angle_diff_;
      params.max_size_diff = max_size_diff_;

      feature_matching::StereoFeatureMatcher matcher;
      matcher.setParams(params);
      std::vector<cv::DMatch> matches;
      matcher.match(key_points_left, descriptors_left, key_points_right,
                    descriptors_right, matching_threshold_, matches);

      // calculate 3D world points
      feature_matching::StereoDepthEstimator depth_estimator;
      depth_estimator.setCameraInfo(*l_info_msg, *r_info_msg);
      std::vector<cv::Point3d> world_points(matches.size());
      std::vector<cv::KeyPoint> matched_key_points_left(matches.size());
      cv::Mat matched_descriptors_left(matches.size(), descriptors_left.cols, descriptors_left.type());
      for (size_t i = 0; i < matches.size(); ++i)
      {
        depth_estimator.calculate3DPoint(key_points_left[matches[i].queryIdx].pt,
                                        key_points_right[matches[i].trainIdx].pt,
                                        world_points[i]);
        matched_key_points_left[i] = key_points_left[matches[i].queryIdx];
        cv::Mat descriptor_src = descriptors_left.row(matches[i].queryIdx);
        cv::Mat descriptor_tgt = matched_descriptors_left.row(i);
        descriptor_src.copyTo(descriptor_tgt);
      }

      // convert to msg format again
      PointCloud::Ptr point_cloud(new PointCloud());
      feature_extraction_ros::toMsg(world_points, *point_cloud);
      point_cloud->header = l_info_msg->header;

      vision_msgs::Features3D::Ptr features_3d_msg(new vision_msgs::Features3D());
      feature_extraction_ros::toMsg(matched_key_points_left, matched_descriptors_left,
                                    world_points, *features_3d_msg);
      features_3d_msg->header = l_info_msg->header;

      pub_point_cloud_.publish(point_cloud);
      pub_features_3d_.publish(features_3d_msg);

      ros::WallTime end_time = ros::WallTime::now();
      NODELET_INFO("%zu left, %zu right features, %zu matches, %f sec.", 
          key_points_left.size(), key_points_right.size(), matches.size(), 
          (end_time - start_time).toSec());

      if (show_debug_image_)
      {
        cv::Mat canvas;
        cv::drawMatches(l_cv_ptr->image, key_points_left, 
                        r_cv_ptr->image, key_points_right, matches, canvas);
        cv::rectangle(canvas, region_of_interest_.tl(),
                      region_of_interest_.br(), cv::Scalar(0, 0, 255), 3);
        cv::Point r_shift(l_image_msg->width, 0);
        cv::rectangle(canvas, region_of_interest_right.tl() + r_shift,
                      region_of_interest_right.br() + r_shift,
                      cv::Scalar(0, 0, 255), 3);
        cv::imshow(window_name_, canvas);
        cv::waitKey(50);
      }
    }
    catch (cv_bridge::Exception& e)
    {
      NODELET_ERROR("cv_bridge exception: %s", e.what());
    }
  }

  // called by region of interest server
  void setRegionOfInterest(const cv::Rect& roi)
  {
    region_of_interest_ = roi;
    NODELET_INFO("Region of interest set to: (%i, %i), %ix%i",
                 region_of_interest_.x,
                 region_of_interest_.y,
                 region_of_interest_.width,
                 region_of_interest_.height);
  }

private:

  // Subscriptions
  message_filters::Subscriber<sensor_msgs::Image> sub_l_image_, sub_r_image_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> sub_l_info_, sub_r_info_;
  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image,
                                                    sensor_msgs::Image, 
                                                    sensor_msgs::CameraInfo, 
                                                    sensor_msgs::CameraInfo> ExactPolicy;
  typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
  boost::shared_ptr<ExactSync> exact_sync_;
  bool subscribed_;   // stores if anyone is subscribed
  ros::Subscriber sub_region_of_interest_;

  // Publications
  ros::Publisher pub_point_cloud_;
  ros::Publisher pub_features_3d_;

  // detector & extractor
  feature_extraction::KeyPointDetector::Ptr key_point_detector_;
  feature_extraction::DescriptorExtractor::Ptr descriptor_extractor_;

  boost::shared_ptr<RegionOfInterestServer> region_of_interest_server_;
  cv::Rect region_of_interest_;

  // the stereo matching constraints
  double max_y_diff_, max_angle_diff_;
  int max_size_diff_;

  // threshold for matching
  double matching_threshold_;

  bool show_debug_image_;
  std::string window_name_;
};

} // end of namespace


#include <pluginlib/class_list_macros.h>
PLUGINLIB_DECLARE_CLASS(feature_extraction,
                        StereoFeatureExtractor,
                        feature_extraction_ros::StereoFeatureExtractorNodelet, nodelet::Nodelet);

