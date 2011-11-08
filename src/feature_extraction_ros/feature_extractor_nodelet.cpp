#include <nodelet/nodelet.h>
#include <ros/ros.h>

#include <image_transport/image_transport.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/RegionOfInterest.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include <vision_msgs/Features.h>

#include "feature_extraction/feature_extractor_factory.h"
#include "feature_extraction/drawing.h"

namespace feature_extraction_ros
{
class FeatureExtractorNodelet : public nodelet::Nodelet
{
  public:
    FeatureExtractorNodelet() : show_image_(false)
    { }

  private:
    virtual void onInit()
    {
        ros::NodeHandle nh = getNodeHandle();
        ros::NodeHandle& private_nh = getPrivateNodeHandle();
        
        it_.reset(new image_transport::ImageTransport(getNodeHandle()));

        // TODO subscribe / unsubscribe on demand?
        sub_image_ = it_->subscribe("image", 1, &FeatureExtractorNodelet::imageCb, this);
        sub_region_of_interest_ = nh.subscribe("region_of_interest", 10, &FeatureExtractorNodelet::setRegionOfInterest, this);
        
        pub_features_ = private_nh.advertise<vision_msgs::Features>("features", 1);

        private_nh.param("show_image", show_image_, false);
        int max_num_key_points;
        private_nh.param("max_num_key_points", max_num_key_points, 5000);
        std::string feature_extractor_name;
        private_nh.param("feature_extractor", feature_extractor_name, 
                std::string("SURF"));
        feature_extraction::FeatureExtractor::Ptr feature_extractor = 
            feature_extraction::FeatureExtractorFactory::create(feature_extractor_name);
        if (feature_extractor.get() == NULL)
        {
            NODELET_FATAL("Cannot create feature extractor with name %s",
                    feature_extractor_name.c_str());
        }
        else
        {
            feature_extractor->setMaxNumKeyPoints(max_num_key_points);
        }

        NODELET_INFO_STREAM("Parameters: " 
                  << " feature_extractor = " << feature_extractor_name
                  << " max_num_key_points = " << max_num_key_points
                  << " show image = " << (show_image_ ? "yes" : "no"));
        feature_extractor_ = feature_extractor;

        window_name_ = feature_extractor_name + " features";
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
            namespace enc = sensor_msgs::image_encodings;
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(image_msg);
            
            // Calculate features
            std::vector<feature_extraction::KeyPoint> key_points;
            cv::Mat descriptors;
            feature_extractor_->extract(cv_ptr->image, key_points, descriptors, region_of_interest_);

            NODELET_DEBUG("%zu features extracted.", key_points.size());
            if (key_points.size() == 0)
            {
                return;
            }

            if (pub_features_.getNumSubscribers() > 0)
            {
              vision_msgs::Features::Ptr features_msg(new vision_msgs::Features());
              features_msg->header = image_msg->header;
              features_msg->key_points.resize(key_points.size());
              for (size_t i = 0; i < key_points.size(); ++i)
              {
                features_msg->key_points[i].x = key_points[i].pt.x;
                features_msg->key_points[i].y = key_points[i].pt.y;
                features_msg->key_points[i].size = key_points[i].size;
                features_msg->key_points[i].angle = key_points[i].angle;
                features_msg->key_points[i].response = key_points[i].response;
                features_msg->key_points[i].octave = key_points[i].octave;
              }
              assert(descriptors.isContinuous());
              assert(descriptors.depth() == CV_32F);
              assert(descriptors.channels() == 1);
              features_msg->descriptor_data.insert(features_msg->descriptor_data.begin(), descriptors.data,
                  descriptors.data + descriptors.cols * descriptors.rows);
              pub_features_.publish(features_msg);
            }

            if (show_image_)
            {
                cv::Mat canvas = cv_ptr->image.clone();
                drawKeyPoints(canvas, key_points);
                cv::rectangle(canvas, region_of_interest_.tl(),
                        region_of_interest_.br(), cv::Scalar(0, 0, 255), 3);
                cv::imshow(window_name_, canvas);
                cv::waitKey(5);
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

    // Publications
    ros::Publisher pub_features_;

    // the extractor
    feature_extraction::FeatureExtractor::Ptr feature_extractor_;

    // the current roi
    cv::Rect region_of_interest_;

    // show debug image
    bool show_image_;

    // title of the opencv window
    std::string window_name_;
 
};

} // end of namespace

#include <pluginlib/class_list_macros.h>
PLUGINLIB_DECLARE_CLASS(feature_extraction, 
    FeatureExtractor, 
    feature_extraction_ros::FeatureExtractorNodelet, nodelet::Nodelet);

