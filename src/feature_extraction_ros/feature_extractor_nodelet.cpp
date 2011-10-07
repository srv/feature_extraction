#include <nodelet/nodelet.h>
#include <ros/ros.h>

#include <image_transport/image_transport.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/RegionOfInterest.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <vision_msgs/Features.h>

#include "feature_extraction/feature_extractor_factory.h"
#include "feature_extraction/drawing.h"

namespace feature_extraction_ros
{

struct Feature2D
{
    float x;
    float y;
    float data[64];
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
}

POINT_CLOUD_REGISTER_POINT_STRUCT (feature_extraction_ros::Feature2D,
                                (float, x, x)
                                (float, y, y)
                                (float[64], data, data)
                                )

namespace feature_extraction_ros
{
class FeatureExtractorNodelet : public nodelet::Nodelet
{
  public:
    FeatureExtractorNodelet() : show_image_(false)
    { }

    typedef pcl::PointCloud<Feature2D> FeatureCloud;

  private:
    virtual void onInit()
    {
        ros::NodeHandle nh = ros::NodeHandle(getNodeHandle(), "feature_extractor");
        ros::NodeHandle& private_nh = getPrivateNodeHandle();
        
        it_.reset(new image_transport::ImageTransport(getNodeHandle()));

        // TODO subscribe / unsubscribe on demand?
        sub_image_ = it_->subscribe("image", 1, &FeatureExtractorNodelet::imageCb, this);
        sub_region_of_interest_ = nh.subscribe("region_of_interest", 10, &FeatureExtractorNodelet::setRegionOfInterest, this);
        
        pub_feature_cloud_ = nh.advertise<FeatureCloud>("feature_cloud", 1);
        pub_features_ = nh.advertise<vision_msgs::Features>("features", 1);

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
            // bridge to opencv
            namespace enc = sensor_msgs::image_encodings;
            cv_bridge::CvImageConstPtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(image_msg, enc::BGR8);
            
            cv::Mat image = cv_ptr->image;

            // Calculate features
            std::vector<feature_extraction::KeyPoint> key_points;
            cv::Mat descriptors;
            feature_extractor_->extract(image, key_points, descriptors, region_of_interest_);

            NODELET_DEBUG("%zu features extracted.", key_points.size());
            if (key_points.size() == 0)
            {
                return;
            }

            if (pub_feature_cloud_.getNumSubscribers() > 0)
            {
              FeatureCloud::Ptr feature_cloud(new FeatureCloud());
              feature_cloud->header = image_msg->header;
              feature_cloud->points.resize(key_points.size());
              for (size_t i = 0; i < key_points.size(); ++i)
              {
                  Feature2D& feature = feature_cloud->points[i];
                  std::vector<float> descriptor = descriptors.row(i);
                  std::copy(descriptor.begin(), descriptor.end(), feature.data);
                  feature.x = key_points[i].pt.x;
                  feature.y = key_points[i].pt.y;
              }
              pub_feature_cloud_.publish(feature_cloud);
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
              features_msg->descriptor_length = descriptors.cols;
              assert(descriptors.isContinuous());
              assert(descriptors.depth() == CV_32F);
              assert(descriptors.channels() == 1);
              features_msg->descriptor_data.insert(features_msg->descriptor_data.begin(), descriptors.data,
                  descriptors.data + descriptors.cols * descriptors.rows);
              pub_features_.publish(features_msg);
            }

            if (show_image_)
            {
                cv::Mat canvas = image.clone();
                drawKeyPoints(canvas, key_points);
                cv::rectangle(canvas, region_of_interest_.tl(),
                        region_of_interest_.br(), cv::Scalar(0, 0, 255), 3);
                cv::namedWindow("Feature Extraction", 0);
                cv::imshow("Feature Extraction", canvas);
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
    ros::Publisher pub_feature_cloud_;
    ros::Publisher pub_features_;

    // the extractor
    feature_extraction::FeatureExtractor::Ptr feature_extractor_;

    // the current roi
    cv::Rect region_of_interest_;

    // show debug image
    bool show_image_;
 
};

} // end of namespace

#include <pluginlib/class_list_macros.h>
PLUGINLIB_DECLARE_CLASS(feature_extraction, 
    FeatureExtractor, 
    feature_extraction_ros::FeatureExtractorNodelet, nodelet::Nodelet);

