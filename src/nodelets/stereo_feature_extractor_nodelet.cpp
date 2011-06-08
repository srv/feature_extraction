#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_proc/advertisement_checker.h>

#include <image_geometry/stereo_camera_model.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include "stereo_camera_model.h"
#include "stereo_feature_extractor.h"
#include "feature_extractor_factory.h"


static const char* WINDOW_NAME = "Stereo Features";

namespace stereo_feature_extraction
{

class StereoFeatureExtractorNodelet : public nodelet::Nodelet
{
  public:
    StereoFeatureExtractorNodelet() :
        stereo_camera_model_(new StereoCameraModel())
    { }


  private:
    virtual void onInit()
    {
        ros::NodeHandle nh = getNodeHandle();
        ros::NodeHandle& private_nh = getPrivateNodeHandle();
        it_.reset(new image_transport::ImageTransport(nh));
        subscribed_ = false; // no subscription yet
        
        ros::SubscriberStatusCallback connect_cb = 
            boost::bind(&StereoFeatureExtractorNodelet::connectCb, this);

        pub_points2_  = nh.advertise<sensor_msgs::PointCloud2>("stereo_features",  1, 
                connect_cb, connect_cb);

        // Synchronize inputs. Topic subscriptions happen on demand in the 
        // connection callback. Optionally do approximate synchronization.
        int queue_size;
        private_nh.param("queue_size", queue_size, 5);
        bool approx;
        private_nh.param("approximate_sync", approx, false);

        private_nh.param("max_y_diff", max_y_diff_, 2.0);
        private_nh.param("max_angle_diff", max_angle_diff_, 2.0);
        private_nh.param("max_size_diff", max_size_diff_, 2);
        private_nh.param("visualize", visualize_, true);

        std::string feature_extractor_name;
        private_nh.param("feature_extractor", feature_extractor_name, 
                std::string("SURF"));
        FeatureExtractor::Ptr feature_extractor = 
            FeatureExtractorFactory::create(feature_extractor_name);
        if (feature_extractor.get() == NULL)
        {
            NODELET_FATAL("Cannot create feature extractor with name %s",
                    feature_extractor_name.c_str());
        }
        else
        {
            stereo_feature_extractor_.setFeatureExtractor(feature_extractor);
        }
        stereo_feature_extractor_.setCameraModel(stereo_camera_model_);

        std::cout << "Parameters: max_y_diff = " << max_y_diff_
                  << " max_angle_diff = " << max_angle_diff_
                  << " max_size_diff = " << max_size_diff_
                  << " feature_extractor = " << feature_extractor_name
                  << " visualize = " << visualize_ << std::endl;
        if (approx)
        {
            approximate_sync_.reset(
                    new ApproximateSync(ApproximatePolicy(queue_size),
                                        sub_l_image_, sub_r_image_,
                                        sub_l_info_, sub_r_info_));
            approximate_sync_->registerCallback(
                    boost::bind(&StereoFeatureExtractorNodelet::imageCb,
                                this, _1, _2, _3, _4));
        }
        else
        {
            exact_sync_.reset(new ExactSync(ExactPolicy(queue_size),
                                            sub_l_image_, sub_r_image_,
                                            sub_l_info_, sub_r_info_));
            exact_sync_->registerCallback(
                    boost::bind(&StereoFeatureExtractorNodelet::imageCb,
                                this, _1, _2, _3, _4));
        }

        if (visualize_) cv::namedWindow(WINDOW_NAME, 0);

        NODELET_INFO("Waiting for client subscriptions.");
    }

    // Handles (un)subscribing when clients (un)subscribe
    void connectCb()
    {
        if (pub_points2_.getNumSubscribers() == 0)
        {
            NODELET_INFO("No more clients connected, unsubscribing from camera.");
            sub_l_image_  .unsubscribe();
            sub_r_image_  .unsubscribe();
            sub_l_info_   .unsubscribe();
            sub_r_info_   .unsubscribe();
            subscribed_ = false;
        }
        else if (!subscribed_)
        {
            NODELET_INFO("Client connected, subscribing to camera.");
            ros::NodeHandle &nh = getNodeHandle();
            // Queue size 1 should be OK; the one that matters is the synchronizer queue size.
            sub_l_image_  .subscribe(*it_, "left/image_rect", 1);
            sub_r_image_  .subscribe(*it_, "right/image_rect", 1);
            sub_l_info_   .subscribe(nh,   "left/camera_info", 1);
            sub_r_info_   .subscribe(nh,   "right/camera_info", 1);
            subscribed_ = true;
        }
    }

    void imageCb(const sensor_msgs::ImageConstPtr& l_image_msg,
                 const sensor_msgs::ImageConstPtr& r_image_msg,
                 const sensor_msgs::CameraInfoConstPtr& l_info_msg,
                 const sensor_msgs::CameraInfoConstPtr& r_info_msg)
    {
        // bridge to opencv
        namespace enc = sensor_msgs::image_encodings;
        cv_bridge::CvImageConstPtr cv_ptr_left;
        cv_bridge::CvImageConstPtr cv_ptr_right;
        try
        {
            cv_ptr_left = cv_bridge::toCvShare(l_image_msg, enc::BGR8);
            cv_ptr_right = cv_bridge::toCvShare(r_image_msg, enc::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            NODELET_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Update the camera model
        stereo_camera_model_->fromCameraInfo(*l_info_msg, *r_info_msg);
        cv::Mat mask;
        // Calculate stereo features
        std::vector<StereoFeature> stereo_features = stereo_feature_extractor_.extract(
            cv_ptr_left->image, cv_ptr_right->image, mask, mask, 
            max_y_diff_, max_angle_diff_, max_size_diff_);

        if (visualize_)
        {
            paintStereoFeatures(cv_ptr_left->image, stereo_features);
        }

        NODELET_INFO("%i stereo features", stereo_features.size());
        std::cout << stereo_features.size() << " stereo features" << std::endl;

        // Fill in new PointCloud2 message (2D image-like layout)
        sensor_msgs::PointCloud2Ptr points_msg = 
            boost::make_shared<sensor_msgs::PointCloud2>();
        points_msg->header = l_image_msg->header;

        points_msg->height = stereo_features.size();
        points_msg->width  = 1;
        points_msg->fields.resize(4);
        points_msg->fields[0].name = "x";
        points_msg->fields[0].offset = 0;
        points_msg->fields[0].count = 1;
        points_msg->fields[0].datatype = sensor_msgs::PointField::FLOAT32;
        points_msg->fields[1].name = "y";
        points_msg->fields[1].offset = 4;
        points_msg->fields[1].count = 1;
        points_msg->fields[1].datatype = sensor_msgs::PointField::FLOAT32;
        points_msg->fields[2].name = "z";
        points_msg->fields[2].offset = 8;
        points_msg->fields[2].count = 1;
        points_msg->fields[2].datatype = sensor_msgs::PointField::FLOAT32;
        points_msg->fields[3].name = "rgb";
        points_msg->fields[3].offset = 12;
        points_msg->fields[3].count = 1;
        points_msg->fields[3].datatype = sensor_msgs::PointField::FLOAT32;
        static const int STEP = 16;
        points_msg->point_step = STEP;
        points_msg->row_step = points_msg->point_step * points_msg->width;
        points_msg->data.resize (points_msg->row_step * points_msg->height);
        points_msg->is_dense = false; // there may be invalid points

        float bad_point = std::numeric_limits<float>::quiet_NaN ();

        for (size_t i = 0; i < stereo_features.size(); ++i)
        {
            int offset = i * points_msg->point_step;
            const cv::Point3d& point = stereo_features[i].world_point;
            float x = point.x;
            float y = point.y;
            float z = point.z;
            // pack data into point message data
            memcpy (&points_msg->data[offset + 0], &x, sizeof (float));
            memcpy (&points_msg->data[offset + 4], &y, sizeof (float));
            memcpy (&points_msg->data[offset + 8], &z, sizeof (float));
            const std::string& encoding = l_image_msg->encoding;
            const cv::Point2f& image_point = stereo_features[i].key_point.pt;
            cv::Vec3b bgr;
            if (encoding == enc::BGR8)
            {
                bgr = cv_ptr_left->image.at<cv::Vec3b>(image_point.y, image_point.x);
            }
            else if (encoding == enc::MONO8)
            {
                unsigned char value = 
                    cv_ptr_left->image.at<unsigned char>(image_point.y, image_point.x);
                bgr[0] = value; bgr[1] = value; bgr[2] = value;
            }
            else
            {
                bgr[0] = 0; bgr[1] = 0; bgr[2] = 0;
                NODELET_WARN_THROTTLE(30, 
                        "Could not fill color channel of the point cloud, "
                        "unsupported encoding '%s'", encoding.c_str());
            }
            int32_t rgb_packed = (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
            memcpy (&points_msg->data[offset + 12], &rgb_packed, sizeof (int32_t));
         }
        /*
        for (int v = 0; v < mat.rows; ++v)
        {
        for (int u = 0; u < mat.cols; ++u, offset += STEP)
        {
            if (isValidPoint(mat(v,u)))
            {
            // x,y,z,rgba
            memcpy (&points_msg->data[offset + 0], &mat(v,u)[0], sizeof (float));
            memcpy (&points_msg->data[offset + 4], &mat(v,u)[1], sizeof (float));
            memcpy (&points_msg->data[offset + 8], &mat(v,u)[2], sizeof (float));
            }
            else
            {
            memcpy (&points_msg->data[offset + 0], &bad_point, sizeof (float));
            memcpy (&points_msg->data[offset + 4], &bad_point, sizeof (float));
            memcpy (&points_msg->data[offset + 8], &bad_point, sizeof (float));
            }
        }
        }

        // Fill in color
        namespace enc = sensor_msgs::image_encodings;
        const std::string& encoding = l_image_msg->encoding;
        offset = 0;
        if (encoding == enc::MONO8)
        {
        const cv::Mat_<uint8_t> color(l_image_msg->height, l_image_msg->width,
                                        (uint8_t*)&l_image_msg->data[0],
                                        l_image_msg->step);
        for (int v = 0; v < mat.rows; ++v)
        {
            for (int u = 0; u < mat.cols; ++u, offset += STEP)
            {
            if (isValidPoint(mat(v,u)))
            {
                uint8_t g = color(v,u);
                int32_t rgb = (g << 16) | (g << 8) | g;
                memcpy (&points_msg->data[offset + 12], &rgb, sizeof (int32_t));
            }
            else
            {
                memcpy (&points_msg->data[offset + 12], &bad_point, sizeof (float));
            }
            }
        }
        }
        else if (encoding == enc::RGB8)
        {
        const cv::Mat_<cv::Vec3b> color(l_image_msg->height, l_image_msg->width,
                                        (cv::Vec3b*)&l_image_msg->data[0],
                                        l_image_msg->step);
        for (int v = 0; v < mat.rows; ++v)
        {
            for (int u = 0; u < mat.cols; ++u, offset += STEP)
            {
            if (isValidPoint(mat(v,u)))
            {
                const cv::Vec3b& rgb = color(v,u);
                int32_t rgb_packed = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
                memcpy (&points_msg->data[offset + 12], &rgb_packed, sizeof (int32_t));
            }
            else
            {
                memcpy (&points_msg->data[offset + 12], &bad_point, sizeof (float));
            }
            }
        }
        }
        else if (encoding == enc::BGR8)
        {
        const cv::Mat_<cv::Vec3b> color(l_image_msg->height, l_image_msg->width,
                                        (cv::Vec3b*)&l_image_msg->data[0],
                                        l_image_msg->step);
        for (int v = 0; v < mat.rows; ++v)
        {
            for (int u = 0; u < mat.cols; ++u, offset += STEP)
            {
            if (isValidPoint(mat(v,u)))
            {
                const cv::Vec3b& bgr = color(v,u);
                int32_t rgb_packed = (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
                memcpy (&points_msg->data[offset + 12], &rgb_packed, sizeof (int32_t));
            }
            else
            {
                memcpy (&points_msg->data[offset + 12], &bad_point, sizeof (float));
            }
            }
        }
        }
        else
        {
        NODELET_WARN_THROTTLE(30, "Could not fill color channel of the point cloud, "
                                "unsupported encoding '%s'", encoding.c_str());
        }

        pub_points2_.publish(points_msg);
        }

*/
        pub_points2_.publish(points_msg);
    }

    void paintStereoFeatures(const cv::Mat& image, 
            const std::vector<StereoFeature>& stereo_features)
    {
        cv::Mat canvas = image.clone();
        for (size_t i = 0; i < stereo_features.size(); ++i)
        {
            cv::Point center(cvRound(stereo_features[i].key_point.pt.x),
                             cvRound(stereo_features[i].key_point.pt.y));
            int radius = cvRound(stereo_features[i].key_point.size / 2);
            cv::circle(canvas, center, radius, cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow(WINDOW_NAME, canvas);
        cv::waitKey(10);
    }

            
    boost::shared_ptr<image_transport::ImageTransport> it_;
    image_transport::SubscriberFilter sub_l_image_;
    image_transport::SubscriberFilter sub_r_image_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> sub_l_info_, sub_r_info_;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image,
        sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactPolicy;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
        sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ApproximatePolicy;
    typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
    typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;
    boost::shared_ptr<ExactSync> exact_sync_;
    boost::shared_ptr<ApproximateSync> approximate_sync_;
    bool subscribed_; // stores if anyone is subscribed

    // Publications
    ros::Publisher pub_points2_;

    // Processing state (note: only safe because we're single-threaded!)
    image_geometry::StereoCameraModel model_;
    cv::Mat_<cv::Vec3f> points_mat_; // scratch buffer

    // Error reporting when input topics are not advertised
    boost::shared_ptr<image_proc::AdvertisementChecker> check_inputs_;

    // the camera model
    StereoCameraModel::Ptr stereo_camera_model_;

    // the extractor
    StereoFeatureExtractor stereo_feature_extractor_;

    // matching parameters
    double max_y_diff_;
    double max_angle_diff_;
    int max_size_diff_;

    // visualization
    bool visualize_;
 
};

PLUGINLIB_DECLARE_CLASS(stereo_feature_extraction, 
    stereo_feature_extractor, 
    stereo_feature_extraction::StereoFeatureExtractorNodelet, nodelet::Nodelet);
}

