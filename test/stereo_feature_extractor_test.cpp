#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>

#include "ros/package.h"

#include "stereo_feature_extractor.h"
#include "feature_extractor_factory.h"
#include "sensor_msgs/CameraInfo.h"
#include "image_geometry/stereo_camera_model.h"


using namespace stereo_feature_extraction;

TEST(StereoFeatureExtractor, runTest)
{
    cv::Mat image_left(768, 1024, CV_8UC3);
    cv::Mat image_right(768, 1024, CV_8UC3);

    //cv::randu(image_left, cv::Scalar(0, 0, 0), cv::Scalar(50, 50, 50));
    //cv::randu(image_right, cv::Scalar(0, 0, 0), cv::Scalar(50, 50, 50));
    image_left = cv::Scalar::all(0);
    image_right = cv::Scalar::all(0);

    cv::rectangle(image_left, cv::Point(20, 20), cv::Point(120, 120), cv::Scalar::all(255), CV_FILLED);
    cv::rectangle(image_right, cv::Point(21, 21), cv::Point(121, 121), cv::Scalar::all(255), CV_FILLED);

    FeatureExtractor::Ptr feature_extractor = 
        FeatureExtractorFactory::create("CvSURF");

    // read calibration data to fill stereo camera model
    std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
    std::string left_file = path + "/data/artificial_calibration_left.yaml";
    std::string right_file = path + "/data/artificial_calibration_right.yaml";

    StereoCameraModel::Ptr stereo_camera_model = 
        StereoCameraModel::Ptr(new StereoCameraModel());
    stereo_camera_model->fromCalibrationFiles(left_file, right_file);
    StereoFeatureExtractor extractor(feature_extractor, stereo_camera_model);

    cv::Mat mask_left, mask_right;
    double max_y_diff = 2.0;
    double max_angle_diff = 5.0;
    int max_size_diff = 2;
    std::vector<StereoFeature> stereo_features = 
        extractor.extract(image_left, image_right, mask_left, mask_right,
                max_y_diff, max_angle_diff, max_size_diff);
    std::cout << "Found " << stereo_features.size() << " stereo features." << std::endl;

}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

