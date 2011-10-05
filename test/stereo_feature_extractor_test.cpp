#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>

#include <ros/package.h>

#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/stereo_camera_model.h>

#include <feature_extraction/feature_extractor_factory.h>

#include "stereo_feature_extraction/stereo_feature_extractor.h"
#include "stereo_feature_extraction/drawing.h"

using namespace stereo_feature_extraction;
using feature_extraction::FeatureExtractor;
using feature_extraction::FeatureExtractorFactory;

StereoFeatureExtractor createStandardExtractor()
{
    FeatureExtractor::Ptr feature_extractor = 
        FeatureExtractorFactory::create("SURF");

    // read calibration data to fill stereo camera model
    std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
    std::string left_file = path + "/data/real_calibration_left.yaml";
    std::string right_file = path + "/data/real_calibration_right.yaml";

    StereoCameraModel::Ptr stereo_camera_model = 
        StereoCameraModel::Ptr(new StereoCameraModel());
    stereo_camera_model->fromCalibrationFiles(left_file, right_file);

    StereoFeatureExtractor extractor;
    extractor.setFeatureExtractor(feature_extractor);
    extractor.setCameraModel(stereo_camera_model);
    return extractor;
}

void runTest(StereoFeatureExtractor::MatchMethod match_method)
{
    std::string path = ros::package::getPath(ROS_PACKAGE_NAME);

    // load image
    cv::Mat image_left = cv::imread(path + "/data/left_100cm.jpg");
    ASSERT_FALSE(image_left.empty());
    cv::Mat image_right = cv::imread(path + "/data/right_100cm.jpg");
    ASSERT_FALSE(image_right.empty());

    StereoFeatureExtractor extractor = createStandardExtractor();

    extractor.setMatchMethod(match_method);
    double time = (double)cv::getTickCount();
    StereoFeatureSet stereo_feature_set = 
        extractor.extract(image_left, image_right);
    time = ((double)cv::getTickCount() - time)/cv::getTickFrequency() * 1000;
    std::cout << "Found " << stereo_feature_set.stereo_features.size() 
        << " stereo features" << " in " << time << "ms." << std::endl;
}

TEST(StereoFeatureExtractor, keyPointToBlockRunTest)
{
    runTest(StereoFeatureExtractor::KEY_POINT_TO_BLOCK);
}

TEST(StereoFeatureExtractor, keyPointToKeyPointRunTest)
{
    runTest(StereoFeatureExtractor::KEY_POINT_TO_KEY_POINT);
}

TEST(StereoFeatureExtractor, roiTest)
{
    std::string path = ros::package::getPath(ROS_PACKAGE_NAME);

    // load image
    cv::Mat image_left = cv::imread(path + "/data/left_100cm.jpg");
    ASSERT_FALSE(image_left.empty());
    cv::Mat image_right = cv::imread(path + "/data/right_100cm.jpg");
    ASSERT_FALSE(image_right.empty());

    StereoFeatureExtractor extractor = createStandardExtractor();

    // try run on region of interest 
    int roi_x = image_left.cols / 4;
    int roi_y = image_left.rows / 4;
    int roi_width = image_left.cols / 2;
    int roi_height = image_left.rows / 2;
    cv::Rect roi(roi_x, roi_y, roi_width, roi_height);
    extractor.setRegionOfInterest(roi);
    extractor.setMinDepth(0.5);
    extractor.setMaxDepth(1.5);
    StereoFeatureSet stereo_feature_set = 
        extractor.extract(image_left, image_right);
    std::vector<StereoFeature>& stereo_features = 
        stereo_feature_set.stereo_features;
    EXPECT_GT(stereo_features.size(), 0);

    // check for right depth and coordinates inside roi
    for (size_t i = 0; i < stereo_features.size(); ++i)
    {
        EXPECT_NEAR(stereo_features[i].world_point.z, 1.0, 0.05); // 5cm tolerance
        EXPECT_GE(stereo_features[i].key_point_left.pt.x, roi_x);
        EXPECT_GE(stereo_features[i].key_point_left.pt.y, roi_y);
        EXPECT_LT(stereo_features[i].key_point_left.pt.x, roi_x + roi_width);
        EXPECT_LT(stereo_features[i].key_point_left.pt.y, roi_y + roi_height);
    }
}

TEST(StereoFeatureExtractor, depthResolutionTest)
{
    std::string path = ros::package::getPath(ROS_PACKAGE_NAME);

    for (int k = 50; k <= 100; k+=10)
    {
        std::cout << "*** test for approx. " << k << "cm distance ***" << std::endl;
        // load image
        std::ostringstream filename;
        filename << path << "/data/left_" << k << "cm.jpg";
        cv::Mat image_left = cv::imread(filename.str());
        ASSERT_FALSE(image_left.empty());
        filename.str("");
        filename << path << "/data/right_" << k << "cm.jpg";
        cv::Mat image_right = cv::imread(filename.str());
        ASSERT_FALSE(image_right.empty());

        FeatureExtractor::Ptr feature_extractor = 
            FeatureExtractorFactory::create("SURF");

        // read calibration data to fill stereo camera model
        std::string left_file = path + "/data/real_calibration_left.yaml";
        std::string right_file = path + "/data/real_calibration_right.yaml";

        StereoCameraModel::Ptr stereo_camera_model = 
            StereoCameraModel::Ptr(new StereoCameraModel());
        stereo_camera_model->fromCalibrationFiles(left_file, right_file);

        StereoFeatureExtractor extractor;
        extractor.setFeatureExtractor(feature_extractor);
        extractor.setMatchMethod(StereoFeatureExtractor::KEY_POINT_TO_KEY_POINT);
        extractor.setCameraModel(stereo_camera_model);

        StereoFeatureSet stereo_feature_set = 
            extractor.extract(image_left, image_right);
        std::vector<StereoFeature>& stereo_features =
            stereo_feature_set.stereo_features;
        std::cout << "Found " << stereo_features.size() << " stereo features." << std::endl;

        ASSERT_TRUE(stereo_features.size() > 0);

        pcl::PointCloud<pcl::PointXYZ> cloud;

        // Fill in the cloud data
        cloud.width  = stereo_features.size();
        cloud.height = 1;

        // we cut off everything that is inside a border of 100px because
        // the test images plane was not big enough
        for (size_t i = 0; i < stereo_features.size (); ++i)
        {
            if (stereo_features[i].key_point_left.pt.x > 100 &&
                stereo_features[i].key_point_left.pt.x < image_left.cols - 100 &&
                stereo_features[i].key_point_left.pt.y > 100 &&
                stereo_features[i].key_point_left.pt.y < image_left.rows - 100)
            {
                pcl::PointXYZ point;
                point.x = stereo_features[i].world_point.x;
                point.y = stereo_features[i].world_point.y;
                point.z = stereo_features[i].world_point.z;
                cloud.push_back(point);
            }
        }
        pcl::ModelCoefficients coefficients;
        pcl::PointIndices inliers;
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients (true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.05);
        seg.setInputCloud(cloud.makeShared());
        seg.segment(inliers, coefficients);

        // there must be some inliers
        ASSERT_FALSE(inliers.indices.size() == 0);

        std::cout << "plane model inliers: " << inliers.indices.size () << std::endl;
        double a = coefficients.values[0];
        double b = coefficients.values[1];
        double c = coefficients.values[2];
        double d = coefficients.values[3];

        std::cout << "fitted plane coefficients: " << a << " " << b << " " 
            << c << " " << d << std::endl;

        // check if the computed plane has the right distance and normal vector
        EXPECT_NEAR(fabs(d), k/100.0, 0.05);
        EXPECT_NEAR(a, 0.0, 0.1);
        EXPECT_NEAR(b, 0.0, 0.1);
        EXPECT_NEAR(std::abs(c), 1.0, 0.1);

        std::vector<double> distances(inliers.indices.size());
        for (size_t i = 0; i < inliers.indices.size(); ++i)
        {
            const pcl::PointXYZ& point = cloud.points[inliers.indices[i]];
            distances[i] = std::abs(a * point.x + b * point.y + c * point.z + d);
        }

        // http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        double m2 = 0; // helper for floating variance computation
        double mean_distance = 0;
        double min_distance = std::numeric_limits<double>::max();
        double max_distance = 0;
        for (size_t i = 0; i < distances.size(); ++i)
        {
            if (distances[i] > max_distance) max_distance = distances[i];
            if (distances[i] < min_distance) min_distance = distances[i];
            double delta = distances[i] - mean_distance;
            mean_distance += delta / (i + 1);
            m2 += delta * (distances[i] - mean_distance);
        }
        double variance = m2 / (distances.size() - 1);
        double stddev = sqrt(variance);
        std::cout << "distances from fitted plane: mean = " << mean_distance
            << " stddev = " << stddev << " min = " << min_distance 
            << " max = " << max_distance << std::endl;
        EXPECT_LT(mean_distance, 0.01); // we expect a smaller error than 1cm
        EXPECT_LT(stddev, 0.005);        // with a std deviation of 0.05cm
    }
}

TEST(StereoFeatureExtractor, roiSpeedTest)
{
    std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
    path += "/data/";

    // load image
    cv::Mat image_left = cv::imread(path + "left_100cm.jpg");
    ASSERT_FALSE(image_left.empty());
    cv::Mat image_right = cv::imread(path + "right_100cm.jpg");
    ASSERT_FALSE(image_right.empty());

    FeatureExtractor::Ptr feature_extractor = 
        FeatureExtractorFactory::create("SURF");

    // read calibration data to fill stereo camera model
    std::string left_file = path + "real_calibration_left.yaml";
    std::string right_file = path + "real_calibration_right.yaml";

    StereoCameraModel::Ptr stereo_camera_model = 
        StereoCameraModel::Ptr(new StereoCameraModel());
    stereo_camera_model->fromCalibrationFiles(left_file, right_file);

    StereoFeatureExtractor extractor;
    extractor.setFeatureExtractor(feature_extractor);
    extractor.setCameraModel(stereo_camera_model);

    double time = (double)cv::getTickCount();
    StereoFeatureSet stereo_feature_set = 
        extractor.extract(image_left, image_right);
    std::vector<StereoFeature>& stereo_features = 
        stereo_feature_set.stereo_features;
    time = ((double)cv::getTickCount() - time)/cv::getTickFrequency() * 1000;
    std::cout << "Image " << image_left.cols << "x" << image_left.rows 
        << " found " << stereo_features.size() << " stereo features" 
        << " in " << time << "ms." << std::endl;

    for (int w = 100; w < 800; w += 100)
    {
        cv::Rect roi(0, 0, w, w);
        extractor.setRegionOfInterest(roi);
        double time = (double)cv::getTickCount();
        stereo_feature_set = extractor.extract(image_left, image_right);
        time = ((double)cv::getTickCount() - time)/cv::getTickFrequency() * 1000;
        std::cout << "Image " << w << "x" << w 
            << " found " << stereo_feature_set.stereo_features.size()
            << " stereo features" << " in " << time << "ms." << std::endl;
    }
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

