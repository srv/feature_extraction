#include <ros/package.h>

#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>
#include "feature_extractor_factory.h"

#include "Surf.h"

using namespace stereo_feature_extraction;

TEST(Extractor, runTest)
{
    std::vector<std::string> extractor_names;
    extractor_names.push_back("CvSURF");
    extractor_names.push_back("SURF");

    std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
    cv::Mat image = cv::imread(path + "/data/black_box.jpg");
    ASSERT_FALSE(image.empty());

    for (size_t i = 0; i < extractor_names.size(); ++i)
    {
        FeatureExtractor::Ptr extractor =
            FeatureExtractorFactory::create(extractor_names[i]);

        cv::Mat mask;
        std::vector<KeyPoint> key_points;
        cv::Mat descriptors;
        double time = (double)cv::getTickCount();
        extractor->extract(image, mask, key_points, descriptors);
        time = ((double)cv::getTickCount() - time)/cv::getTickFrequency() * 1000;

        ASSERT_GT(key_points.size(), 0);
        std::cout << extractor_names[i] << " extracted " << key_points.size() 
            << " key points in " << time << "ms." << std::endl;
    }
}

TEST(Extractor, maxNumPointsTest)
{
    std::vector<std::string> extractor_names;
    extractor_names.push_back("CvSURF");
    extractor_names.push_back("SURF");

    std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
    cv::Mat image = cv::imread(path + "/data/black_box.jpg");
    ASSERT_FALSE(image.empty());

    for (size_t i = 0; i < extractor_names.size(); ++i)
    {
        std::cout << "*** Extractor is " << extractor_names[i] << " ***" << std::endl;
        for (int p = 10; p <= 100; p+=10)
        {
            FeatureExtractor::Ptr extractor =
                FeatureExtractorFactory::create(extractor_names[i]);

            cv::Mat mask;
            std::vector<KeyPoint> key_points;
            cv::Mat descriptors;
            extractor->setMaxNumKeyPoints(p);
            extractor->extract(image, mask, key_points, descriptors);
            EXPECT_LE((int)key_points.size(), p);
            std::cout << "extracted " << key_points.size() 
                << " key points (max was set to " << p << ")" << std::endl;
        }
    }
}


// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

