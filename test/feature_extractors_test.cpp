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
    for (size_t i = 0; i < extractor_names.size(); ++i)
    {
        FeatureExtractor::Ptr extractor =
            FeatureExtractorFactory::create(extractor_names[i]);

        FeatureExtractor::Ptr extractor2 =
            FeatureExtractorFactory::create(extractor_names[i]);


        cv::Mat image(600, 800, CV_8UC1, cv::Scalar(0));
        int rect_width = 100;
        int rect_height = 80;
        cv::Point rect_tl(image.cols / 2 - rect_width / 2, image.rows / 2 - rect_height / 2);
        cv::Point rect_br = rect_tl + cv::Point(rect_width, rect_height);
        cv::rectangle(image, rect_tl, rect_br, cv::Scalar::all(255), CV_FILLED);

        cv::Point shift(0, 0);

        cv::Mat image2(600, 800, CV_8UC1, cv::Scalar(0));
        cv::rectangle(image2, rect_tl + shift, rect_br + shift, cv::Scalar::all(255), CV_FILLED);
        
        cv::Mat mask;
        std::vector<KeyPoint> key_points;
        cv::Mat descriptors;
        extractor->extract(image, mask, key_points, descriptors);

        std::vector<KeyPoint> key_points2;
        cv::Mat descriptors2;
        extractor2->extract(image2, mask, key_points2, descriptors2);

        ASSERT_EQ(key_points.size(), key_points2.size());
        ASSERT_EQ(descriptors.rows, descriptors2.rows);
        ASSERT_EQ(key_points.size(), descriptors.rows);
        std::cout << extractor_names[i] << " extracted " << key_points.size() << " key points." << std::endl;

        for (size_t i = 0; i < key_points.size(); ++i)
        {
            const KeyPoint& kp1 = key_points[i];
            const KeyPoint& kp2 = key_points2[i];
            const cv::Mat desc1 = descriptors.row(i);
            const cv::Mat desc2 = descriptors2.row(i);
            EXPECT_EQ(kp2.pt.x - kp1.pt.x, shift.x);
            EXPECT_EQ(kp2.pt.y - kp1.pt.y, shift.y);
            ASSERT_EQ(desc1.cols, desc2.cols);
            for (int j = 0; j < desc1.cols; ++j)
            {
                EXPECT_EQ(desc1.at<float>(0,i), desc2.at<float>(0,i));
            }
        }
    }
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

