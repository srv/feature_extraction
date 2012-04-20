#include <ros/package.h>

#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "feature_extraction/key_point_detector_factory.h"
#include "feature_extraction/descriptor_extractor_factory.h"

using namespace feature_extraction;

struct Params
{
  std::string detector_name;
  std::string extractor_name;
};

std::ostream& operator<<(std::ostream& os, const Params& params) {
  os << "Detector: " << params.detector_name << ", Extractor: " << params.extractor_name;
  return os;
}

std::vector<Params> generateParams()
{
  std::vector<Params> params;

  std::vector<std::string> detector_names = KeyPointDetectorFactory::getDetectorNames();
  std::vector<std::string> extractor_names = DescriptorExtractorFactory::getExtractorNames();
  for (size_t i = 0; i < detector_names.size(); ++i)
  {
    for (size_t j = 0; j < extractor_names.size(); ++j)
    {
      Params p;
      p.detector_name = detector_names[i];
      p.extractor_name = extractor_names[j];
      params.push_back(p);
    }
  }
  return params;
}

class DetectorExtractorCombinationsTest : public ::testing::TestWithParam<Params>
{
};

TEST_P(DetectorExtractorCombinationsTest, runTest)
{

  std::cout << "Testing detector '" << GetParam().detector_name << "' "
               "with extractor '" << GetParam().extractor_name <<"'" << std::endl;
  std::string path = ros::package::getPath(ROS_PACKAGE_NAME);
  cv::Mat image = cv::imread(path + "/data/black_box.jpg");
  ASSERT_FALSE(image.empty());
 
  KeyPointDetector::Ptr detector = 
    KeyPointDetectorFactory::create(GetParam().detector_name);
  std::vector<cv::KeyPoint> key_points;
  double time = (double)cv::getTickCount();
  detector->detect(image, key_points);
  EXPECT_GT(key_points.size(), 0);

  DescriptorExtractor::Ptr extractor = 
    DescriptorExtractorFactory::create(GetParam().extractor_name);
  cv::Mat descriptors;
  extractor->extract(image, key_points, descriptors);
  time = ((double)cv::getTickCount() - time)/cv::getTickFrequency() * 1000;
  EXPECT_GT(key_points.size(), 0);
  EXPECT_EQ(key_points.size(), descriptors.rows);
}

INSTANTIATE_TEST_CASE_P(DetectorExtractorCombinationsTests, 
                        DetectorExtractorCombinationsTest,
                        ::testing::ValuesIn(generateParams()));

