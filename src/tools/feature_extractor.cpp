#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "feature_extraction/key_point_detector_factory.h"
#include "feature_extraction/descriptor_extractor_factory.h"

using namespace feature_extraction;
namespace po = boost::program_options;

int main(int argc, char** argv)
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("image,I", po::value<std::string>()->required(), "input image")
    ("key_point_detector,K", po::value<std::string>()->default_value("SmartSURF"), "key point detector")
    ("descriptor_extractor,E", po::value<std::string>()->default_value("SmartSURF"), "descriptor extractor")
  ;

  po::variables_map vm;
  try
  {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (const po::error& error)
  {
    std::cerr << "Error parsing program options: " << std::endl;
    std::cerr << "  " << error.what() << std::endl;
    std::cerr << desc << std::endl;
    return -1;
  }

  // extract params
  std::string image_file = vm["image"].as<std::string>();
  std::string key_point_detector_name = vm["key_point_detector"].as<std::string>();
  std::string descriptor_extractor_name = vm["descriptor_extractor"].as<std::string>();

  // create instances
  KeyPointDetector::Ptr key_point_detector =
    KeyPointDetectorFactory::create(key_point_detector_name);
  if (key_point_detector.get() == 0)
  {
    std::cerr << "Cannot create key point detector with name '"
              << key_point_detector_name << "'" << std::endl;
    return -2;
  }
  DescriptorExtractor::Ptr descriptor_extractor =
    DescriptorExtractorFactory::create(descriptor_extractor_name);
  if (descriptor_extractor.get() == 0)
  {
    std::cerr << "Cannot create descriptor extractor with name '"
              << descriptor_extractor_name << "'" << std::endl;
    return -2;
  }

  // load image (as 1 channel)
  cv::Mat image = cv::imread(image_file, 0);

  // extract key points and descriptors
  std::vector<cv::KeyPoint> key_points;
  key_point_detector->detect(image, key_points);
  cv::Mat descriptors;
  descriptor_extractor->extract(image, key_points, descriptors);

  std::cout << "Extracted " << key_points.size() << " descriptors" << std::endl;

  return 0;
}

