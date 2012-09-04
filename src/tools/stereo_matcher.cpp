#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "feature_matching/stereo_feature_matcher.h"

namespace po = boost::program_options;

int main(int argc, char** argv)
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("left_features,L", po::value<std::string>()->required(), "left features file")
    ("right_features,R", po::value<std::string>()->required(), "right features file")
    ("max_y_diff,Y", po::value<double>()->default_value(2.0), "maximum y difference for matching keypoints")
    ("max_angle_diff,A", po::value<double>()->default_value(4.0), "maximum angle difference for matching keypoints")
    ("max_size_diff,S", po::value<int>()->default_value(5), "maximum size difference for matching keypoints")
    ("matching_threshold,T", po::value<double>()->default_value(0.8), "matching threshold")
    ("output_matches_file,O", po::value<std::string>(), "filename for output matches")
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
    return EXIT_FAILURE;
  }

  // extract params
  std::string left_features_file = vm["left_features"].as<std::string>();
  std::string right_features_file = vm["right_features"].as<std::string>();
  double max_y_diff = vm["max_y_diff"].as<double>();
  double max_angle_diff = vm["max_angle_diff"].as<double>();
  int max_size_diff = vm["max_size_diff"].as<int>();
  double matching_threshold = vm["matching_threshold"].as<double>();

  // load key points and descriptors
  std::vector<cv::KeyPoint> key_points_left;
  cv::Mat descriptors_left;
  cv::FileStorage fsl(left_features_file, cv::FileStorage::READ);
  cv::read(fsl["key_points"], key_points_left);
  fsl["descriptors"] >> descriptors_left;

  std::vector<cv::KeyPoint> key_points_right;
  cv::Mat descriptors_right;
  cv::FileStorage fsr(right_features_file, cv::FileStorage::READ);
  cv::read(fsr["key_points"], key_points_right);
  fsr["descriptors"] >> descriptors_right;

  std::cout << "Loaded " << key_points_left.size() << " descriptors for left image" << std::endl;
  std::cout << "Loaded " << key_points_right.size() << " descriptors for right image" << std::endl;

  // configure and perform matching
  feature_matching::StereoFeatureMatcher::Params params;
  params.max_y_diff = max_y_diff;
  params.max_angle_diff = max_angle_diff;
  params.max_size_diff = max_size_diff;

  feature_matching::StereoFeatureMatcher matcher;
  matcher.setParams(params);
  std::vector<cv::DMatch> matches;
  matcher.match(key_points_left, descriptors_left, key_points_right,
                descriptors_right, matching_threshold, matches);

  std::cout << "Found " << matches.size() << " matches." << std::endl;

  if (vm.count("output_matches_file"))
  {
    std::string filename = vm["output_matches_file"].as<std::string>();
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "left_features" << left_features_file;
    fs << "right_features" << right_features_file;
    fs << "max_y_diff" << max_y_diff;
    fs << "max_angle_diff" << max_angle_diff;
    fs << "max_size_diff" << max_size_diff;
    fs << "matching_threshold" << matching_threshold;
    // TODO make this work
    //fs << "matches" << matches;
  }
  return EXIT_SUCCESS;
}

