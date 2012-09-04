#include "feature_extraction/exceptions.h"

#include "feature_extraction/key_point_detector_factory.h"

#include "feature_extraction/cv_key_point_detector.h"
#include "feature_extraction/smart_surf_key_point_detector.h"

std::vector<std::string> feature_extraction::KeyPointDetectorFactory::getDetectorNames()
{
  std::vector<std::string> names;
  names.push_back("SmartSURF");
  names.push_back("CvFAST");
  names.push_back("CvSTAR");
  names.push_back("CvSIFT");
  names.push_back("CvSURF");
  names.push_back("CvORB");
  names.push_back("CvMSER");
  names.push_back("CvGFTT");
  names.push_back("CvHARRIS");
  // names.push_back("CvDense"); // silly "detector"
  // names.push_back("CvSimpleBlob");
  return names;
}

feature_extraction::KeyPointDetector::Ptr 
feature_extraction::KeyPointDetectorFactory::create(const std::string& name)
{
  if (name.substr(0, 2) == "Cv")
  {
    return KeyPointDetector::Ptr(new CvKeyPointDetector(name.substr(2)));
  }
  else if (name == "SmartSURF")
  {
    return KeyPointDetector::Ptr(new SmartSurfKeyPointDetector());
  }
  else
  {
    throw InvalidKeyPointDetectorName(name);
    return KeyPointDetector::Ptr();
  }
}

