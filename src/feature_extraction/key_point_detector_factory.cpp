#include "feature_extraction/exceptions.h"

#include "feature_extraction/key_point_detector_factory.h"

#include "feature_extraction/cv_key_point_detector.h"
#include "feature_extraction/smart_surf_key_point_detector.h"

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

