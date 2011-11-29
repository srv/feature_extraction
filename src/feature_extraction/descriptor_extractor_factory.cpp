#include "feature_extraction/exceptions.h"

#include "feature_extraction/descriptor_extractor_factory.h"

#include "feature_extraction/cv_descriptor_extractor.h"
#include "feature_extraction/smart_surf_descriptor_extractor.h"

feature_extraction::DescriptorExtractor::Ptr 
feature_extraction::DescriptorExtractorFactory::create(const std::string& name)
{
  if (name.substr(0, 2) == "Cv")
  {
    return DescriptorExtractor::Ptr(new CvDescriptorExtractor(name.substr(2)));
  }
  else if (name == "SmartSURF")
  {
    return DescriptorExtractor::Ptr(new SmartSurfDescriptorExtractor());
  }
  else
  {
    throw InvalidDescriptorExtractorName(name);
    return DescriptorExtractor::Ptr();
  }
}

