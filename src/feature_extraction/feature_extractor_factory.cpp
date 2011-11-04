#include "feature_extraction/feature_extractor_factory.h"

#include "feature_extraction/cv_surf_extractor.h"
#include "feature_extraction/smart_surf_extractor.h"

using namespace feature_extraction;

FeatureExtractor::Ptr FeatureExtractorFactory::create(const std::string& name)
{
    if (name == "CvSURF")
    {
        return FeatureExtractor::Ptr(new CvSurfExtractor());
    }
    else if (name == "SmartSURF")
    {
        return FeatureExtractor::Ptr(new SmartSurfExtractor());
    }
    else
    {
        FeatureExtractor::Ptr ptr;
        return ptr;
    }
}

