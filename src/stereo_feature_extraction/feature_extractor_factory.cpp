#include "feature_extractor_factory.h"

#include "cv_surf_extractor.h"
#include "surf_extractor.h"

using namespace stereo_feature_extraction;

FeatureExtractor::Ptr FeatureExtractorFactory::create(const std::string& name)
{
    if (name == "CvSURF")
    {
        return FeatureExtractor::Ptr(new CvSurfExtractor());
    }
    else if (name == "SURF")
    {
        return FeatureExtractor::Ptr(new SurfExtractor());
    }
    else
    {
        FeatureExtractor::Ptr ptr;
        return ptr;
    }
}

