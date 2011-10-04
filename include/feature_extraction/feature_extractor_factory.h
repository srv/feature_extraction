#ifndef FEATURE_EXTRACTOR_FACTORY_H
#define FEATURE_EXTRACTOR_FACTORY_H

#include "feature_extractor.h"

namespace feature_extraction
{

/**
* \class FeatureExtractorFactory
*/
class FeatureExtractorFactory
{

  public:

    /**
    * Factory method
    * \param name the name of the extractor to create
    * \return pointer to the created extractor. If no extractor with
    *         given name could be created, the returned pointer is invalid.
    */
    static FeatureExtractor::Ptr create(const std::string& name);
};

} 

#endif


