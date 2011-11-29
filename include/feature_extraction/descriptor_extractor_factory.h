#ifndef DESCRIPTOR_EXTRACTOR_FACTORY_H
#define DESCRIPTOR_EXTRACTOR_FACTORY_H

#include "descriptor_extractor.h"

namespace feature_extraction
{

class DescriptorExtractorFactory
{

public:

  /**
   * \param name the name of the extractor to create
   * \return pointer to the created extractor. If no extractor with
   *         given name could be created, the returned pointer is invalid.
   */
  static DescriptorExtractor::Ptr create(const std::string& name);

};

}

#endif


