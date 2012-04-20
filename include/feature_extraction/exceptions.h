#include <stdexcept>

namespace feature_extraction
{

class InvalidKeyPointDetectorName : public std::runtime_error
{
public:
  InvalidKeyPointDetectorName(const std::string& name) : 
    std::runtime_error("invalid detector " + name )
  {}
};

class InvalidDescriptorExtractorName : public std::runtime_error
{
public:
  InvalidDescriptorExtractorName(const std::string& name) : 
    std::runtime_error("invalid extractor " + name )
  {}
};

}

