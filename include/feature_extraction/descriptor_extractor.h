#ifndef DESCRIPTOR_EXTRACTOR_H
#define DESCRIPTOR_EXTRACTOR_H

#include <vector>

#include <boost/shared_ptr.hpp>

namespace cv
{
class KeyPoint;
class Mat;
}

namespace feature_extraction
{

/**
 * \class DescriptorExtractor
 * \brief Common interface for descriptor extractors
 */
class DescriptorExtractor
{

public:

  /**
   * Default constructor
   */
  DescriptorExtractor() {}

  /**
   * Virtual destructor
   */
  virtual ~DescriptorExtractor() {}

  /**
   * Extraction interface.
   * \param image the input image.
   * \param key_points input vector of computed key points, may be modified
   * \param descriptors output matrix for descriptors
   */
  virtual void extract(const cv::Mat& image,
                       std::vector<cv::KeyPoint>& key_points,
                       cv::Mat& descriptors) = 0;

  /// Ptr type
  typedef boost::shared_ptr<DescriptorExtractor> Ptr;
};

}

#endif


