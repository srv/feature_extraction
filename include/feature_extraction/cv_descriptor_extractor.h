#ifndef CV_DESCRIPTOR_EXTRACTOR_H
#define CV_DESCRIPTOR_EXTRACTOR_H

#include <opencv2/features2d/features2d.hpp>

#include "descriptor_extractor.h"

namespace feature_extraction
{

/**
 * \class DescriptorExtractor
 * \brief Wrapper for OpenCV descriptor extractors
 */
class CvDescriptorExtractor : public DescriptorExtractor
{

public:

  /**
   * Create descriptor extractor for given name
   * \param name see http://opencv.itseez.com/modules/features2d/doc/common_interfaces_of_descriptor_extractors.html#descriptorextractor-create
   * Throws an exception if descriptor extractor with given name does not exist
   */
  CvDescriptorExtractor(const std::string& name);

  /**
   * Virtual destructor
   */
  virtual ~CvDescriptorExtractor() {}

  /**
   * Extraction interface.
   * \param image the input image.
   * \param key_points input vector of computed key points, may be modified
   * \param descriptors output matrix for descriptors
   */
  virtual void extract(const cv::Mat& image,
                       std::vector<cv::KeyPoint>& key_points,
                       cv::Mat& descriptors);

protected:

  cv::Ptr<cv::DescriptorExtractor> cv_extractor_;

  static bool nonfree_module_initialized_;

};

}

#endif


