#ifndef SMART_SURF_DESCRIPTOR_EXTRACTOR_H
#define SMART_SURF_DESCRIPTOR_EXTRACTOR_H

#include "feature_extraction/descriptor_extractor.h"
#include "feature_extraction/smart_surf.h"

namespace feature_extraction
{

/**
* \class SmartSurfExtractor
* \brief Extractor that uses our own implementation of SURF
*/
class SmartSurfDescriptorExtractor : public DescriptorExtractor
{

public:
  /**
  * Constructs a surf extractor with default parameters.
  */
  SmartSurfDescriptorExtractor();

  // see base class for documentation
  void extract(const cv::Mat& image, 
               std::vector<cv::KeyPoint>& key_points,
               cv::Mat& descriptors);

private:

  visual_odometry::SmartSurf smart_surf_;

};

}

#endif

