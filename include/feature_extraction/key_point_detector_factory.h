#ifndef KEY_POINT_DETECTOR_FACTORY_H
#define KEY_POINT_DETECTOR_FACTORY_H

#include "key_point_detector.h"

namespace feature_extraction
{

class KeyPointDetectorFactory
{

public:

  /**
   * \param name the name of the detector to create
   * \return pointer to the created detector. If no detector with
   *         given name could be created, the returned pointer is invalid.
   */
  static KeyPointDetector::Ptr create(const std::string& name);

};

}

#endif


