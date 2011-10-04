#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <cassert>
#include <boost/shared_ptr.hpp>

#include <opencv2/features2d/features2d.hpp>

#include "key_point.h"

namespace feature_extraction
{

/**
* \class FeatureExtractor
* \brief Common interface for feature extractors
*/
class FeatureExtractor
{

  public:

    static const int DEFAULT_MAX_NUM_KEY_POINTS = 200;

    /**
    * Default constructor
    */
    FeatureExtractor() : max_num_key_points_(DEFAULT_MAX_NUM_KEY_POINTS)
    {
    }

    /**
    * Virtual destructor
    */
    virtual ~FeatureExtractor() {};

    /**
    * Extraction interface.
    * \param image the input image.
    * \param key_points output vector for computed key points
    * \param descriptors output matrix for descriptors 
    * \param roi region of interest where the features have to be extracted
    */
    virtual void extract(const cv::Mat& image, 
            std::vector<KeyPoint>& key_points,
            cv::Mat& descriptors, const cv::Rect& roi = cv::Rect()) = 0;

    /**
    * set the maximum number of key points to extract
    */
    inline void setMaxNumKeyPoints(int num)
    {
        assert(num > 0);
        max_num_key_points_ = num;
    }

    /**
    * \return maximum number of key points
    */
    inline int getMaxNumKeyPoints()
    {
        return max_num_key_points_;
    }

    /// Ptr type
    typedef boost::shared_ptr<FeatureExtractor> Ptr;

  protected:

    int max_num_key_points_;
};

}

#endif


