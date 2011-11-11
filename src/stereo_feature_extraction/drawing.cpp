#include <opencv2/core/core.hpp>

#include <feature_extraction/drawing.h>

#include "stereo_feature_extraction/drawing.h"
#include "stereo_feature_extraction/stereo_feature.h"

namespace stereo_feature_extraction
{
    void drawStereoFeatures(cv::Mat& canvas, const cv::Mat& left_image,
            const cv::Mat& right_image,
            const std::vector<StereoFeature>& stereo_features)
    {
        // create one big image
        assert(left_image.type() == right_image.type());
        int rows = std::max(left_image.rows, right_image.rows);
        canvas.create(rows, left_image.cols + right_image.cols, 
                left_image.type());
        cv::Mat left_canvas_hdr(canvas, cv::Range(0, left_image.rows),
                cv::Range(0, left_image.cols));
        cv::Mat right_canvas_hdr(canvas, 
                cv::Range(0, right_image.rows),
                cv::Range(left_image.cols, left_image.cols + right_image.cols));
        left_image.copyTo(left_canvas_hdr);
        right_image.copyTo(right_canvas_hdr);

        // paint stereo features
        for (size_t i = 0; i < stereo_features.size(); ++i)
        {
            feature_extraction::drawKeyPoint(left_canvas_hdr, stereo_features[i].key_point_left);
            feature_extraction::drawKeyPoint(right_canvas_hdr, stereo_features[i].key_point_right);
            cv::Point p1 = stereo_features[i].key_point_left.pt;
            cv::Point p2 = stereo_features[i].key_point_right.pt;
            p2.x += left_image.cols;
            cv::line(canvas, p1, p2, cv::Scalar(0, 255, 0), 1); 
        }
    }
}
