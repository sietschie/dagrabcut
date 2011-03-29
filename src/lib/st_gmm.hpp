#ifndef _ST_GMM_H_
#define _ST_GMM_H_

#include <opencv2/core/core.hpp>
#include "st_gaussian.hpp"
#include "st_mm.hpp"

using namespace cv;

class ST_GMM_Component
{
public:
    double weight;
    ST_Gaussian gauss;
    ST_GMM_Component();
    ST_GMM_Component(const cv::Mat &component );

};

class ST_GMM : public ST_MM<ST_GMM_Component> {
public:
    ST_GMM():ST_MM<ST_GMM_Component>() {}
};

#endif /* _ST_GMM_H_ */
