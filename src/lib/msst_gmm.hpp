#ifndef _MSST_GMM_H_
#define _MSST_GMM_H_

#include <opencv2/core/core.hpp>
#include "st_gaussian.hpp"
#include "msst_mm.hpp"

using namespace cv;

class MSST_GMM_Component
{
public:
    double weight;
    MSST_Gaussian gauss;
    //MSST_GMM_Component();
    //MSST_GMM_Component(const cv::Mat &component );

};

class MSST_GMM : public MSST_MM<MSST_GMM_Component> {
public:
    MSST_GMM():MSST_MM<MSST_GMM_Component>() {}
};

#endif /* _MSST_GMM_H_ */
