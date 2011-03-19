#ifndef _GMM_H_
#define _GMM_H_

#include <opencv2/core/core.hpp>
#include "gaussian.hpp"
#include "mm.hpp"

using namespace cv;

class GMM_Component
{
public:
    double weight;
    Gaussian gauss;
    GMM_Component();
    GMM_Component(cv::Mat component );

};

class GMM : public MM<GMM_Component> {
public:
    GMM():MM<GMM_Component>() {}
};

#endif /* _GMM_H_ */
