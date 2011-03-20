#ifndef _MM_H_
#define _MM_H_

#include <opencv2/core/core.hpp>
#include "gaussian.hpp"

template <class Component>
class MM {
public:
    static const int dim = 3;

    MM();
    double operator()( const cv::Vec3d color ) const;
    double operator()( int ci, const cv::Vec3d color ) const;
    int whichComponent( const cv::Vec3d color ) const;

    void initLearning();
    void addSample( int ci, const cv::Vec3d color );
    void endLearning();
    cv::Mat getModel();
    void setModel(const cv::Mat &model);
    void setComponentsCount(int size);
    int getComponentsCount();

    double KLdiv(const MM& rhs);
    double KLsym(MM& rhs);

protected:
    std::vector<Component> components;
    std::vector<std::vector<cv::Vec3b> > samples;

};

#include "mm.cpp"

#endif /* _MM_H_ */
