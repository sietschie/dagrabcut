#ifndef _GMM_H_
#define _GMM_H_

#include <opencv2/core/core.hpp>
#include "gaussian.hpp"

using namespace cv;

class GMM_Component
{
public:
    double weight;
    Gaussian gauss;
    GMM_Component();
    GMM_Component(cv::Mat component );

};

class GMM
{
public:
    static const int dim = 3;

    GMM( Mat& _model, int _componensCount = 5);
    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;
    int whichComponent( const Vec3d color ) const;

    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();
    Mat getModel();
    int getComponentsCount();

private:
    Mat model;
    int componentsCount;
    std::vector<GMM_Component*> components;
    void calcInverseCovAndDeterm( int ci );
    std::vector<std::vector<cv::Vec3b> > samples;
    Mat updateModel();
};

#endif /* _GMM_H_ */
