#ifndef _GRABCUT_H_
#define _GRABCUT_H_

#include "opencv2/imgproc/imgproc.hpp"
#include "gaussian.hpp"

using namespace cv;

void cg_grabCut( const Mat& img, Mat& mask, Rect rect, 
                         Mat& bgdModel, Mat& fgdModel,
                         int iterCount, int mode = GC_EVAL );

class GMM_Component
{
public:
    double weight;
    Gaussian gauss;
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

#endif /* _GRABCUT_H_ */
