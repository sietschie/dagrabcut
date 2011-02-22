#ifndef _GRABCUT_H_
#define _GRABCUT_H_

#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

void cg_grabCut( const Mat& img, Mat& mask, Rect rect, 
                         Mat& bgdModel, Mat& fgdModel,
                         int iterCount, int mode = GC_EVAL );


class GMM
{
public:
    static const int componentsCount = 5;

    GMM( Mat& _model );
    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;
    int whichComponent( const Vec3d color ) const;

    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();

private:
    void calcInverseCovAndDeterm( int ci );
    Mat model;
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][3][3];
    double covDeterms[componentsCount];

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};

#endif /* _GRABCUT_H_ */
