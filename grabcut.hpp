#ifndef _GRABCUT_H_
#define _GRABCUT_H_

#include "opencv2/imgproc/imgproc.hpp"
#include "gaussian.hpp"
#include "gmm.hpp"

using namespace cv;

void cg_grabCut( const Mat& img, Mat& mask, Rect rect, 
                         Mat& bgdModel, Mat& fgdModel,
                         int iterCount, int mode = GC_EVAL );

#endif /* _GRABCUT_H_ */
