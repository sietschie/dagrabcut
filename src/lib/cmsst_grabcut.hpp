#ifndef _GRABCUT_H_
#define _GRABCUT_H_

#include "opencv2/imgproc/imgproc.hpp"
#include "st_gaussian.hpp"
#include "msst_gmm.hpp"

using namespace cv;

void cg_cmsst_grabCut( const Mat& img, const MSStructureTensorImage& MSST_img, Mat& mask, Mat& initial_mask, Mat& initial_mask_color, Mat& initial_mask_msst, Rect rect,
                 Mat& bgdModel, Mat& fgdModel, Mat& MSST_bgdModel, Mat& MSST_fgdModel,
                 int iterCount, double &xi, int mode = GC_EVAL );

void cg_cmsst_interactive_grabCut( const MSStructureTensorImage& img, Mat& mask, Mat& initial_mask, Mat& initial_mask_color, Mat& initial_mask_msst, Rect rect,
             Mat& bgdModel, Mat& fgdModel, Mat& MSST_bgdModel, Mat& MSST_fgdModel,
             int iterCount, double &xi, int mode );


#endif /* _GRABCUT_H_ */
