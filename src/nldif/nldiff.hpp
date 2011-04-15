#ifndef _NLDIFF_H_
#define _NLDIFF_H_

#include <opencv2/core/core.hpp>

cv::Mat nldiff(const cv::Mat &src, double stepsize, int numsteps, double sigma = 2.0, double p = 0.6 );

#endif // _NLDIFF_H_
