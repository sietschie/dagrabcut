#ifndef _SHARED_H_
#define _SHARED_H_

#include <string>
#include <opencv2/core/core.hpp>

void readImageAndMask(std::string filename, cv::Mat& image, cv::Mat& mask);

#endif /* _SHARED_H_ */

