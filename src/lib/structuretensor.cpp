#include "structuretensor.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

StructureTensorImage::StructureTensorImage(const cv::Mat& image, double sigma)
{
    // Compute Image gradient in both directions for each channel
    cv::Mat xgrad;
    cv::Sobel(image, xgrad, CV_64F, 1, 0, 3);

    cout << "Sobel.type() = " << xgrad.type() << endl;

    cv::Mat ygrad;
    cv::Sobel(image, ygrad, CV_64F, 0, 1, 3);


    // compute structure tensor for each pixel
    cv::Mat stmat(image.rows, image.cols, CV_64FC3, Scalar(0,0,0));

    cout << "stmat.type() = " << stmat.type() << endl;
    cv::Point p;
    for( p.y = 0; p.y < stmat.rows; p.y++ )
    {
        for( p.x = 0; p.x < stmat.cols; p.x++ )
        {
            for(int c=0;c<image.channels();c++)
            {
                stmat.at<Vec3d>(p)[0] += xgrad.at<Vec3d>(p)[c]*xgrad.at<Vec3d>(p)[c];
                stmat.at<Vec3d>(p)[1] += ygrad.at<Vec3d>(p)[c]*ygrad.at<Vec3d>(p)[c];
                stmat.at<Vec3d>(p)[2] += xgrad.at<Vec3d>(p)[c]*ygrad.at<Vec3d>(p)[c];
            }
        }
    }

    // average on some of the pixels <- size of the window = scale?
    int ksize = ((int)(4*sigma)) * 2 + 1;
    GaussianBlur(stmat, blurredstmat, Size(ksize, ksize), sigma);

//    blurredstmat = stmat;

    tensors.reserve(stmat.rows * stmat.cols);
    for( p.y = 0; p.y < stmat.rows; p.y++ )
    {
        for( p.x = 0; p.x < stmat.cols; p.x++ )
        {
            tensors.push_back( StructureTensor( blurredstmat.at<Vec3d>(p) ) );
        }
    }
}

cv::Mat StructureTensorImage::getImage() {
    return blurredstmat;
}

StructureTensor::StructureTensor(const cv::Vec3d& t) {
    st.create(2,2,CV_64FC1);
    st.at<double>(0,0) = t[0];
    st.at<double>(1,1) = t[1];
    st.at<double>(0,1) = t[2];
    st.at<double>(1,0) = t[2];
}

cv::Mat StructureTensor::getMatrix() {
    return st;
}


double distance(StructureTensor& stl, StructureTensor& str)
{
    Mat l = stl.getMatrix();
    Mat r = str.getMatrix();
    Mat l_inv; 
    invert( l , l_inv);
    Mat r_inv;
    invert( r, r_inv );

    double res = sqrt( (trace(l_inv * r + r_inv * l)[0] - 4.0)/4.0 );

    return res;
}

StructureTensor mean(std::vector<StructureTensor>& list){

    Mat A(2,2,CV_64FC1,Scalar(0));
    Mat B(2,2,CV_64FC1,Scalar(0));

    for(int i=0; i<list.size(); i++)
    {
        Mat st = list[0].getMatrix();
        A += st;
        Mat st_inv;
        invert(st, st_inv);
        B += st_inv;
    }

    Mat B_sqrt;
    sqrt(B, B_sqrt);

    Mat B_inv;
    invert(B, B_inv);

    Mat B_inv_sqrt;
    invert(B_inv, B_inv_sqrt);

    Mat BAB = B_sqrt * A * B_sqrt;

    Mat BAB_sqrt;
    invert(BAB, BAB_sqrt);

    Mat res = B_inv_sqrt * BAB_sqrt * B_inv_sqrt;
}


