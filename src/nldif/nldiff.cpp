#include "nldiff.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

void aosiso(const Mat &image, const Mat &d, double t, Mat &result); // image -> image, d -> difusivity, t -> timestep


cv::Mat nldiff(const cv::Mat &src, double stepsize, int numsteps, double sigma, double parameter_p)
{
    Mat intermediate;
    src.convertTo(intermediate, CV_64FC3);

    for(int i=0;i<numsteps;i++) {

        // blurr image
        int ksize = ((int)(4*sigma)) * 2 + 1;
        Mat blurred_image;
        GaussianBlur(intermediate, blurred_image, Size(ksize, ksize), sigma);

        // compute gradient
        cv::Mat xgrad;
        cv::Sobel(blurred_image, xgrad, CV_64F, 1, 0, 3);

        cv::Mat ygrad;
        cv::Sobel(blurred_image, ygrad, CV_64F, 0, 1, 3);

        Mat gradgrad = xgrad.mul(xgrad) + ygrad.mul(ygrad);
        Mat grad;
        sqrt(gradgrad, grad);


        // compute difusivity matrix
        double eps = 1e-3;

        Mat difusivity(grad.size(), CV_64FC1);
        Point p;
        for(p.y=0;p.y<grad.rows;p.y++)
        for(p.x=0;p.x<grad.cols;p.x++)
        {
            double dist = 0.0;
            for(int c = 0; c<grad.channels();c++)
                dist += grad.at<Vec3d>(p)[c] * grad.at<Vec3d>(p)[c];
            dist = sqrt(dist);
            difusivity.at<double>(p) = 1.0 / (pow(dist,parameter_p) + eps);
        }

        // diffuse each channel independently
        vector<Mat> splitted;
        split(intermediate, splitted);

        vector<Mat> result_splitted;
        result_splitted.resize(splitted.size());

        for(int ch=0;ch<splitted.size(); ch++)
        {
            aosiso(splitted[ch], difusivity, stepsize, result_splitted[ch]);
        }
        merge(result_splitted, intermediate);
   }

    Mat result;
    intermediate.convertTo(result, CV_8UC3);

    return result;
}

void thomas(const Mat &a, const Mat &b, const Mat &c, const Mat &d, Mat &x) //a -> Hauptdiagonale, b-> 1. obere Nebendiagonale, c-> 1. untere Nebendiagonale, d -> Mat(c,a,b) * x = d
{
    double n = a.cols;

    Mat m(a.size(), a.type(), Scalar(0,0));
    Mat l(b.size(), b.type(), Scalar(0,0));
    Mat y(d.size(), d.type(), Scalar(0.0));


    x.create(a.size(), a.type());

    // 1. LU decomposition

    for(int col = 0; col<a.cols; col++)
    {
        m.at<double>(0, col) = a.at<double>(0,col);
        y.at<double>(0, col) = d.at<double>(0,col);
    }

    for(int i=1; i<n;i++)
    {
        int i_1 = i-1;
        for(int col = 0; col<a.cols; col++)
        {
            l.at<double>(i_1,col) = c.at<double>(i_1,col) / m.at<double>(i_1, col);
            m.at<double>(i, col) = a.at<double>(i, col) - l.at<double>(i_1, col) * b.at<double>(i_1, col);

            y.at<double>(i, col) = d.at<double>(i, col) - l.at<double>(i_1, col) * y.at<double>(i_1, col);
        }
    }

    for(int col = 0; col<a.cols; col++)
    {
        x.at<double>(n-1, col) = y.at<double>(n-1,col) / m.at<double>(n-1, col);
    }

    for(int i=n-2; i>=0;i--)
    {
        for(int col = 0; col<a.cols; col++)
        {
            x.at<double>(i,col) = ( y.at<double>(i,col) - b.at<double>(i, col) * x.at<double>(i+1, col)) / m.at<double>(i, col);
        }
    }
}

void aosiso(const Mat &image, const Mat &d, double t, Mat &result) // image -> image, d -> difusivity, t -> timestep
{
    assert(image.type() == CV_64FC1);

    //rows
    Mat result_rows;
    {
        Point pt;
        Mat q( image.rows - 1, image.cols, CV_64FC1);

        for(pt.y = 0; pt.y<q.rows; pt.y++)
        for(pt.x = 0; pt.x<q.cols; pt.x++)
        {
            q.at<double>(pt) = d.at<double>(pt.y,pt.x) + d.at<double>(pt.y+1,pt.x);
        }

        Mat p( image.size(), CV_64FC1);
        for(pt.y = 0; pt.y<image.rows; pt.y++)
        for(pt.x = 0; pt.x<image.cols; pt.x++)
        {
            if( pt.y == 0)
            {
                p.at<double>(pt) = q.at<double>(0,pt.x);
            } else if ( pt.y == image.rows-1) {
                p.at<double>(pt) = q.at<double>(pt.y-1,pt.x);
            }
            else {
                p.at<double>(pt) = q.at<double>(pt.y-1,pt.x) + q.at<double>(pt.y,pt.x);
            }
        }

        Mat a = 1 + t * p;

        Mat b = -t * q;

        thomas(a,b,b,image, result_rows);
    }

    //columns
    Mat result_columns;
    {
        Point pt;
        Mat q( image.rows, image.cols - 1, CV_64FC1 );
        for(pt.x = 0; pt.x<image.cols-1; pt.x++)
        for(pt.y = 0; pt.y<image.rows; pt.y++)
        {
            q.at<double>(pt) = d.at<double>(pt.y,pt.x) + d.at<double>(pt.y,pt.x+1);
        }

        Mat p( image.size(), CV_64FC1 );
        for(pt.x = 0; pt.x<image.cols; pt.x++)
        for(pt.y = 0; pt.y<image.rows; pt.y++)
        {
            if( pt.x == 0)
            {
                p.at<double>(pt) = q.at<double>(pt.y,0);
            } else if ( pt.x == image.cols-1) {
                p.at<double>(pt) = q.at<double>(pt.y,pt.x-1);
            }
            else {
                p.at<double>(pt) = q.at<double>(pt.y,pt.x-1) + q.at<double>(pt.y,pt.x);
            }
        }

        Mat p_transp; transpose(p, p_transp);
        Mat a = 1 + t * p_transp;

        Mat q_transp; transpose(q, q_transp);
        Mat b = -t * q_transp;

        Mat image_transp; transpose(image, image_transp);

        Mat result_columns_transp;
        thomas(a,b,b,image_transp, result_columns_transp);

        transpose(result_columns_transp, result_columns);
    }

    result = (result_rows + result_columns) / 2.0;
}
