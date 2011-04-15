#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

Mat scaleMatrix(Mat image);
void printMatrix(Mat &m);

void thomas(const Mat &a, const Mat &b, const Mat &c, const Mat &d, Mat &x) //a -> Hauptdiagonale, b-> 1. obere Nebendiagonale, c-> 1. untere Nebendiagonale, d -> Mat(c,a,b) * x = d
{
    double n = a.cols;

    //Mat m = Mat::zeros(a.size(), a.type());
    Mat m(a.size(), a.type(), Scalar(0,0));
    //Mat l = Mat::zeros(b.size(), b.type()) * 1.0;
    Mat l(b.size(), b.type(), Scalar(0,0));
    //Mat y = Mat::zeros(c.size(), c.type());
    Mat y(d.size(), d.type(), Scalar(0.0));


    x.create(a.size(), a.type());
    //Mat x(a);

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
    
    //cout << "Matrix l: \n";
    //printMatrix(l);

    //return x;
}

void aosiso(const Mat &image, const Mat &d, double t, Mat &result) // image -> image, d -> difusivity, t -> timestep
{
    //cout << "image.type() = " << image.type() << endl;
    assert(image.type() == CV_64FC1);

    //rows
    Mat result_rows;
    {
        Point pt;
        Mat q( image.rows - 1, image.cols, CV_64FC1); //image.type() );

        //cout << "q.rows = " << q.rows << "  q.cols = " << q.cols << endl;
    
        for(pt.y = 0; pt.y<q.rows; pt.y++)
        for(pt.x = 0; pt.x<q.cols; pt.x++)
        {
            //cout << "(" << pt.x << "," << pt.y << ") = " << d.at<double>(pt.y,pt.x) << " + " << d.at<double>(pt.y + 1,pt.x) << endl;
            q.at<double>(pt) = d.at<double>(pt.y,pt.x) + d.at<double>(pt.y+1,pt.x);
        }

        //cout << "image.cols = " << image.cols << "   image.rows = " << image.rows << "  image.type() = " << image.type() << endl;

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

        //imshow( "image", scaleMatrix(a) * 14.0 );
        //cvWaitKey(0);

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
            //cout << "(" << pt.x << "," << pt.y << ") = " << d.at<double>(pt.y,pt.x) << " + " << d.at<double>(pt.y + 1,pt.x) << endl;
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

    //cout << "aosiso ende..." << endl;

}


int main(int argc, char** argv)
{
    string filename = argv[1];
    Mat image = imread( filename, 1 );

    const string winName = "image";
    cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );
    imshow( winName, image );
    //cvWaitKey(0);

    //Mat result = image.clone();

    Mat result;
    image.convertTo(result, CV_64FC3);


    //printMatrix(result);

    for(int i=0;i<10;i++) {
        double sigma = 2;
        int ksize = ((int)(4*sigma)) * 2 + 1;

        Mat blurred_image;

        GaussianBlur(result, blurred_image, Size(ksize, ksize), sigma);

        cv::Mat xgrad;
        cv::Sobel(blurred_image, xgrad, CV_64F, 1, 0, 3);

        cv::Mat ygrad;
        cv::Sobel(blurred_image, ygrad, CV_64F, 0, 1, 3);

        Mat gradgrad = xgrad.mul(xgrad) + ygrad.mul(ygrad);
        Mat grad;
        sqrt(gradgrad, grad);

        //printMatrix(grad);

//        imshow( winName, scaleMatrix(grad) );
        //imshow( winName, grad / 256.0 );
        //cvWaitKey(0);

        double parameter_p = 0.6;
//        double parameter_p = 1.0;
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

        //imshow( winName, difusivity );
        //cvWaitKey(0);

        //cout << "1 result.cols = " << result.cols << "  result.rows = " << result.rows << "  result.type() = " << result.type() << "  channel = " << result.channels() << endl;

        vector<Mat> splitted;

        split(result, splitted);

        vector<Mat> result_splitted;
        result_splitted.resize(splitted.size());

        for(int ch=0;ch<splitted.size(); ch++)
        {
            //Mat aos_input_image = splitted[0];
            //imshow( winName, aos_input_image );
            //cvWaitKey(0);
            //Mat result;
            aosiso(splitted[ch], difusivity, 100, result_splitted[ch]);
            //aosiso(aos_input_image, difusivity, 1.0, result);
            //cout << "3 result.cols = " << result_splitted[ch].cols << "  result.rows = " << result_splitted[ch].rows << "  result.type() = " << result_splitted[ch].type() << "  channel = " << result_splitted[ch].channels() << endl;
        }

        merge(result_splitted, result);

        //imshow( winName, scaleMatrix(result) );
        //cvWaitKey(0);



        /*cout << "2 result.cols = " << result.cols << "  result.rows = " << result.rows << "  result.type() = " << result.type() << "  channel = " << result.channels() << endl;

        cout << "CV_64FC1 = " << CV_64FC1 << endl;
        cout << "CV_64FC3 = " << CV_64FC3 << endl;

        cout << "CV_32FC1 = " << CV_32FC1 << endl;
        cout << "CV_32FC3 = " << CV_32FC3 << endl;*/

        
        //result = scaleMatrix(result);
    }

    imshow( winName, result / 256.0 );
    cvWaitKey(0);



    double sigma = 10;
    int ksize = ((int)(4*sigma)) * 2 + 1;

    Mat blurred_image;

    GaussianBlur(image, blurred_image, Size(ksize, ksize), sigma);

    imshow( winName, blurred_image );
    cvWaitKey(0);

    return 0;
}

Mat scaleMatrix(Mat image)
{
    vector<Mat> splitted;
    split(image, splitted);

    vector<Mat> result_splitted;
    result_splitted.resize(splitted.size());

    for(int ch=0;ch<splitted.size(); ch++)
    {
        double minVal, maxVal;
        minMaxLoc(splitted[ch], &minVal, &maxVal);
        cout << "minVal = " << minVal << "  maxVal = " << maxVal << endl;

        //maxVal = 255; minVal = 0;

        double alpha = (1.0 / (double)(maxVal - minVal)) * 256;
        double beta = minVal * alpha;

        cout << "alpha = " << alpha << " beta = " << beta << endl;
        splitted[ch].convertTo(result_splitted[ch], CV_8UC1, alpha, beta);


        //imshow( "image", result_splitted[ch] );
        //cvWaitKey(0);
    }

    Mat result;
    merge(result_splitted, result);
    return result;
}

void printMatrix(Mat &m)
{
    cout << "print matrix: \n";
    Point pt;
    for(pt.y = 0; pt.y < m.rows; pt.y++)
    {
        for(pt.x = 0; pt.x<m.cols; pt.x++)
        {
            cout << "\t" << m.at<Vec3d>(pt)[0];
        }
        cout << "\n";
    }
}

