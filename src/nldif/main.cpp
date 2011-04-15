#include "opencv2/highgui/highgui.hpp"

#include <iostream>

#include "nldiff.hpp"

using namespace std;
using namespace cv;

Mat scaleMatrix(Mat image);
void printMatrix(Mat &m);

int main(int argc, char** argv)
{
    string filename = argv[1];
    Mat image = imread( filename, 1 );

    const string winName = "image";
    cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );
    imshow( winName, image );
    cvWaitKey(0);

    Mat result = nldiff(image, 100.0, 10, 2, 0.6);

    imshow( winName, result );
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

