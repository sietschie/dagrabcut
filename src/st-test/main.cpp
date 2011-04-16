#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../lib/structuretensor.hpp"

#include <iostream>

using namespace std;
using namespace cv;


class GCApplication
{
public:
    void setImageAndWinName( const Mat& _image, const string& _winName );
    void showImage() const;
private:

    string winName;
    Mat image;
};

void GCApplication::showImage() const
{
    if( image.empty() || winName.empty() )
        return;

    imshow( winName, image );
}

void GCApplication::setImageAndWinName( const Mat& _image, const string& _winName  )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = _image;
    winName = _winName;
}


GCApplication gcapp;

int main( int argc, char** argv )
{
    if( argc!=2 )
    {
        return 1;
    }
    string filename = argv[1];
    if( filename.empty() )
    {
    	cout << "\nDurn, couldn't read in " << argv[1] << endl;
        return 1;
    }
    Mat image = imread( filename, 1 );
    if( image.empty() )
    {
        cout << "\n Durn, couldn't read image filename " << filename << endl;
    	return 1;
    }

    const string winName = "image";
    cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );

    gcapp.setImageAndWinName( image, winName );
    gcapp.showImage();

    int c = cvWaitKey(0);

    for(double sigma = 1.0; sigma <= 4.0; sigma += 1.0)
    {
        StructureTensorImage sti(image, sigma);
        Mat stimage = sti.getImage();

        cout << "image.at<Vec3d>(5,5) = " << image.at<Vec3d>(5,5)[1] << endl;
        cout << "stimage.at<Vec3d>(5,5) = " << stimage.at<Vec3d>(5,5)[1] << endl;

        //gcapp.setImageAndWinName( stimage, winName );
        //gcapp.showImage();
        //c = cvWaitKey(0);

        Mat transformed;
        stimage.convertTo(transformed, CV_8UC3);

        vector<Mat> splitted;
        split(stimage, splitted);

        vector<Mat> result_splitted;
        result_splitted.resize(splitted.size());

        for(int ch=0;ch<splitted.size(); ch++)
        {
            double minVal, maxVal;
            minMaxLoc(splitted[ch], &minVal, &maxVal);
            cout << "minVal = " << minVal << "  maxVal = " << maxVal << endl;

            double alpha = (1.0 / (double)(maxVal - minVal)) * 128;
            double beta = minVal * alpha;
            cout << "alpha = " << alpha << " beta = " << beta << endl;
            splitted[ch].convertTo(result_splitted[ch], CV_8UC1, alpha, beta);
        }

        Mat result;
        merge(result_splitted, result);

        gcapp.setImageAndWinName( result, winName );
        gcapp.showImage();
        c = cvWaitKey(0);

/*        Mat labels;
        const vector<StructureTensor> &tensors = sti.getAllTensors();
        vector<StructureTensor> centers;

        kmeans(tensors, 5, TermCriteria( CV_TERMCRIT_ITER, 10, 0.0), 3, labels, centers);

        Mat clusteredimage = image.clone();

        Point p;
        for(p.x = 0; p.x<clusteredimage.cols;p.x++)
        for(p.y = 0; p.y<clusteredimage.rows;p.y++)
        {
            int label = labels.at<int>( p.y * clusteredimage.cols + p.x );
            if(label == 0) {
                clusteredimage.at<Vec3b>(p)[0] = 255;
                clusteredimage.at<Vec3b>(p)[1] = 0;
                clusteredimage.at<Vec3b>(p)[2] = 0;
            }
            if(label == 1) {
                clusteredimage.at<Vec3b>(p)[0] = 0;
                clusteredimage.at<Vec3b>(p)[1] = 255;
                clusteredimage.at<Vec3b>(p)[2] = 0;
            }
            if(label == 2) {
                clusteredimage.at<Vec3b>(p)[0] = 0;
                clusteredimage.at<Vec3b>(p)[1] = 0;
                clusteredimage.at<Vec3b>(p)[2] = 255;
            }
            if(label == 3) {
                clusteredimage.at<Vec3b>(p)[0] = 255;
                clusteredimage.at<Vec3b>(p)[1] = 255;
                clusteredimage.at<Vec3b>(p)[2] = 0;
            }
            if(label == 4) {
                clusteredimage.at<Vec3b>(p)[0] = 0;
                clusteredimage.at<Vec3b>(p)[1] = 255;
                clusteredimage.at<Vec3b>(p)[2] = 255;
            }
        }
        gcapp.setImageAndWinName( clusteredimage, winName );
        gcapp.showImage();
        c = cvWaitKey(0);*/
   }
 

    cvDestroyWindow( winName.c_str() );
    return 0;
}
