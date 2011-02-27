#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "grabcut.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void help()
{
    cout << "Call:\n"
    		"./grabcut <image_name> <mask_name>\n"
        "\nHot keys: \n"
        "\tESC - quit the program\n"
        "\tn - next iteration\n" << endl;
}

void getBinMask( const Mat& comMask, Mat& binMask )
{
    if( comMask.empty() || comMask.type()!=CV_8UC1 )
        CV_Error( CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
        binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}

class GCApplication
{
public:
    void setImageAndWinName( const Mat& _image, const string& _winName, Mat& _mask );
    void showImage() const;
    int nextIter();
    int getIterCount() const { return iterCount; }
    Mat mask;
    Mat bgdModel, fgdModel;
private:
    const string* winName;
    const Mat* image;
    Mat input_mask;

    bool isInitialized;

    int iterCount;
};

void GCApplication::setImageAndWinName( const Mat& _image, const string& _winName, Mat& _mask  )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    winName = &_winName;
    //mask.create( image->size(), CV_8UC1);
    input_mask = _mask; // TODO: is it good to do it that way? or better pointer to mask?
    mask = input_mask.clone();
}

void GCApplication::showImage() const
{
    if( image->empty() || winName->empty() )
        return;

    Mat res;
    Mat binMask;
    if( !isInitialized )
        image->copyTo( res );
    else
    {
        getBinMask( mask, binMask );
        image->copyTo( res, binMask );
    }

    imshow( *winName, res );
}

int GCApplication::nextIter()
{
    int max_iterations = 100;
    Rect rect;

    if( isInitialized )
        cg_grabCut( *image, mask, rect, bgdModel, fgdModel, max_iterations );
    else
    {
        cg_grabCut( *image, mask, rect, bgdModel, fgdModel, max_iterations, GC_INIT_WITH_MASK );

        isInitialized = true;
    }
    iterCount += max_iterations;

    return iterCount;
}

GCApplication gcapp;

int main( int argc, char** argv )
{
    if( argc!=3 )
    {
    	help();
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
    string maskfilename = argv[2];
    if( maskfilename.empty() )
    {
    	cout << "\nDurn, couldn't read in " << argv[2] << endl;
        return 1;
    }
    Mat mask = imread( maskfilename, 1 );
    if( mask.empty() )
    {
        cout << "\n Durn, couldn't read mask filename " << maskfilename << endl;
    	return 1;
    }

    Mat transform(1, 3, DataType<double>::type);
    transform.at<double>(0,0) = 1.0 * GC_PR_FGD / (255 * 3);
    transform.at<double>(0,1) = 1.0 * GC_PR_FGD / (255 * 3);
    transform.at<double>(0,2) = 1.0 * GC_PR_FGD / (255 * 3);

    Mat mask2(mask.rows, mask.cols, DataType<uchar>::type);

    cv::transform(mask, mask2, transform);
    help();

    const string winName = "image";
    cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );

    gcapp.setImageAndWinName( image, winName, mask2 );
    gcapp.showImage();

    cvWaitKey(0);

    int iterCount = gcapp.getIterCount();
    cout << "<" << iterCount << "... ";
    int newIterCount = gcapp.nextIter();
    if( newIterCount > iterCount )
    {
        gcapp.showImage();
        cout << iterCount << ">" << endl;
    }

    char ymlfilename[200];

    sprintf(ymlfilename, "%s.mask.yml", argv[1]); //TODO: dafuer iostreams benutzen?

    FileStorage fs(ymlfilename, FileStorage::WRITE);
    fs << "mask" << gcapp.mask;


    sprintf(ymlfilename, "%s.gmm.yml", argv[1]); //TODO: dafuer iostreams benutzen?

    FileStorage fs2(ymlfilename, FileStorage::WRITE);
    fs2 << "test" << 2;
    fs2 << "fgdModel" << gcapp.fgdModel;
    fs2 << "bgdModel" << gcapp.bgdModel;

    cvWaitKey(0);


exit_main:
    cvDestroyWindow( winName.c_str() );
    return 0;
}
