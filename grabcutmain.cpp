#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "grabcut.hpp"
#include "hmm.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void help()
{
    cout << "Call:\n"
    		"./grabcut <image_name> <generate_hmm-outputfile>\n"
			"runs grabcut on <image_name> using the supplied hmm as initialisation"
         << endl;
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
    void setImageAndWinName( const Mat& _image, const string& _winName, Mat& _bgdModel, Mat& _fgdModel );
    void showImage() const;
    int nextIter();
    int getIterCount() const { return iterCount; }
    Mat mask;
private:
    const string* winName;
    const Mat* image;
    Mat input_mask;
    Mat bgdModel, fgdModel;
    Mat input_bgdModel, input_fgdModel;

    bool isInitialized;

    int iterCount;
};

void GCApplication::setImageAndWinName( const Mat& _image, const string& _winName, Mat& _bgdModel, Mat& _fgdModel )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    winName = &_winName;

    input_bgdModel = _bgdModel;
    input_fgdModel = _fgdModel;

    bgdModel = input_bgdModel.clone();
    fgdModel = input_fgdModel.clone();
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
        FileStorage fs("test2.yml", FileStorage::WRITE);
        fs << "mask" << binMask;

        image->copyTo( res, binMask );
    }

    imshow( *winName, res );
}

int GCApplication::nextIter()
{
    isInitialized = true;
    int max_iterations = 2;
    Rect rect;

    cout << "begin grabcut" << endl;
    cg_grabCut( *image, mask, rect, bgdModel, fgdModel, max_iterations );
    cout << "end grabcut" << endl;

    iterCount += max_iterations;

    return iterCount;
}

GCApplication gcapp;

int main( int argc, char** argv )
{
    if( argc < 3 )
    {
    	help();
        return 1;
    }

    string hmmfilename = argv[1];
    cout << "Reading " << hmmfilename << "..." << endl;
    if( hmmfilename.empty() )
    {
    	cout << "\nDurn, couldn't read in " << argv[1] << endl;
        return 1;
    }

    HMM fgdHmm, bgdHmm;
    FileStorage fs(hmmfilename, FileStorage::READ);
    readHMM(fs["fgdHmm"], fgdHmm);
    readHMM(fs["bgdHmm"], bgdHmm);
    fs.release();


    string filename = argv[2];
    
    cout << "Reading " << filename << "..." << endl;

    if( filename.empty() )
    {
    	cout << "\nDurn, couldn't read in " << argv[2] << endl;
        return 1;
    }
    Mat image = imread( filename, 1 );
    if( image.empty() )
    {
        cout << "\n Durn, couldn't read image filename " << filename << endl;
    	return 1;
    }
    
    cout << "finished reading HMM from file..." << endl;

    help();

    const string winName = "image";
    cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );

    Mat bgdModel = bgdHmm.get_model();
    Mat fgdModel = fgdHmm.get_model();

    cout << "fgdModel: " << fgdModel << endl;

    gcapp.setImageAndWinName( image, winName, bgdModel, fgdModel );
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

//    char ymlfilename[200];

//    sprintf(ymlfilename, "%s.mask.yml", argv[1]); //TODO: dafuer iostreams benutzen?

//    FileStorage fs(ymlfilename, FileStorage::WRITE);
//    fs << "mask" << gcapp.mask;

    cvWaitKey(0);

exit_main:
    cvDestroyWindow( winName.c_str() );
    return 0;
}
