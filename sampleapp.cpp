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
    		"./grabcut <input_image_name1> <input_image_name2>... <image_name>\n"
			"reads input_images and input_image_names.mask.yml, generates\n"
			"GMM of the combined images, uses this to initialize grabcut on image_name\n"
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
    int max_iterations = 1;
    Rect rect;

    cg_grabCut( *image, mask, rect, bgdModel, fgdModel, max_iterations );

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

    vector<Vec3f> bgdSamples, fgdSamples;

    for(int i=1; i<argc-1; i++)
    {
        string filename = argv[i];
        
        cout << "Reading " << filename << "..." << endl;

        if( filename.empty() )
        {
        	cout << "\nDurn, couldn't read in " << argv[i] << endl;
            return 1;
        }
        Mat image = imread( filename, 1 );
        if( image.empty() )
        {
            cout << "\n Durn, couldn't read image filename " << filename << endl;
        	return 1;
        }
        
        string mask_filename = filename;
        mask_filename.append(".mask.yml");

        FileStorage fs(mask_filename, FileStorage::READ);
        Mat mask; fs["mask"] >> mask;

        Point p;
        for( p.y = 0; p.y < image.rows; p.y++ )
        {
            for( p.x = 0; p.x < image.cols; p.x++ )
            {
                if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                    bgdSamples.push_back( (Vec3f)image.at<Vec3b>(p) );
                else // GC_FGD | GC_PR_FGD
                    fgdSamples.push_back( (Vec3f)image.at<Vec3b>(p) );
            }
        }

    }

    cout << "starting k-Means..." << endl;

    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
    Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
    kmeans( _bgdSamples, GMM::componentsCount, bgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType, 0 );
    Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, GMM::componentsCount, fgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType, 0 );

    cout << "start learning GMM..." << endl;
    
    Mat bgdModel, fgdModel;
    GMM bgdGMM(bgdModel), fgdGMM(fgdModel);

    bgdGMM.initLearning();
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();

    cout << "finished learning GMM..." << endl;

    help();

    string filename = argv[argc-1];
    
    cout << "Reading " << filename << "..." << endl;

    if( filename.empty() )
    {
    	cout << "\nDurn, couldn't read in " << argv[argc-1] << endl;
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

    char ymlfilename[200];

    sprintf(ymlfilename, "%s.naive-mask.yml", argv[1]); //TODO: dafuer iostreams benutzen?

    FileStorage fs(ymlfilename, FileStorage::WRITE);
    fs << "mask" << gcapp.mask;

    cvWaitKey(0);

exit_main:
    cvDestroyWindow( winName.c_str() );
    return 0;
}
