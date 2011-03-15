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
         "./grabcut <input_model> <input_image> <class_number> <output_name>\n"
         "reads input_images and input_image_names.yml, generates\n"
         "GMM of the combined images for class_number\n"
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
    int nextIter(int max_iterations);
    int getIterCount() const {
        return iterCount;
    }
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

int GCApplication::nextIter(int max_iterations = 2)
{
    isInitialized = true;
    Rect rect;

    cg_grabCut( *image, mask, rect, bgdModel, fgdModel, max_iterations );

    iterCount += max_iterations;

    return iterCount;
}

GCApplication gcapp;

void readImageAndMask(string filename, Mat& image, Mat& mask)
{
    cout << "Reading " << filename << "..." << endl;

    if( filename.empty() )
    {
        cout << "\nDurn, couldn't read in " << filename << endl;
        return;
    }
    image = imread( filename, 1 );
    if( image.empty() )
    {
        cout << "\n Durn, couldn't read image filename " << filename << endl;
        return;
    }

    string mask_filename = filename;
    mask_filename.append(".yml");

    FileStorage fs(mask_filename, FileStorage::READ);
    fs["mask"] >> mask;
}

void compareMasks(const Mat &gt,const Mat& segm, int class_number, int& true_positive, int& true_negative, int& false_positive, int& false_negative, int& unknown)
{
    true_positive = 0;
    true_negative = 0;
    false_positive = 0;
    false_negative = 0;
    unknown = 0;

    assert(gt.rows == segm.rows && gt.cols == segm.cols);

    Point p;
    for( p.y = 0; p.y < gt.rows; p.y++ )
    {
        for( p.x = 0; p.x < gt.cols; p.x++ )
        {
            if(gt.at<uchar>(p) == class_number)
            {
                if(segm.at<uchar>(p) == GC_FGD | segm.at<uchar>(p) == GC_PR_FGD)
                {
                    true_positive++;
                } else {
                    false_negative++;
                }
            } else if (gt.at<uchar>(p) == 255) {
                unknown++;
            } else {
                if(segm.at<uchar>(p) == GC_FGD | segm.at<uchar>(p) == GC_PR_FGD)
                {
                    false_positive++;
                } else {
                    true_negative++;
                }
            }
        }
    }
    assert(gt.rows * gt.cols == true_positive + true_negative + false_positive + false_negative + unknown);

}


int main( int argc, char** argv )
{
    if( argc < 3 )
    {
        help();
        return 1;
    }

    vector<Vec3f> bgdSamples, fgdSamples;

    string fn_model = argv[1];
    string fn_image = argv[2];
    int class_number = atoi(argv[3]);
    string fn_output = argv[4];

    Mat fgdModel, bgdModel;
    FileStorage fs(fn_model, FileStorage::READ);
    fs["fgdModel"] >> fgdModel;
    fs["bgdModel"] >> bgdModel;
    fs.release();

    Mat image, mask;
    readImageAndMask(fn_image, image, mask);

    help();

    const string winName = "image";
    cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );

    gcapp.setImageAndWinName( image, winName, bgdModel, fgdModel );
    gcapp.showImage();

    cvWaitKey(1000);

    int iterCount = gcapp.getIterCount();
    cout << "<" << iterCount << "... ";
    int newIterCount = gcapp.nextIter();
    if( newIterCount > iterCount )
    {
        gcapp.showImage();
        cout << newIterCount << ">" << endl;
    }

    FileStorage fs2(fn_output, FileStorage::WRITE);
    fs2 << "mask" << gcapp.mask;
    fs2 << "fgdModel" << fgdModel;
    fs2 << "bgdModel" << bgdModel;

    int tp, tn, fp, fn, unknown;
    compareMasks(mask, gcapp.mask, class_number, tp, tn, fp, fn, unknown);
    cout << "true positive: " << tp << ", true negative: " << tn << ", false positive: " << fp << ", false negative: " << fn << ", unknown: " << unknown << endl;

    double fgd_rate = tp / (double) (tp + fn);
    double bgd_rate = tn / (double) (tn + fp);
    double joint_rate = (fgd_rate + bgd_rate) / 2;

    cout << "fgd: " << fgd_rate << ", bgd: " << bgd_rate << ", joint: " << joint_rate << endl;


    cvWaitKey(0);

exit_main:
    cvDestroyWindow( winName.c_str() );
    return 0;
}
