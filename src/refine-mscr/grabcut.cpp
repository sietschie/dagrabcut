#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

void help()
{
    cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
    		"and then grabcut will attempt to segment it out.\n"
    		"Call:\n"
    		"./grabcut <image_name> <class number>\n"
    	"\nSelect a rectangular area around the object you want to segment\n" <<
        "\nHot keys: \n"
        "\tESC - quit the program\n"
        "\tr - restore the original image\n"
        "\tn - next iteration\n"
        "\ts - save result to yml-file\n"
        "\n"
        "\tleft mouse button - set rectangle\n"
        "\n"
        "\tCTRL+left mouse button - set GC_BGD pixels\n"
        "\tSHIFT+left mouse button - set CG_FGD pixels\n"
        "\n"
        "\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
        "\tSHIFT+right mouse button - set CG_PR_FGD pixels\n" << endl;
}

const Scalar RED = Scalar(0,0,255);
const Scalar PINK = Scalar(230,130,255);
const Scalar BLUE = Scalar(255,0,0);
const Scalar LIGHTBLUE = Scalar(255,255,160);
const Scalar GREEN = Scalar(0,255,0);

const int BGD_KEY = CV_EVENT_FLAG_CTRLKEY;
const int FGD_KEY = CV_EVENT_FLAG_SHIFTKEY;

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
    enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
    static const int radius = 2;
    static const int thickness = -1;

    void reset();
    void setImageAndWinName( const Mat& _image, const Mat& _mask, const string& _winName );
    void showImage() const;
    void mouseClick( int event, int x, int y, int flags, void* param );
    int nextIter();
    int getIterCount() const { return iterCount; }
    int getMasks(Mat& _mask); 
private:
    void setLblsInMask( int flags, Point p, bool isPr );

    const string* winName;
    const Mat *image;
    Mat mask, gt_mask;
    Mat bgdModel, fgdModel;

    uchar lblsState, prLblsState;
    bool isInitialized;

    vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
    int iterCount;
};

int GCApplication::getMasks(Mat& _mask)
{
    _mask = mask;
}

void GCApplication::reset()
{
    bgdPxls.clear(); fgdPxls.clear();
    prBgdPxls.clear();  prFgdPxls.clear();

    isInitialized = false;
    lblsState = NOT_SET;
    prLblsState = NOT_SET;
    iterCount = 0;
}

void GCApplication::setImageAndWinName( const Mat& _image, const Mat& _gt_mask, const string& _winName  )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    winName = &_winName;

    mask = _gt_mask.clone();
    reset();
}

void GCApplication::showImage() const
{
    if( image->empty() || winName->empty() )
        return;

    Mat res;
    Mat binMask;
    /*if( !isInitialized )
        image->copyTo( res );
    else
    {*/
        getBinMask( mask, binMask );
        image->copyTo( res, binMask );
    //}

    vector<Point>::const_iterator it;
    for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )
        circle( res, *it, radius, BLUE, thickness );
    for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )
        circle( res, *it, radius, RED, thickness );
    for( it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it )
        circle( res, *it, radius, LIGHTBLUE, thickness );
    for( it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it )
        circle( res, *it, radius, PINK, thickness );

    //if( rectState == IN_PROCESS || rectState == SET )
    //    rectangle( res, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), GREEN, 2);

    imshow( "aaaa", *image );
    imshow( *winName, res );

    Mat res2;
    image->copyTo( res2, 1 - binMask );
    imshow( "aaaa2", res2 );

}

void GCApplication::setLblsInMask( int flags, Point p, bool isPr )
{
    vector<Point> *bpxls, *fpxls;
    uchar bvalue, fvalue;
    if( !isPr )
    {
        bpxls = &bgdPxls;
        fpxls = &fgdPxls;
        bvalue = GC_BGD;
        fvalue = GC_FGD;
    }
    else
    {
        bpxls = &prBgdPxls;
        fpxls = &prFgdPxls;
        bvalue = GC_PR_BGD;
        fvalue = GC_PR_FGD;
    }
    if( flags & BGD_KEY )
    {
        bpxls->push_back(p);
        circle( mask, p, radius, bvalue, thickness );
    }
    if( flags & FGD_KEY )
    {
        fpxls->push_back(p);
        circle( mask, p, radius, fvalue, thickness );
    }
}

void GCApplication::mouseClick( int event, int x, int y, int flags, void* )
{
    // TODO add bad args check
    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
        {
            bool isb = (flags & BGD_KEY) != 0,
                 isf = (flags & FGD_KEY) != 0;
            if ( (isb || isf) )
                lblsState = IN_PROCESS;
        }
        break;
    case CV_EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels
        {
            bool isb = (flags & BGD_KEY) != 0,
                 isf = (flags & FGD_KEY) != 0;
            if ( (isb || isf) )
                prLblsState = IN_PROCESS;
        }
        break;
    case CV_EVENT_LBUTTONUP:
        if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), false);
            lblsState = SET;
            showImage();
        }
        break;
    case CV_EVENT_RBUTTONUP:
        if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), true);
            prLblsState = SET;
            showImage();
        }
        break;
    case CV_EVENT_MOUSEMOVE:
        if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), false);
            showImage();
        }
        else if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), true);
            showImage();
        }
        break;
    }
}

int GCApplication::nextIter()
{
    Rect rect;
    if( isInitialized )
        grabCut( *image, mask, rect, bgdModel, fgdModel, 1 );
    else
    {
        //if( lblsState == SET || prLblsState == SET )
        grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK );
        //else
            //grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT );

        isInitialized = true;
    }
    iterCount++;

    bgdPxls.clear(); fgdPxls.clear();
    prBgdPxls.clear(); prFgdPxls.clear();

    return iterCount;
}

GCApplication gcapp;

void on_mouse( int event, int x, int y, int flags, void* param )
{
    gcapp.mouseClick( event, x, y, flags, param );
}

int main( int argc, char** argv )
{
    if( argc!=3 && argc !=4 )
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

    int class_number = atoi(argv[2]);

    string mask_filename;
    if(argc == 3) { 
        mask_filename = filename;
        mask_filename.append(".yml");
    } else {
        mask_filename = argv[3];
    }

    Mat gt_mask;
    FileStorage fs(mask_filename, FileStorage::READ);
    fs["mask"] >> gt_mask;

    Mat bin_gt_mask; 
    bin_gt_mask.create( image.size(), CV_8UC1);

    Point p;
    for( p.y = 0; p.y < gt_mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < gt_mask.cols; p.x++ )
        {
            if( gt_mask.at<uchar>(p) != class_number)
                bin_gt_mask.at<uchar>(p) = GC_BGD;
            else // GC_FGD | GC_PR_FGD
                bin_gt_mask.at<uchar>(p) = GC_PR_FGD;
        }
    }

    help();

    const string winName = "image";
    cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );
    cvSetMouseCallback( winName.c_str(), on_mouse, 0 );

    gcapp.setImageAndWinName( image, bin_gt_mask, winName );
    gcapp.showImage();

    for(;;)
    {
        int iterCount, newIterCount;
        int c = cvWaitKey(0);
        switch( (char) c )
        {
        case '\x1b':
            cout << "Exiting ..." << endl;
            goto exit_main;
        case 'r':
            cout << endl;
            gcapp.reset();
            gcapp.showImage();
            break;
        case 'n':
            iterCount = gcapp.getIterCount();
            cout << "<" << iterCount << "... ";
            newIterCount = gcapp.nextIter();
            if( newIterCount > iterCount )
            {
                gcapp.showImage();
                cout << iterCount << ">" << endl;
            }
            else
                cout << "rect must be determined>" << endl;
            break;
        case 's':
            string ymlfilename = filename + ".refined."+ argv[2] +".yml";

            Mat bin_mask;
            gcapp.getMasks(bin_mask);


            
            Mat res_mask = gt_mask.clone(); 

            Point p;
            for( p.y = 0; p.y < gt_mask.rows; p.y++ )
            {
                for( p.x = 0; p.x < gt_mask.cols; p.x++ )
                {
                    // Vordergrund in der Eingabemaske, aber Hintergrund im Grabcutergebnis
                    if( gt_mask.at<uchar>(p) == class_number && ( bin_mask.at<uchar>(p) == GC_BGD || bin_mask.at<uchar>(p) == GC_PR_BGD ))
                        res_mask.at<uchar>(p) = 0;
                }
            }

            cout << "save mask to " << ymlfilename << endl;

            FileStorage fs(ymlfilename, FileStorage::WRITE);
            fs << "mask" << res_mask;
            fs << "initial_mask" << gt_mask;
            break;
        }
    }

exit_main:
    cvDestroyWindow( winName.c_str() );
    return 0;
}
