#include "shared.hpp"

#include <iostream>
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

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

