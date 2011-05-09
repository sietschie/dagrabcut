#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace cv;

void help()
{
    cout << "\n"
    		"Call:\n"
    		"./outlines2mask <image_name> factor\n" << endl;
}


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

    int factor = atoi(argv[2]);

    Mat image_small;
    resize(image, image_small, Size(), 1.0 / factor, 1.0 / factor);

    FileStorage fs2(filename + ".yml", FileStorage::READ);

    Mat mask;
    fs2["mask"] >> mask;

    Mat mask_small;
    resize(mask, mask_small, Size(), 1.0 / factor, 1.0 / factor, INTER_NEAREST);

    imwrite(filename + ".klein.jpg", image_small);

    FileStorage fs3(filename + ".klein.jpg.yml", FileStorage::WRITE);

    fs3 << "mask" << mask_small;
    fs3 << "image_name" << filename + ".klein.jpg";

    imwrite("tmp.mask.bmp", mask);
    imwrite("tmp.mask-small.bmp", mask_small);


    /*FileStorage fs2(filename + ".yml", FileStorage::WRITE);
    fs2 << "image_name" << filename;
    fs2 << "num_contours" << (int) list_of_contours.size();
    fs2 << "img_cols" << image.cols;
    fs2 << "img_rows" << image.rows;
    fs2 << "contour_area" << average;
    fs2 << "mask" << outline;*/
    return 0;
}
