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
    		"./outlines2mask <image_name> class_number\n" << endl;
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

    int class_number = atoi(argv[2]);

    string contour_filename;
    contour_filename = filename;
    contour_filename.append(".contour.yml");

//    Mat gt_mask;
    FileStorage fs(contour_filename, FileStorage::READ);
    FileNodeIterator contours = fs["outlines"].begin();

    vector<Mat> list_of_contours;

    cout << "start reading yml file..." << endl;
    for(;contours != fs["outlines"].end();contours++)
    {
	Mat contour;
	*contours >> contour;
	list_of_contours.push_back(contour);
	cout << "cols = " << contour.cols << "  rows = " << contour.rows << endl;
    }

    string class_name;
    fs["class_name"] >> class_name;
    cout << "class_name = " << class_name << endl;

    string image_name;
    fs["image_name"] >> image_name;
    cout << "image_name = " << image_name << endl;

    Point** pts = new Point*[ list_of_contours.size() ];
    int* npts = new int[ list_of_contours.size() ];

    for(vector<Mat>::iterator itr = list_of_contours.begin(); itr != list_of_contours.end(); itr++)
    for(int i=0; i< list_of_contours.size(); i++)
    {
	pts[i] = new Point[list_of_contours[i].rows];
	for(int j=0; j<list_of_contours[i].rows; j++)
	{
	    pts[i][j].x = list_of_contours[i].at<int>(j,0);
	    pts[i][j].y = list_of_contours[i].at<int>(j,1);
	    //cout << "pts.x = " << pts[i][j].x << "  pts.y = " << pts[i][j].y << endl;

	}
	npts[i] = list_of_contours[i].rows;
    }

    Mat outline(image.size(), CV_8UC1, Scalar(0,0,0));
    fillPoly(outline, (const Point**) pts, npts, list_of_contours.size(), Scalar(255,0,0));

    string mask_filename = filename + ".gt.bmp";
    imwrite( mask_filename, outline );


    fillPoly(outline, (const Point**) pts, npts, list_of_contours.size(), Scalar(class_number,0,0));

    double sum=0;
    MatConstIterator_<uchar> it = outline.begin<uchar>(), it_end = outline.end<uchar>();
    for(; it != it_end; ++it)
	if( *it != 0 )
	    sum += 1.0;

    double average = sum / (outline.cols * outline.rows);

    
    FileStorage fs2(filename + ".yml", FileStorage::WRITE);
    fs2 << "image_name" << filename;
    fs2 << "num_contours" << (int) list_of_contours.size();
    fs2 << "img_cols" << image.cols;
    fs2 << "img_rows" << image.rows;
    fs2 << "contour_area" << average;
    fs2 << "mask" << outline;


    bool move = false;
    if( image.cols < 200 || image.rows < 200 )
    {
	cout << "file to small..." << endl;
	move = true;
    }

    if( image.cols * image.rows > 1000000 )
    {
	cout << "file to big..." << endl;
	move = true;
    }

    if( list_of_contours.size() == 0 )
    {
	cout << "no outline..." << endl;
	move = true;
    }

    if( average < 0.05 )
    {
	cout << "region to small... " << average << endl;
	move = true;
    }

    if( move && system(NULL))
    {	
	cout << "move this file... " << endl;
	system("mkdir aussortiert");
	string mvcommand = "mv "+filename+"*  aussortiert/";
	system( mvcommand.c_str() );
    }

/*    Mat bin_gt_mask; 
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

    help();*/

    return 0;
}
