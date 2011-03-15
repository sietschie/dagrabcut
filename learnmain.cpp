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
         "./grabcut <input_image_name1> <input_image_name2>... <class_number> <model_name>\n"
         "reads input_images and input_image_names.yml, generates\n"
         "GMM of the combined images for class_number\n"
         << endl;
}

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

void learnGMMfromSamples(vector<Vec3f> samples, Mat& model)
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;
    const int componentsCount = 5;

    Mat labels;
    CV_Assert( !samples.empty() );
    Mat _samples( (int)samples.size(), 3, CV_32FC1, &samples[0][0] );

    kmeans( _samples, componentsCount, labels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType, 0 );

    cout << "start learning GMM..." << endl;

    GMM gmm(model);

    gmm.initLearning();
    for( int i = 0; i < (int)samples.size(); i++ )
        gmm.addSample( labels.at<int>(i,0), samples[i] );
    gmm.endLearning();
}

int main( int argc, char** argv )
{
    if( argc < 3 )
    {
        help();
        return 1;
    }

    vector<Vec3f> bgdSamples, fgdSamples;

    int class_number = atoi(argv[argc-2]);

    for(int i=1; i<argc-2; i++)
    {
        string filename = argv[i];

        cout << "Reading " << filename << "..." << endl;

        Mat image, mask;
        readImageAndMask(filename, image, mask);

        Point p;
        for( p.y = 0; p.y < image.rows; p.y++ )
        {
            for( p.x = 0; p.x < image.cols; p.x++ )
            {
                if( mask.at<uchar>(p) != class_number)
                    bgdSamples.push_back( (Vec3f)image.at<Vec3b>(p) );
                else // GC_FGD | GC_PR_FGD
                    fgdSamples.push_back( (Vec3f)image.at<Vec3b>(p) );
            }
        }

    }

    Mat bgdModel, fgdModel;
    learnGMMfromSamples(bgdSamples, bgdModel);
    learnGMMfromSamples(fgdSamples, fgdModel);

    help();

    FileStorage fs2(argv[argc-1], FileStorage::WRITE);
    fs2 << "fgdModel" << fgdModel;
    fs2 << "bgdModel" << bgdModel;

    return 0;
}
