//#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "hmm.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    HMM fgdHmm, bgdHmm;
    for(int i=1; i<argc-1; i++)
    {
        string imagename = argv[i];
        
        std::cout << "Reading " << imagename << "..." << std::endl;

        if( imagename.empty() )
        {
        	cout << "\nDurn, couldn't read in " << argv[i] << endl;
            return 1;
        }
        Mat image = imread( imagename, 1 );
        if( image.empty() )
        {
            cout << "\n Durn, couldn't read image filename " << imagename << endl;
        	return 1;
        }

        string go_filename = imagename;
        go_filename.append(".grabcut-output.yml");

        FileStorage fs(go_filename, FileStorage::READ);
        Mat mask; fs["mask"] >> mask;
        Mat compIdxs; fs["componentIndexes"] >> compIdxs;
        Mat bgdModel; fs["bgdModel"] >> bgdModel;
        Mat fgdModel; fs["fgdModel"] >> fgdModel;

        fgdHmm.add_model(fgdModel, compIdxs, mask & 1, image);



    }
    fgdHmm.cluster_once();
    fgdHmm.cluster_once();
    fgdHmm.cluster_once();
    fgdHmm.cluster_once();

    FileStorage fs2("test.yml", FileStorage::WRITE);
    fs2 << "hmm" << fgdHmm;
    fs2.release();

    HMM testHmm;
    FileStorage fs3("test.yml", FileStorage::READ);
    readHMM(fs3["hmm"], testHmm);
    fs3.release();

    FileStorage fs4("test2.yml", FileStorage::WRITE);
    fs4 << "hmm" << testHmm;
    fs4.release();
}
