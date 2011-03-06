//#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "hmm.hpp"

using namespace cv;
using namespace std;

void print_mean_variance(vector<double> list)
{
	double sum = 0;
	vector<double>::iterator itr;
	for(itr = list.begin(); itr != list.end(); itr++)
	{
		sum += *itr;
	}
	double mean = sum / list.size();

	double sum_squaredmeandiff = 0;

	for(itr = list.begin(); itr != list.end(); itr++)
	{
		double meandiff = *itr - mean;
		sum_squaredmeandiff += meandiff * meandiff;
	}
	double variance = sum_squaredmeandiff / list.size();

	cout << "Mean: " << mean << "   Variance: " << variance << endl;

}

int main( int argc, char** argv )
{


    HMM fgdHmm, bgdHmm;
    vector<HMM> fgdHmms, bgdHmms;


    for(int i=1; i<argc-1; i++)
    {
        HMM cur_fgdHmm, cur_bgdHmm;
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

        cur_fgdHmm.add_model(fgdModel, compIdxs, mask & 1, image);
        cur_bgdHmm.add_model(bgdModel, compIdxs, 1 - (mask & 1), image);

        fgdHmms.push_back(cur_fgdHmm);
        bgdHmms.push_back(cur_bgdHmm);
    }
    
    vector<double> fgdDivs;
    for(int i=0;i<fgdHmms.size();i++)
    for(int j=i+1;i<fgdHmms.size();i++)
    {
        fgdDivs.push_back(fgdHmms[i].KLsym(fgdHmms[j]));
    }
	cout << "fgdDivs:  ";
	print_mean_variance(fgdDivs);

    vector<double> bgdDivs;
    for(int i=0;i<bgdHmms.size();i++)
    for(int j=i+1;i<bgdHmms.size();i++)
    {
        bgdDivs.push_back(bgdHmms[i].KLsym(bgdHmms[j]));
    }
	cout << "bgdDivs:  ";
	print_mean_variance(bgdDivs);

    vector<double> betweenDivs;
    for(int i=0;i<bgdHmms.size();i++)
    for(int j=0;i<fgdHmms.size();i++)
    {
        betweenDivs.push_back(bgdHmms[i].KLsym(bgdHmms[j]));
    }
	cout << "betweenDivs:  ";
	print_mean_variance(betweenDivs);


    for(int i=0;i<fgdHmms.size();i++)
        fgdHmm.add_model(fgdHmms[i]);
    for(int i=0;i<bgdHmms.size();i++)
        bgdHmm.add_model(bgdHmms[i]);

    cout << "KLdiv: " << fgdHmm.KLdiv(bgdHmm) << endl;
    cout << "KLdiv: " << bgdHmm.KLdiv(fgdHmm) << endl;

    fgdHmm.normalize_weights();
    bgdHmm.normalize_weights();

    while( fgdHmm.components.size() > 5)
    {
        fgdHmm.cluster_once();
        bgdHmm.cluster_once();
    }

    FileStorage fs2(argv[argc-1], FileStorage::WRITE);
    fs2 << "first" << 2;
    fs2 << "files" << "[";
    for(int i=1; i<argc-1;i++)
    {
        fs2 << argv[i];
    }
    fs2 << "]";
    fs2 << "fgdHmm" << fgdHmm;
    fs2 << "bgdHmm" << bgdHmm;
    fs2.release();

    fgdHmm.free_components();
    bgdHmm.free_components();

/*    HMM testHmm;
    FileStorage fs3("test.yml", FileStorage::READ);
    readHMM(fs3["fgdHmm"], testHmm);
    fs3.release();

    FileStorage fs4("test2.yml", FileStorage::WRITE);
    fs4 << "fgdHmm" << testHmm;
    fs4.release();*/
}