/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

//#include "precomp.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "maxflow/graph.h"
#include <limits>

#include <iostream>

using namespace cv;

/*
This is implementation of image segmentation algorithm GrabCut described in
"GrabCut — Interactive Foreground Extraction using Iterated Graph Cuts".
Carsten Rother, Vladimir Kolmogorov, Andrew Blake.
 */

#include "st_gaussian.hpp"
#include "msst_grabcut.hpp"
#include "structuretensor.hpp"

/*
  Calculate beta - parameter of GrabCut algorithm.
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
double calcBeta( const MSStructureTensorImage& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            vector<StructureTensor> color = img.getTensor(y,x);
            if( x>0 ) // left
            {
                double dist = MS_distance2(color, img.getTensor(y,x-1));
                beta += dist * dist;
            }
            if( y>0 && x>0 ) // upleft
            {
                double dist = MS_distance2(color, img.getTensor(y-1,x-1));
                beta += dist * dist;
            }
            if( y>0 ) // up
            {
                double dist = MS_distance2(color, img.getTensor(y-1,x));
                beta += dist * dist;
            }
            if( y>0 && x<img.cols-1) // upright
            {
                double dist = MS_distance2(color, img.getTensor(y-1,x+1));
                beta += dist * dist;
            }
        }
    }
    beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );
    std::cout << "beta = " << beta << std::endl;
    return beta;
}

/*
  Calculate weights of noterminal vertices of graph.
  beta and gamma - parameters of GrabCut algorithm.
 */
void calcNWeights( const MSStructureTensorImage& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma)
{
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            vector<StructureTensor> color = img.getTensor(y,x);
            if( x-1>=0 ) // left
            {
                double dist = MS_distance2(color,img.getTensor(y,x-1));
                leftW.at<double>(y,x) = gamma * exp(-beta*dist*dist);
            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                double dist = MS_distance2(color,img.getTensor(y-1,x-1));
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*dist*dist);
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                double dist = MS_distance2(color,img.getTensor(y-1,x));
                upW.at<double>(y,x) = gamma * exp(-beta*dist*dist);
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols-1 && y-1>=0 ) // upright
            {
                double dist = MS_distance2(color,img.getTensor(y-1,x+1));
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*dist*dist);
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
}

/*
  Check size, type and element values of mask matrix.
 */
void checkMask( const Mat& img, const Mat& mask )
{
    if( mask.empty() )
        CV_Error( CV_StsBadArg, "mask is empty" );
    if( mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "mask must have CV_8UC1 type" );
    if( mask.cols != img.cols || mask.rows != img.rows )
        CV_Error( CV_StsBadArg, "mask must have as many rows and cols as img" );
    for( int y = 0; y < mask.rows; y++ )
    {
        for( int x = 0; x < mask.cols; x++ )
        {
            uchar val = mask.at<uchar>(y,x);
            if( val!=GC_BGD && val!=GC_FGD && val!=GC_PR_BGD && val!=GC_PR_FGD )
                CV_Error( CV_StsBadArg, "mask element value must be equel"
                          "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD" );
        }
    }
}

/*
  Initialize mask using rectangular.
*/
void initMaskWithRect( Mat& mask, int rows, int cols, Rect rect )
{
    mask.create( rows, cols, CV_8UC1 );
    mask.setTo( GC_BGD );

    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, rows);
    rect.height = min(rect.height, cols);

    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

/*
  Initialize GMM background and foreground models using kmeans algorithm.
*/
void initGMMs( const MSStructureTensorImage& img, const Mat& mask, MSST_GMM& bgdGMM, MSST_GMM& fgdGMM )
{
    const int componentsCount = 5;
    const int kMeansItCount = 20;
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    vector<vector<StructureTensor> > bgdSamples, fgdSamples;
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            vector<StructureTensor> st;
            st = img.getTensor(p.x, p.y);
            if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                bgdSamples.push_back( st );
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back( st );
        }
    }
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
    vector<vector<StructureTensor> > bgd_centers, fgd_centers;
    MSST_kmeans( bgdSamples, componentsCount, 
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 1, bgdLabels, bgd_centers );
    MSST_kmeans( fgdSamples, componentsCount, 
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 1, fgdLabels, fgd_centers );

    bgdGMM.setComponentsCount(componentsCount);
    bgdGMM.initLearning();
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();

    fgdGMM.setComponentsCount(componentsCount);
    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();

/*    std::cout << "kmeans center: " << std::endl;
    for(int i=0;i<5;i++)
    {
        Mat center = fgd_centers[i].getMatrix();
        std::cout << center.at<double>(0,0) << "  " << center.at<double>(1,1) << "  " << center.at<double>(0,1) << std::endl;
    }

    std::cout << "gmm center: " << std::endl;
    for(int i=0;i<5;i++)
    {
        Mat center = fgdGMM.components[i].gauss.mean;
        std::cout << center.at<double>(0,0) << "  " << center.at<double>(1,1) << "  " << center.at<double>(0,1) << std::endl;
    }*/

}

/*
  Assign GMMs components for each pixel.
*/
void assignGMMsComponents( const MSStructureTensorImage& img, const Mat& mask, const MSST_GMM& bgdGMM, const MSST_GMM& fgdGMM, Mat& compIdxs )
{
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            vector<StructureTensor> color = img.getTensor(p.x, p.y);
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                                  bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);

            //std::cout << " compIdxs.at<int>(p) = " << compIdxs.at<int>(p) << std::endl;
        }
    }
}

/*
  Learn GMMs parameters.
*/
void learnGMMs( const MSStructureTensorImage& img, const Mat& mask, const Mat& compIdxs, MSST_GMM& bgdGMM, MSST_GMM& fgdGMM )
{
    int componentsCount = 5;
    bgdGMM.setComponentsCount(5);
    bgdGMM.initLearning();
    fgdGMM.setComponentsCount(5);
    fgdGMM.initLearning();
    Point p;
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        for( p.y = 0; p.y < mask.rows; p.y++ )
        {
            for( p.x = 0; p.x < mask.cols; p.x++ )
            {
                if( compIdxs.at<int>(p) == ci )
                {
                    if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                        bgdGMM.addSample( ci, img.getTensor(p.x,p.y) );
                    else
                        fgdGMM.addSample( ci, img.getTensor(p.x,p.y) );
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

/*
  Construct GCGraph
*/
void constructGCGraph( const MSStructureTensorImage& img, const Mat& mask, const MSST_GMM& bgdGMM, const MSST_GMM& fgdGMM, double lambda,
                       const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                       Graph<double, double, double>& graph )
{
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.add_node();
            vector<StructureTensor> color = img.getTensor(p.x,p.y);

            // set t-weights
            double fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                fromSource = -log( bgdGMM(color) );
                toSink = -log( fgdGMM(color) );
//                fromSource = -log( 100 );
//                toSink = -log( 1000000000 );
//                std::cout << "bgd = " << bgdGMM(color) << "  fgd = " << fgdGMM(color) << std::endl;
            }
            else if( mask.at<uchar>(p) == GC_BGD )
            {
                fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            graph.add_tweights( vtxIdx, fromSource, toSink );

            // set n-weights
            if( p.x>0 )
            {
                double w = leftW.at<double>(p);
                graph.add_edge( vtxIdx, vtxIdx-1, w, w );
            }
            if( p.x>0 && p.y>0 )
            {
                double w = upleftW.at<double>(p);
                graph.add_edge( vtxIdx, vtxIdx-mask.cols-1, w, w );
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p);
                graph.add_edge( vtxIdx, vtxIdx-mask.cols, w, w );
            }
            if( p.x<mask.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p);
                graph.add_edge( vtxIdx, vtxIdx-mask.cols+1, w, w );
            }
        }
    }
}

/*
  Estimate segmentation using MaxFlow algorithm
*/
double estimateSegmentation( Graph<double, double, double>& graph, Mat& mask )
{
    double flow = graph.maxflow();

    int prfgd_c = 0;

    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                if( graph.what_segment( p.y*mask.cols+p.x /*vertex index*/ ) == Graph<double, double, double>::SOURCE )
                {
                    mask.at<uchar>(p) = GC_PR_FGD;
                    prfgd_c++;
                }
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
    std::cout << "prfgd_c = " << prfgd_c << std::endl;

    return flow;
}

void cg_msst_grabCut( const MSStructureTensorImage& img, Mat& mask, Rect rect,
                 Mat& bgdModel, Mat& fgdModel,
                 int iterCount, int mode )
{
//    if( img.empty() )
//        CV_Error( CV_StsBadArg, "image is empty" );
//    if( img.type() != CV_8UC3 )
//        CV_Error( CV_StsBadArg, "image mush have CV_8UC3 type" );

    MSST_GMM bgdGMM, fgdGMM;
    bgdGMM.setModel(bgdModel);
    fgdGMM.setModel(fgdModel);

    Mat compIdxs( mask.size(), CV_32SC1 );

    if( mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK )
    {
        if( mode == GC_INIT_WITH_RECT )
            initMaskWithRect( mask, img.rows, img.cols, rect );
//        else // flag == GC_INIT_WITH_MASK
//            checkMask( img, mask );
        initGMMs( img, mask, bgdGMM, fgdGMM );
    }

    mask.create( img.cols, img.rows, CV_8UC1 );
    mask.setTo( GC_PR_FGD );

    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                double bgd = bgdGMM(img.getTensor(p.x,p.y));
                double fgd = fgdGMM(img.getTensor(p.x,p.y));
                if( bgd > fgd)
                    mask.at<uchar>(p) = GC_PR_BGD;
                else
                    mask.at<uchar>(p) = GC_PR_FGD;
            }
        }
    }


    if( iterCount <= 0)
        return;

//    if( mode == GC_EVAL )
//        checkMask( img, mask );

    const double gamma = 50;
    const double lambda = 9*gamma;
    const double beta = calcBeta( img );

    Mat leftW, upleftW, upW, uprightW;
    calcNWeights( img, leftW, upleftW, upW, uprightW, beta, gamma );

    //TODO: gibts fuer epsilon nich was passendes in opencv?
    double last_flow = 0.0;
    double current_flow = 100.0;
    double eps = 0.001;

    int vtxCount = mask.cols*mask.rows,
        edgeCount = 2*(4*mask.cols*mask.rows - 3*(mask.cols + mask.rows) + 2);

    for( int i = 0; i < iterCount; i++ )
    {
        Graph<double, double, double> graph(vtxCount,edgeCount);
        assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
        learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph );
        current_flow = estimateSegmentation( graph, mask );
        std::cout << " diff flow:" << abs(last_flow - current_flow) << " " << std::flush;

        if( abs(last_flow - current_flow) < eps )
            break;

        last_flow = current_flow;
    }
    Mat new_bgdModel = bgdGMM.getModel();
    Mat new_fgdModel = fgdGMM.getModel();

    new_bgdModel.copyTo(bgdModel);
    new_fgdModel.copyTo(fgdModel);
}


void cg_msst_interactive_grabCut( const MSStructureTensorImage& img, Mat& mask, Rect rect,
             Mat& bgdModel, Mat& fgdModel,
             int iterCount, int mode )
{
//    if( img.empty() )
//        CV_Error( CV_StsBadArg, "image is empty" );
//    if( img.type() != CV_8UC3 )
//        CV_Error( CV_StsBadArg, "image mush have CV_8UC3 type" );

    MSST_GMM bgdGMM, fgdGMM;
    if(!bgdModel.empty())
        bgdGMM.setModel(bgdModel);
    if(!fgdModel.empty())
        fgdGMM.setModel(fgdModel);

    Mat compIdxs( img.rows, img.cols, CV_32SC1 );

    if( mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK )
    {
        if( mode == GC_INIT_WITH_RECT )
            initMaskWithRect( mask, img.rows, img.cols, rect );
//        else // flag == GC_INIT_WITH_MASK
//            checkMask( img, mask );
        initGMMs( img, mask, bgdGMM, fgdGMM );
    }

    if( iterCount <= 0)
        return;

//    if( mode == GC_EVAL )
//        checkMask( img, mask );

    const double gamma = 5;
    const double lambda = 9*gamma;
    const double beta = calcBeta( img );

    Mat leftW, upleftW, upW, uprightW;
    calcNWeights( img, leftW, upleftW, upW, uprightW, beta, gamma );

    //TODO: gibts fuer epsilon nich was passendes in opencv?
    double last_flow = 0.0;
    double current_flow = 100.0;
    double eps = 0.001;

    int vtxCount = img.cols*img.rows,
        edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);

    for( int i = 0; i < iterCount; i++ )
    {
        std::cout << "start iteration" << std::endl;
        Graph<double, double, double> graph(vtxCount,edgeCount);
        assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
        learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph );
        current_flow = estimateSegmentation( graph, mask );
        std::cout << " diff flow:" << abs(last_flow - current_flow) << " " << std::flush;

        if( abs(last_flow - current_flow) < eps )
            break;

        last_flow = current_flow;
    }
}
