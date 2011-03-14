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


#include "gmm.hpp"
#include <iostream>

using namespace cv;

/*
 GMM - Gaussian Mixture Model
*/


GMM::GMM( Mat& _model, int _componentsCount )
{
    componentsCount = _componentsCount;
    const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
    if( _model.empty() )
    {
        _model.create( modelSize, componentsCount, CV_64FC1 );
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_64FC1) || (_model.rows != modelSize) || (_model.cols != componentsCount) )
        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == modelSize and cols == componentsCount" );

    model = _model;

    for(int i=0;i<componentsCount;i++)
    {
        Gaussian gauss;
        int c=0;
        gauss.mean = Mat(3,1, CV_64FC1);
        for(int j=0; j < 3; j++)
        {
            double value = model.at<double>(c,i);
            gauss.mean.at<double>(j,0) = value;
            c++;
        }

        gauss.cov = Mat(3,3, CV_64FC1);
        for(int j=0;j<3;j++)
        for(int k=0;k<3;k++)
        {
            gauss.cov.at<double>(j,k) = model.at<double>(c,i);
            c++;
        }       
        GMM_Component *gmmc = new GMM_Component;
        gmmc->gauss = gauss;
        gmmc->weight = model.at<double>(c,i);
        components.push_back(gmmc); 
    }
}

int GMM::getComponentsCount()
{
    return components.size();
}

Mat GMM::getModel()
{
    return model;
}

Mat GMM::updateModel()
{
    for(int i=0;i<componentsCount;i++)
    {
        int c=0;
        for(int j=0; j < 3; j++)
        {
            double value = components[i]->gauss.mean.at<double>(j,0);
            model.at<double>(c,i) = value;
            c++;
        }

        for(int j=0;j<3;j++)
        for(int k=0;k<3;k++)
        {
            model.at<double>(c,i) = components[i]->gauss.cov.at<double>(j,k);
            c++;
        }       
        model.at<double>(c,i) = components[i]->weight;
    }
    return model;
}

double GMM::operator()( const Vec3d color ) const
{
    double res = 0;
    for( int ci = 0; ci < components.size(); ci++ )
        res += components[ci]->weight * (*this)(ci, color );
    return res;
}

double GMM::operator()( int ci, const Vec3d color ) const
{
    double res = 0;
    GMM_Component* gmmc = components[ci];
    if( gmmc->weight > 0 )
    {
        double covDeterms = determinant(gmmc->gauss.cov);
        //cout << "ci: " << ci << "  covDeterms: " << covDeterms << endl;
        //cout << gmmc->weight << endl;
        //cout << gmmc->gauss.mean << endl;
        //cout << gmmc->gauss.cov << endl;
        if(covDeterms != covDeterms)
        {
            std::cout << "covDeterms: " << covDeterms << std::endl;
            std::cout << gmmc->gauss.cov << std::endl;
        }
		CV_Assert( covDeterms > std::numeric_limits<double>::epsilon() );
        Vec3d diff = color;
        diff[0] -= gmmc->gauss.mean.at<double>(0,0);
        diff[1] -= gmmc->gauss.mean.at<double>(1,0);
        diff[2] -= gmmc->gauss.mean.at<double>(2,0);

        Mat inverseCovs = gmmc->gauss.cov.inv();

        double mult = 0.0;
        for(int i=0;i<3;i++)
        {
            double row = 0.0;
            for(int j=0;j<3;j++)
            {
                row += diff[j] * inverseCovs.at<double>(j,i);
            }
            mult += diff[i] * row;
        }

        //double test = diff * (inverseCovs * diff);

        //double mult = diff[0]*(diff[0]*inverseCovs[0][0] + diff[1]*inverseCovs[1][0] + diff[2]*inverseCovs[2][0])
        //           + diff[1]*(diff[0]*inverseCovs[0][1] + diff[1]*inverseCovs[1][1] + diff[2]*inverseCovs[2][1])
        //           + diff[2]*(diff[0]*inverseCovs[0][2] + diff[1]*inverseCovs[1][2] + diff[2]*inverseCovs[2][2]);
        res = 1.0f/sqrt(covDeterms) * exp(-0.5f*mult);
    }
    return res;
}

int GMM::whichComponent( const Vec3d color ) const
{
    int k = 0;
    double max = 0;

    for( int ci = 0; ci < componentsCount; ci++ )
    {
		double p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}

void GMM::initLearning()
{
    samples.resize(componentsCount);
}

void GMM::addSample( int ci, const Vec3d color )
{
    samples[ci].push_back(color);
}

void GMM::endLearning()
{
    int numSamples = 0;
    for(int i=0;i<samples.size();i++)
    {
        numSamples += samples[i].size();
    }

    for(int i=0;i<samples.size();i++)
    {
        components[i]->gauss.compute_from_samples(samples[i]);
        components[i]->weight = samples[i].size() / (double) numSamples;

    }
    updateModel();
}
