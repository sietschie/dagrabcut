#include <iostream>
#include "st_mm.hpp"

using namespace cv;

template <class Component>
ST_MM<Component>::ST_MM()
{
}

template <class Component>
void ST_MM<Component>::setModel(const Mat& model) 
{
    while(components.size() != 0)
    {
        components.pop_back();
    }


    for(int i=0; i<model.cols; i++)
    {
        ST_Gaussian gauss;
        assert(model.rows == 5);

        gauss.mean.template at<double>(0,0) = model.at<double>(0,i);
        gauss.mean.template at<double>(1,1) = model.at<double>(1,i);
        gauss.mean.template at<double>(1,0) = model.at<double>(2,i);
        gauss.mean.template at<double>(0,1) = model.at<double>(2,i);
        gauss.cov.template at<double>(0,0) = model.at<double>(3,i);

        Component MMc;
        MMc.gauss = gauss;
        MMc.weight = model.at<double>(4,i);
        components.push_back(MMc);
    }
}

template <class Component>
void ST_MM<Component>::setComponentsCount(int size)
{
    if( components.size() == size ) 
        return;

    while( components.size() < size ) 
    {
        Component MMc;
        MMc.gauss.mean = Mat(2,2, CV_64FC1);
        MMc.gauss.cov = Mat(1,1, CV_64FC1);
        MMc.weight = 0.0;
        components.push_back(MMc);
    }

    while( components.size() > size )
    {
        components.pop_back();
    }
}

template <class Component>
int ST_MM<Component>::getComponentsCount()
{
    return components.size();
}

template <class Component>
Mat ST_MM<Component>::getModel()
{
//    const int modelSize = dim/*mean*/ + dim*dim/*covariance*/ + 1/*component weight*/;
    const int modelSize = 5;
    Mat model;
    model.create( modelSize, components.size(), CV_64FC1 );
    model.setTo(Scalar(0));

    for(int i=0; i<components.size(); i++) //TODO: iterator
    {
        model.at<double>(0,i) = components[i].gauss.mean.template at<double>(0,0);
        model.at<double>(1,i) = components[i].gauss.mean.template at<double>(1,1);
        model.at<double>(2,i) = components[i].gauss.mean.template at<double>(1,0);
        model.at<double>(3,i) = components[i].gauss.cov.template at<double>(0,0);
        model.at<double>(4,i) = components[i].weight;
    }
    return model;
}

template <class Component>
double ST_MM<Component>::operator()( const StructureTensor color ) const
{
    double res = 0;
    for( int ci = 0; ci < components.size(); ci++ )
        res += components[ci].weight * (*this)(ci, color );
    return res;
}

template <class Component>
double ST_MM<Component>::operator()( int ci, const StructureTensor color ) const
{
    double res = 0;
    const Component &MMc = components[ci];
    if( MMc.weight > 0 )
    {
        double dist = distance2(color, MMc.gauss.mean);
//        std::cout << "dist = " << dist << std::endl;
        double sqr_dist = dist * dist;

        double mul = sqr_dist / (2 * MMc.gauss.cov.template at<double>(0,0));
        res = exp(-mul) / (2 * M_PI * MMc.gauss.cov.template at<double>(0,0));
//        double mul = sqr_dist / (2);
//        res = exp(-mul) / (2 * M_PI);
//        return exp(-dist*dist);        
    }

        // squared distance between ST_i and Input_ST
        // divide by 2* variance 
        // exponentiate
        // multiply by 1/sqrt 2 pi variance


//FIXME: fuer ST anpassen
/*        double covDeterms = determinant(MMc.gauss.cov);
        //cout << "ci: " << ci << "  covDeterms: " << covDeterms << endl;
        //cout << MMc.weight << endl;
        //cout << MMc.gauss.mean << endl;
        //cout << MMc.gauss.cov << endl;
        if(covDeterms != covDeterms)
        {
            std::cout << "covDeterms: " << covDeterms << std::endl;
            std::cout << MMc.gauss.cov << std::endl;
        }
        CV_Assert( covDeterms > std::numeric_limits<double>::epsilon() );
        Vec3d diff = color;
        diff[0] -= MMc.gauss.mean.template at<double>(0,0);
        diff[1] -= MMc.gauss.mean.template at<double>(1,0);
        diff[2] -= MMc.gauss.mean.template at<double>(2,0);

        Mat inverseCovs = MMc.gauss.cov.inv();

        double mult = 0.0;
        for(int i=0; i<3; i++)
        {
            double row = 0.0;
            for(int j=0; j<3; j++)
            {
                row += diff[j] * inverseCovs.at<double>(j,i);
            }
            mult += diff[i] * row;
        }

        //double test = diff * (inverseCovs * diff);

        //double mult = diff[0]*(diff[0]*inverseCovs[0][0] + diff[1]*inverseCovs[1][0] + diff[2]*inverseCovs[2][0])
        //           + diff[1]*(diff[0]*inverseCovs[0][1] + diff[1]*inverseCovs[1][1] + diff[2]*inverseCovs[2][1])
        //           + diff[2]*(diff[0]*inverseCovs[0][2] + diff[1]*inverseCovs[1][2] + diff[2]*inverseCovs[2][2]);
        res = 1.0f/sqrt(covDeterms) * exp(-0.5f*mult);*/
    return res;
}

template <class Component>
int ST_MM<Component>::whichComponent( const StructureTensor color ) const
{
    int k = 0;
    double max = -100000000.0;

    for( int ci = 0; ci < components.size(); ci++ ) //TODO: iterator
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

template <class Component>
void ST_MM<Component>::initLearning()
{
    samples.resize(components.size()); //TODO: ist das noetig?
    for(int i=0; i< samples.size(); i++)
        samples[i].resize(0);
}

template <class Component>
void ST_MM<Component>::addSample( int ci, const StructureTensor color )
{
    assert(ci < samples.size());

    samples[ci].push_back(color);
}

template <class Component>
void ST_MM<Component>::endLearning()
{
    int numSamples = 0;
    for(int i=0; i<samples.size(); i++)
    {
        numSamples += samples[i].size();
    }

    for(int i=0; i<samples.size(); i++)
    {
        components[i].gauss.compute_from_samples(samples[i]);
        components[i].weight = samples[i].size() / (double) numSamples;

    }
}

template <class Component>
double ST_MM<Component>::KLsym(ST_MM& rhs)
{
    return KLdiv(rhs) + rhs.KLdiv(*this);
}

template <class Component>
double ST_MM<Component>::KLdiv(const ST_MM& rhs)
{
    vector<int> mapping(components.size());
    for(int i=0;i<components.size();i++)
    {
        int min_component = 0;
        double min = 10e+100; //TODO: richtig grossen wert aus limits nehmen
        for(int j=0;j<rhs.components.size();j++)
        {
            double div = components[i].gauss.KLdiv(rhs.components[j].gauss) - log(rhs.components[j].weight);
            if( div < min )
            {
                min = div;
                min_component = j;
            }
            mapping[i] = min_component;
        }
    }
    
    double div = 0.0;
    for(int i=0;i<components.size();i++)
    {
        if( components[i].weight != 0 )
        {
            double kl = components[i].gauss.KLdiv(rhs.components[mapping[i]].gauss);
            double summand =  log( components[i].weight / rhs.components[mapping[i]].weight);
            div += components[i].weight * (kl + summand);
        }
    }
    return div;
}
