#include <iostream>
#include "mm.hpp"

using namespace cv;

template <class Component>
MM<Component>::MM()
{
}

template <class Component>
void MM<Component>::setModel(const Mat& model) 
{
    while(components.size() != 0)
    {
        components.pop_back();
    }


    for(int i=0; i<model.cols; i++)
    {
        Gaussian gauss;
        int c=0;
        gauss.mean = Mat(dim,1, CV_64FC1);
        for(int j=0; j < dim; j++)
        {
            double value = model.at<double>(c,i);
            gauss.mean.at<double>(j,0) = value;
            c++;
        }

        gauss.cov = Mat(dim,dim, CV_64FC1);
        for(int j=0; j<dim; j++)
            for(int k=0; k<dim; k++)
            {
                gauss.cov.at<double>(j,k) = model.at<double>(c,i);
                c++;
            }
        Component MMc;
        MMc.gauss = gauss;
        MMc.weight = model.at<double>(c,i);
        components.push_back(MMc);
    }
}

template <class Component>
void MM<Component>::setComponentsCount(int size)
{
    if( components.size() == size ) 
        return;

    while( components.size() < size ) 
    {
        Component MMc;
        MMc.gauss.mean = Mat(dim,1, CV_64FC1);
        MMc.gauss.cov = Mat(dim,dim, CV_64FC1);
        MMc.weight = 0.0;
        components.push_back(MMc);
    }

    while( components.size() > size )
    {
        components.pop_back();
    }
}

template <class Component>
int MM<Component>::getComponentsCount()
{
    return components.size();
}

template <class Component>
Mat MM<Component>::getModel()
{
    const int modelSize = dim/*mean*/ + dim*dim/*covariance*/ + 1/*component weight*/;
    Mat model;
    model.create( modelSize, components.size(), CV_64FC1 );
    model.setTo(Scalar(0));

    for(int i=0; i<components.size(); i++) //TODO: iterator
    {
        int c=0;
        for(int j=0; j < 3; j++)
        {
            double value = components[i].gauss.mean.template at<double>(j,0);
            model.at<double>(c,i) = value;
            c++;
        }

        for(int j=0; j<3; j++)
            for(int k=0; k<3; k++)
            {
                model.at<double>(c,i) = components[i].gauss.cov.template at<double>(j,k);
                c++;
            }
        model.at<double>(c,i) = components[i].weight;
    }
    return model;
}

template <class Component>
double MM<Component>::operator()( const Vec3d color ) const
{
    double res = 0;
    for( int ci = 0; ci < components.size(); ci++ )
        res += components[ci].weight * (*this)(ci, color );
    return res;
}

template <class Component>
double MM<Component>::operator()( int ci, const Vec3d color ) const
{
    double res = 0;
    const Component &MMc = components[ci];
    if( MMc.weight > 0 )
    {
        double covDeterms = determinant(MMc.gauss.cov);
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
        res = 1.0f/sqrt(covDeterms) * exp(-0.5f*mult);
    }
    return res;
}

template <class Component>
int MM<Component>::whichComponent( const Vec3d color ) const
{
    int k = 0;
    double max = 0;

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
void MM<Component>::initLearning()
{
    samples.resize(components.size()); //TODO: ist das noetig?
}

template <class Component>
void MM<Component>::addSample( int ci, const Vec3d color )
{
    assert(ci < samples.size());

    samples[ci].push_back(color);
}

template <class Component>
void MM<Component>::endLearning()
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
double MM<Component>::KLsym(MM& rhs)
{
    return KLdiv(rhs) + rhs.KLdiv(*this);
}

template <class Component>
double MM<Component>::KLdiv(const MM& rhs)
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
