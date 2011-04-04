#ifndef _MSST_MM_H_
#define _MSST_MM_H_

#include <opencv2/core/core.hpp>
#include "st_gaussian.hpp"
#include "structuretensor.hpp"

template <class Component>
class MSST_MM {
public:
    static const int dim = 3;

    MSST_MM();
    double operator()( const std::vector<StructureTensor> color ) const;
    double operator()( int ci, const std::vector<StructureTensor> color ) const;
    int whichComponent( const std::vector<StructureTensor> color ) const;

    void initLearning();
    void addSample( int ci, const std::vector<StructureTensor> );
    void endLearning();
    cv::Mat getModel();
    void setModel(const cv::Mat &model);
    void setComponentsCount(int size);
    int getComponentsCount();

    double KLdiv(const MSST_MM& rhs);
    double KLsym(MSST_MM& rhs);

    std::vector<Component> components;
protected:
    std::vector<std::vector<std::vector<StructureTensor> > > samples;

};

#include "msst_mm.cpp"

#endif /* _MSST_MM_H_ */
