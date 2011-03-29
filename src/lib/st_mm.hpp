#ifndef _ST_MM_H_
#define _ST_MM_H_

#include <opencv2/core/core.hpp>
#include "st_gaussian.hpp"
#include "structuretensor.hpp"

template <class Component>
class ST_MM {
public:
    static const int dim = 3;

    ST_MM();
    double operator()( const StructureTensor color ) const;
    double operator()( int ci, const StructureTensor color ) const;
    int whichComponent( const StructureTensor color ) const;

    void initLearning();
    void addSample( int ci, const StructureTensor );
    void endLearning();
    cv::Mat getModel();
    void setModel(const cv::Mat &model);
    void setComponentsCount(int size);
    int getComponentsCount();

    double KLdiv(const ST_MM& rhs);
    double KLsym(ST_MM& rhs);

    std::vector<Component> components;
protected:
    std::vector<std::vector<StructureTensor> > samples;

};

#include "st_mm.cpp"

#endif /* _ST_MM_H_ */
