#ifndef QDPLUS_H
#define QDPLUS_H

#include <vector> 
#include "../BlackScholes.h"
#include "FastAmericanOptionPricing.h"

class QDPlus {
public:
    std::vector<double> QDPlus::exerciseBoundary(
        const BlackScholes& bs, 
        std::vector<double>& tVec,
        int maxIter,
        double ts, 
        double tf
    );

private:    
    struct QDParams {
        double lambda;
        double dlambda;
        double omega;
        double h; 

        QDParams(double lambda_, double dlambda_, double omega_, double h_)
            : lambda(lambda_), dlambda(dlambda_), omega(omega_), h(h_) {}
    };
    static QDParams getQDParams(const BlackScholes& bs, double T);
};

#endif