#ifndef QDPLUS_H
#define QDPLUS_H

#include <vector> 
#include "../BlackScholes.h"
#include "FastAmericanOptionPricing.h"

class QDPlus {
private:
    BlackScholes bs;
    std::vector<double> boundary;
    std::vector<double> tauNodes;
    double X;
    double tauMax;
    int n, m, l;

public: 
    QDPlus(const BlackScholes& bs_, int n_, int m_, int l_, double tauMax_);

    void initBoundary();
    double qdPlusApprox(double tau);

};


#endif