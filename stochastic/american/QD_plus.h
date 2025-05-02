#ifndef QDPLUS_H
#define QDPLUS_H

#include <vector> 
#include "../BlackScholes.h"
#include "FastAmericanOptionPricing.h"

class QDPlus {
public:
    static double exerciseBoundary(const BlackScholes& bs, double T);

private:    
};

#endif