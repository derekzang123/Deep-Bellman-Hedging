#include "Utils.h"
#include <cmath>


double Utils::NPDF(double x) {
    const double PI = atan(1) * 4;
    return (1.0 / sqrt(2.0 * PI)) * exp(-0.5 * x * x);
}

double Utils::NCDF(double x) {
    // Approximation of the cumulative distribution function for standard normal distribution
    const double b1 =  0.319381530;
    const double b2 = -0.356563782;
    const double b3 =  1.781477937;
    const double b4 = -1.821255978;
    const double b5 =  1.330274429;
    const double p  =  0.2316419;
    const double c  =  0.39894228;

    if (x >= 0.0) {
        double t = 1.0 / (1.0 + p * x);
        return (1.0 - c * exp(-x * x / 2.0) * t * 
                (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
    } else {
        double t = 1.0 / (1.0 - p * x);
        return (c * exp(-x * x / 2.0) * t * 
                (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
    }
}