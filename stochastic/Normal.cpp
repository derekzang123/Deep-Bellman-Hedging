#include "Normal.h"
#include <cmath>

namespace Normal { 
    const double PI = atan(1) * 4;
    const double SQRT_TWO_PI = std::sqrt(2.0 * PI);

    double PDF(double x, double mean, double stddev) {
        double z = (x - mean) / stddev;
        return (1.0 / (stddev * SQRT_TWO_PI)) * std::exp(-0.5 * z * z);
    }

    double CDF(double x, double mean, double stddev) {
        double z = (x - mean) / (stddev * std::sqrt(2.0));
        return 0.5 * (1.0 + std::erf(z));
    }
}