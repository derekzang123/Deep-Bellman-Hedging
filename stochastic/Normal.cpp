#include "Normal.h"
#include <cmath>

namespace Normal { 
    const double PI = 3.14159265358979323846;
    const double SQRT_TWO_PI = std::sqrt(2.0 * PI);

    // Normal PDF: f(x) = (1/(σ√(2π))) * exp(-((x-μ)^2)/(2σ^2))
    double PDF(double x, double mean, double stddev) {
        double z = (x - mean) / stddev;
        return (1.0 / (stddev * SQRT_TWO_PI)) * std::exp(-0.5 * z * z);
    }

    // Normal CDF: Φ(x) = (1/2) * [1 + erf((x-μ)/(σ√2))]
    double CDF(double x, double mean, double stddev) {
        double z = (x - mean) / (stddev * std::sqrt(2.0));
        return 0.5 * (1.0 + std::erf(z));
    }
}