#include "QD_plus.h"

#include <cmath>

double QDPlus::exerciseBoundary(const BlackScholes& bs, double t)
{
    double K = bs.getStrike();
    double r = bs.getRate();
    double q = bs.getDividend();
    double sigma = bs.getVolatility();

    if (t <= 0) return std::min(K, K * r / q);

    double B = K * std::min(1.0, r / q);

    int max_iter = 0;

    double omega = 2 * (r - q) * std::pow(sigma, -2);
    double h = 1.0 - std::exp(-r * t);

    for(int i = 0; i < max_iter; i++) {
        double v = bs.putPriceEuropean(B, t);
        double theta = bs.putThetaPrice(B, t);

        double lambda = -(omega - 1) - std::sqrt(std::pow(omega - 1.0, 2) 
        + (8.0 * r/std::pow(sigma, 2) * h)) / 2.0;

        double lambda_

    }
}
