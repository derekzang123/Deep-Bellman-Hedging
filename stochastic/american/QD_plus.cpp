#include "../Utils.h"
#include "QD_plus.h"
#include "../BlackScholes.h"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

double p(double S, double t, double K, const BlackScholes &bs) {
    return std::exp(-bs.getRate() * t) * K * Utils::NCDF(-d2(S, t, K, bs)) -
    S * std::exp(-bs.getDividend() * t) * Utils::NCDF(-d1(S, t, K, bs));
}

double d1(double S, double t, double K,  const BlackScholes &bs) {
  return (std::log(S * exp(t * bs.getRate() - bs.getDividend() / K))) / (bs.getVolatility() * std::sqrt(t)) +
         0.5 * bs.getVolatility() * std::sqrt(t);
}

double d2(double S, double t, double K, const BlackScholes &bs) {
  return d1(S, t, K, bs) - bs.getVolatility() * std::sqrt(t);
}

std::vector<double> QD_(const BlackScholes &bs, std::vector<double> &tVec,
                        int maxIter, double ts, double tf) 
{
    std::vector<double> BvsI(tVec.size());
    const double K = bs.getStrike();
    const double r = bs.getRate();
    const double d = bs.getDividend();
    const double s = bs.getVolatility();
    const double N = 2 * (r - d) / (s * s);
    const double M = 2 * r / (s * s);
    
    for (size_t i = 0; i < tVec.size(); ++i) {
        const double t = tVec[i];

        if (t == 0) {
            BvsI[i] = K * std::min(1.0, r / d);
        }

        const double h = 1 - std::exp(-r * t);
        const double qd = (1 - N - std::sqrt(std::pow(N - 1, 2) + 4 * M / h)) / 2;
        double B = K * std::min(1.0, r / d);
        int iter = 0;

        do {
            double F = 1 - std::exp(-d * t) * Utils::NCDF(-d1(B, t, K, bs)) - B  + qd * (K - B * t - p(B, t, K, bs));
            double dFdB = (1 - qd) * (1 - std::exp(-d * t) * Utils::NPDF(d1(B, t, K, bs)) / (B * s * std::sqrt(t)) + 1);

            double sN = F / dFdB;
            B -= sN;
            if (std::abs(F) < tf && std::abs(sN) < ts) {
                break;
            }
            if (iter == maxIter) {
                throw std::runtime_error("QD+ failed to converge within 'maxIter' iterations");
            }
        } while (++iter < maxIter);
        BvsI[i] = B;
    }
    return BvsI;
}