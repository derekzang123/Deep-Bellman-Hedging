#include "QD_plus.h"
#include "../Utils.h"

#include <vector>
#include <cmath>
#include <stdexcept>

QDPlus::QDParams QDPlus::getQDParams(const BlackScholes& bs, double t) { 
    const double r = bs.getRate();
    const double q = bs.getDividend();
    const double s = bs.getVolatility();

    const double h = 1.0 - std::exp(-r * t);
    const double omega = 2 * (r - q) * std::pow(s, -2);
    const double lambda = (1.0 - omega - std::sqrt(std::pow(omega - 1.0, 2) 
    + (8.0 * r/std::pow(s, 2) * h))) / 2.0;

    const double dlambda = (2.0 * r / std::pow(s, 2)) / 
        std::pow(s, 2) * std::pow(h, 2) * std::sqrt(std::pow(omega - 1.0, 2) 
        + (8.0 * r/std::pow(s, 2) * h)) / 2.0;
    
    return QDParams(lambda, dlambda, omega, h);
} 

std::vector<double> QDPlus::exerciseBoundary(
    const BlackScholes& bs, 
    std::vector<double>& tVec,
    int maxIter,
    double ts, 
    double tf
) 
{
    std::vector<double> BvsI(tVec.size());

    const double K = bs.getStrike();      
    const double r = bs.getRate();        
    const double q = bs.getDividend();    
    const double s = bs.getVolatility();  

    for (size_t i = 0; i < tVec.size(); ++i) {
        const double t = tVec[i];

        if (t <= 0) {
            BvsI[i] = std::min(1,0);  // Placeholder for exact values
        }
        const QDParams qd = getQDParams(bs, t);
        
        double B = K * std::min(1.0, r / q);

        int iter = 0;
        do {
            const double dP = bs.getDplus(t, B / K);
            const double dM = bs.getDminus(t, B / K);

            const double ncdfNegDp = Utils::NCDF(-dP);
            const double ncdfNegDm = Utils::NCDF(-dM);
            const double ncdfDp = Utils::NCDF(dP);

            const dual l = qd.lambda;
            const dual v = bs.putPriceEuropean(B, t);
            const dual th = bs.putThetaPrice(t, B);
            const double D = 2.0 * qd.lambda + qd.omega - 1.0;
            const dual c0 = -((1.0 - qd.h) * s * s) / (r * D)
                            * (1.0 / qd.h - std::exp(r * t) * th / (r * (K - B - v))
                            - qd.dlambda / D);

            auto f = [&](dual q, dual t, dual B, dual K, dual l, dual c0, dual v) -> dual 
            {
                return (-std::exp(-q * t) * Utils::NCDF(-bs.getDplus(t, B / K)) 
                        + (l + c0) * (K - B - v) / B + 1.0);
            };

            dual dfdb = derivative(f, q, t, B, K, qd.lambda, c0, v);
            double dfdB = derivative(f, q, t, B, K, qd.lambda, c0, v);
            double dfdB2 = derivative(dfdb, wrt(B), at(q, t, B, K, qd.lambda, c0, v));

            double sH = (2.0 * f * dfdB) / (2.0 * std::pow(dfdB, 2) - dfdB2 * f);
            B -= sH;

            double f_ = (-std::exp(-q * t) * Utils::NCDF(-bs.getDplus(t, B / K)) 
                        + (qd.lambda + c0) * (K - B - v) / B + 1.0);

            if (std::abs(f_) < tf && std::abs(sH) < ts) {
                break;
            }
            if (iter == maxIter) {
                throw std::runtime_error("QD+ failed to converge within maxIter iterations");
            }
        } while (++iter < maxIter);
        BvsI[i] = B;
    }
    return BvsI;
}