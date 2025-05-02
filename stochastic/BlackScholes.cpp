#include "BlackScholes.h"
#include "american/FastAmericanOptionPricing.h"
#include "american/QD_plus.h"
#include "./Utils.h"


BlackScholes::BlackScholes(double S_, double K_, double r_, double q_, double T_, double vol_, 
    int N_, int steps_, OptionType type_, ExerciseStyle style_)
    : S(S_), K(K_), r(r_), q(q_), T(T_), vol(vol_), N(N_), steps(steps_), type(type_), style(style_),
    rng(std::random_device{}()), dist(0.0, 1.0) { }

double BlackScholes::getDplus(double tau, double z) const { 
    double num = (std::log(z) + (r - q) * tau + 0.5 * vol * vol * tau);
    double den = (vol * std::sqrt(tau));
    return (double)(num / den);
}

double BlackScholes::getDminus(double tau, double z) const { 
    double num = (std::log(z) + (r - q) * tau - 0.5 * vol * vol * tau);
    double den = (vol * std::sqrt(tau));
    return (double)(num / den);
}

double BlackScholes::price() const {
    if (style == ExerciseStyle::American) {
        int n = 1, m = 1, l = 1;
        double tauMax = 1.0; // placeholders
        return priceAmerican(n, m, l, tauMax); 
    }
    return priceEuropean();
}

double BlackScholes::putPriceEuropean(double B, double t) const {
    double t1 = std::exp(-r * t) * K *
        Utils::NCDF(-getDminus(t, B / K));
    double t2 = -B * std::exp(-q * t) *
        Utils::NCDF(-getDplus(t, B / K));
    return t1 - t2;
}

double BlackScholes::putThetaPrice(double t, double Bt) const {
    if (t <= 0) return 0.0;

    double t1 = r * K * std::exp(-r * t) * 
        Utils::NCDF(-getDminus(t, Bt / K));
    double t2 = -q * Bt * std::exp(-q * t) * 
        Utils::NCDF(-getDplus(t, Bt / K));
    double t3 = -vol * Bt / 2 * std::sqrt(t) *
        std::exp(-q * t) * Utils::NPDF(getDplus(t, Bt / K));
    
    return t1 - t2 - t3;
}





double BlackScholes::priceAmerican(int n, int m, int l, double tauMax) const {
    auto [xVec, zVec] = computeNodes(n + 1, tauMax);
    auto tauNodes = computeCollocation(xVec);
    auto [quadNodes, quadWeights] = quadrature(l, 0.5);
    QDPlus qd(*this, n, m, l, tauMax); 
    qd.initBoundary();
    auto B = qd.getBoundary();
    double X = (r >= q) ? K : K * (r/q);
    std::vector<double> H;

    for(auto& B_i : B) {
        H.push_back(std::pow(std::log(B_i / X), 2));
    }

    auto coeffs = cWeights(H);
    
    for(int i = 1; i < m; i++) {
        /// 6: ...pushback(qC(t_i - t_i * (1 + y_k)^2 / 4))
    }

    /*
    N = getN(tauNodes,B);
    D = getD(tauNodes,B);
    */

    
    
    // Remainder of Fast American logic
    return 0.0;
}
