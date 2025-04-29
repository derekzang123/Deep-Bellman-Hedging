#include "FastAmericanOptionPricing.h"
#include "../BlackScholes.h"
#include "../Utils.h"
#include <vector> 
#include <cmath> 

/* STEP 1: Compute Chebyshev Nodes */
std::pair<std::vector<double>, std::vector<double>> computeNodes(int n, int tMax) { 
    std::vector<double> zVec, xVec;
    for (int i = 0; i < n; i++) {
        double z = -1 * std::cos((i*PI)/n);
        double x = std::sqrt(tMax) / 2;
        x *= 1 + z;
        xVec.push_back(x);
        zVec.push_back(z);
    }
    return {xVec, zVec};
}

std::vector<double> computeCollocation(std::vector<double> xVec) {
    std::vector<double> tau;
    for (int i = 0; i <= xVec.size() - 1; i++) {
        double t = std::pow(xVec[i], 2);
        tau.push_back(t);
    }
    return tau;
}


/* STEP 2: Tanh-Sinh Quadrature*/
std::pair<std::vector<double>, std::vector<double>> quadrature(int l, double h) {
    std::vector<double> nodes;
    std::vector<double> weights;
    int half_l = (l + 1) / 2;
    for (int k = 0; k < half_l; k++) {
        double n = std::tanh(0.5 * PI * std::sinh(k*h));
        double w = (0.5 * h * std::cosh(k * h)) / std::pow(std::cosh(0.5 * PI * std::sinh(k * h)), 2);
        nodes.push_back(n);
        weights.push_back(w);
        if (k > 0 || l % 2 == 0) {
            nodes.push_back(-n);
            weights.push_back(w);
        }
    }
    return {nodes, weights};
}

std::vector<double> cWeights(std::vector<double> H) { 
    int N = H.size() - 1;
    std::vector<double> a (N + 1);
    for(int k = 0; k <= N; k++) { 
        a[k] = (H[0] + (k % 2 == 0 ? 1.0 : -1.0) * H[N]) / (2.0 * N);
        for (int i = 1; i < N; i++) {
            a[k] += 2.0 / N * H[i] * std::cos(PI * i * k / N);
        }
    }
    return a;
}

std::vector<double> cBasis(double z, int N, bool F) {
    std::vector<double> TVec (N + 1);
    if (N >= 1) {
        TVec[0] = 1.0;
    }
    TVec[1] = F && N >= 2 ? z : 2 * z;
    for (size_t n = 2; n <= N; n ++) {
        TVec[n] = 2.0 * TVec[1] * TVec[n-1] - TVec[n-2];
    }
    return TVec;
}

double qC(double z, const std::vector<double>& a) {
    const int N = a.size() - 1; 
    if (N < 0) return 0.0; 

    double b_k1 = a[N];
    double b_k2 = 0.0; 
    
    for (int k = N - 1; k >= 1; --k) {
        double b_k = a[k] + 2.0 * z * b_k1 - b_k2;
        b_k2 = b_k1;
        b_k1 = b_k;
    }
    return a[0] + z * b_k1 - b_k2;
}

double quadSum (std::function<double(double)>& f, 
                std::vector<double> nodes, 
                std::vector<double> weights) 
{
    return std::transform_reduce(
        nodes.begin(), nodes.end(),
        weights.begin(), 
        0.0, 
        std::plus<>(),
        [&f](double x, double w) {
            return f(x) * w;
        }
    );
}

std::function<double(double)> K1I(
    double t, 
    double q, 
    double s,
    const std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
    return [t, q, s, &B, &bs](double y) {
        double u = t * std::pow(y + 1.0, 2) / 4.0;
        return std::exp(q * u) * Utils::NCDF(bs.getDplus(t - u, B(t) / B(u)));
    };
}

std::function<double(double)> K2I(
    double t, 
    double q, 
    double s,
    const std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
    return [t, q, s, &B, &bs](double y) {
        double u = t * std::pow(y + 1.0, 2) / 4.0;
        return (std::exp(q * u) / (s * std::sqrt(t - u))) * 
            Utils::NCDF(bs.getDplus(t - u, B(t) / B(u)));
    };
}

std::function<double(double)> K3I(
    double t, 
    double r, 
    double s,
    const std::function<double(double)>& B,
    const BlackScholes& bs
)
{
    return [t, r, s, &B, &bs](double y) {
        double u = t * std::pow(y + 1.0, 2) / 4.0;
        return (std::exp(r * u) / (s * std::sqrt(t - u))) * 
        Utils::NCDF(bs.getDminus(t - u, B(t) / B(u)));
    };
}

double K1(
    double t, 
    double q, 
    double s,
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
	auto integrand = K1I(t, q, s, B, bs);
	return std::exp(q * t) / 2.0 * t * quadSum(integrand, n, w);
}

double K2(
    double t, 
    double q, 
    double s,
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
	auto integrand = K2I(t, q, s, B, bs);
	return std::exp(q * t) * std::sqrt(t) * quadSum(integrand, n, w);
}

double K3(
    double t, 
    double r,
    double s,
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
	auto integrand = K3I(t, r, s, B, bs);
	return std::exp(r * t) * std::sqrt(t) * quadSum(integrand, n, w);
}

double N(
    double t, 
    double r, 
    double s,
    double K,
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
)
{
	return Utils::NCDF(bs.getDminus(t, B(t) / K)) / (s * std::sqrt(t)) + r * 
    K3(t, r, s, n, w, B, bs);
}

double D(
    double t, 
    double q, 
    double s,
    double K,
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
)
{
	return Utils::NCDF(bs.getDplus(t, B(t) / K)) / (s * std::sqrt(t)) + 
    Utils::NCDF(bs.getDplus(t, B(t) / K)) + q * (K1(t, q, s, n, w, B, bs) + 
    K2(t, q, s, n, w, B, bs));
}

double K_(double t, double r, double q, double K) 
{
    return K * std::exp(-(r - q) * t);
}

std::function<double(double)> N_I (
    double t, 
    double r, 
    double s,
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
    return [t, r, s, &B, &bs](double y) {
        double u = t * std::pow(y + 1.0, 2) / 4.0;
        return (std::exp(r * u) * bs.getDminus(t, B(t) / B(u))) / (B(t) * 
        std::pow(s, 2) * (t - u)) * Utils::NCDF(bs.getDminus(t - u, B(t) / B(u)));
    };
}

double N_ (
    double t, 
    double r, 
    double s, 
    double K, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
    auto integrand = N_I(t, r, s, B, bs);
    return -bs.getDminus(t, B(t) / K) * Utils::NCDF(bs.getDminus(t, B(t))) / 
    (B(t) * std::pow(s, 2) * t) - r * quadSum(integrand, n, w); 

}

std::function<double(double)> D_I(
    double t,
    double r,
    double s,
    double K,  
    std::function<double(double)>& B,
    const BlackScholes& bs
)
{
    return [t, r, s, K, &B, &bs](double y) {
        double u = t * std::pow(y + 1.0, 2) / 4.0;
        return (B(u) / K) * (std::exp(r * u) * bs.getDminus(t - u, B(t) / B(u))) / 
               (s * s * (t - u)) * Utils::NCDF(bs.getDminus(t - u, B(t) / B(u)));
    };
}

double D_ (
    double t, 
    double r, 
    double q,
    double s, 
    double K,
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{ 
    auto integrand = D_I(t, r, s, K, B, bs);

    return - K_(t, r, q, K) / B(t) * bs.getDminus(t, B(t) / K) * 
    Utils::NPDF(bs.getDminus(t, B(t) / K)) / (B(t) * std::pow(s, 2) * t) - 
    q * K_(t, r, q, K) / B(t) * quadSum(integrand, n , w);
}

double f(
    double t, 
    double r, 
    double q,
    double s,
    double K,
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
    return K_(t, r, q, K) * N(t, r, s, K, n, w, B, bs) / D(t, q, s, K, n, w, B, bs);
}

double f_ (
    double t, 
    double r, 
    double q, 
    double s,
    double K,
    std::function<double(double)>& B,
    std::vector<double> w, 
    std::vector<double> n,
    const BlackScholes& bs
    ) 
{
    return K_(t, r, q, K) * N_(t, r, s, K, n, w, B, bs) / D(t, q, s, K, n, w, B, bs) - 
    D_(t, r, q, s, K, n, w, B, bs) * N(t, r, s, K, n, w, B, bs) / 
    std::pow(D(t, q, s, K, n, w, B, bs), 2);
}

