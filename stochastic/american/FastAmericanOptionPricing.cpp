#include "FastAmericanOptionPricing.h"
#include "./QD_plus.h"
#include "../BlackScholes.h"
#include "../Utils.h"
#include <vector> 
#include <cmath> 
#include <iterator>
#include <stdexcept>


/* STEP 1: Compute Chebyshev Nodes */
std::vector<double> computeNodes(int N, int tM) { 
    std::vector<double> xVec;
    for (int i = 0; i < N; i++) {
        double z = -1 * std::cos((i*PI) / N);
        double x = std::sqrt(tM) / 2;
        x *= 1 + z;
        xVec.push_back(x);
    }
    return xVec;
}

std::vector<double> computeCollocation(std::vector<double>& xVec) {
    std::vector<double> tVec;
    for (int i = 0; i <= xVec.size() - 1; i++) {
        double t = std::pow(xVec[i], 2);
        tVec.push_back(t);
    }
    return tVec;
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

std::pair<std::vector<double>, std::vector<double>> quadNodes(int l, double h) {
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
    const std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
    double q = bs.getDividend();
    double s = bs.getVolatility();
    return [t, q, s, &B, &bs](double y) {
        double u = t * std::pow(y + 1.0, 2) / 4.0;
        return std::exp(q * u) * Utils::NCDF(bs.getDplus(t - u, B(t) / B(u)));
    };
}

std::function<double(double)> K2I(
    double t, 
    const std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
    double q = bs.getDividend();
    double s = bs.getVolatility();
    return [t, q, s, &B, &bs](double y) {
        double u = t * std::pow(y + 1.0, 2) / 4.0;
        return (std::exp(q * u) / (s * std::sqrt(t - u))) * 
            Utils::NCDF(bs.getDplus(t - u, B(t) / B(u)));
    };
}

std::function<double(double)> K3I(
    double t, 
    const std::function<double(double)>& B,
    const BlackScholes& bs
)
{
    double r = bs.getRate();
    double s = bs.getVolatility();
    return [t, r, s, &B, &bs](double y) {
        double u = t * std::pow(y + 1.0, 2) / 4.0;
        return (std::exp(r * u) / (s * std::sqrt(t - u))) * 
        Utils::NCDF(bs.getDminus(t - u, B(t) / B(u)));
    };
}

double K1(
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
	auto integrand = K1I(t, B, bs);
	return std::exp(bs.getDividend() * t) / 2.0 * t * quadSum(integrand, n, w);
}

double K2(
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
	auto integrand = K2I(t, B, bs);
	return std::exp(bs.getDividend() * t) * std::sqrt(t) * quadSum(integrand, n, w);
}

double K3(
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
	auto integrand = K3I(t, B, bs);
	return std::exp(bs.getRate() * t) * std::sqrt(t) * quadSum(integrand, n, w);
}

std::function<double(double)> N_I (
    double t, 
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
    double r = bs.getRate();
    double s = bs.getVolatility();
    return [t, r, s, &B, &bs](double y) {
        double u = t * std::pow(y + 1.0, 2) / 4.0;
        return (std::exp(r * u) * bs.getDminus(t, B(t) / B(u))) / (B(t) * 
        std::pow(s, 2) * (t - u)) * Utils::NCDF(bs.getDminus(t - u, B(t) / B(u)));
    };
}

double N(
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
)
{
	return Utils::NCDF(bs.getDminus(t, B(t) / bs.getStrike())) / (bs.getVolatility() * std::sqrt(t)) + bs.getRate() * K3(t, n, w, B, bs);
}

std::function<double(double)> D_I(
    double t,
    std::function<double(double)>& B,
    const BlackScholes& bs
)
{
    double r = bs.getRate();
    double s = bs.getVolatility();
    double K = bs.getStrike();
    return [t, r, s, K, &B, &bs](double y) {
        double u = t * std::pow(y + 1.0, 2) / 4.0;
        return (B(u) / K) * (std::exp(r * u) * bs.getDminus(t - u, B(t) / B(u))) / 
               (s * s * (t - u)) * Utils::NCDF(bs.getDminus(t - u, B(t) / B(u)));
    };
}

double D(
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
)
{
	return Utils::NCDF(bs.getDplus(t, B(t) / bs.getStrike())) / (bs.getVolatility() * std::sqrt(t)) + 
    Utils::NCDF(bs.getDplus(t, B(t) / bs.getStrike())) + bs.getDividend() * (K1(t, n, w, B, bs) + 
    K2(t, n, w, B, bs));
}

double N_ (
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
    auto integrand = N_I(t, B, bs);
    return -bs.getDminus(t, B(t) / bs.getStrike()) * Utils::NCDF(bs.getDminus(t, B(t))) / 
    (B(t) * std::pow(bs.getVolatility(), 2) * t) - bs.getRate() * quadSum(integrand, n, w); 

}


double D_ (
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{ 
    auto integrand = D_I(t, B, bs);

    return - bs.getKStar(t) / B(t) * bs.getDminus(t, B(t) / bs.getStrike()) * 
    Utils::NPDF(bs.getDminus(t, B(t) / bs.getStrike())) / (B(t) * std::pow(bs.getVolatility(), 2) * t) - 
    bs.getDividend() * bs.getKStar(t) / B(t) * quadSum(integrand, n , w);
}

double f(
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
    return bs.getKStar(t) * N(t, n, w, B, bs) / D(t, n, w, B, bs);
}

double f_ (
    double t, 
    std::vector<double> n,
    std::vector<double> w, 
    std::function<double(double)>& B,
    const BlackScholes& bs
) 
{
    return bs.getKStar(t) * N_(t, n, w, B, bs) / D(t, n, w, B, bs) - 
    D_(t, n, w, B, bs) * N(t, n, w, B, bs) / std::pow(D(t, n, w, B, bs), 2);
}

double L2N (const std::vector<double>& vec) {
    double sum = 0.0;
    for(double v : vec) { 
        sum += v*v;
    }
    return std::sqrt(sum);
}

std::vector<double> operation(
    std::vector<double> a, 
    std::vector<double> b, 
    char op
) 
{
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of equal size");
    }

    std::vector<double> result;
    result.reserve(a.size());

    if (op == '-') {
        std::transform(a.begin(), a.end(), b.begin(),
                       std::back_inserter(result),
                       [](double x, double y) { return x - y; });
    }
    else if (op == '+') {
        std::transform(a.begin(), a.end(), b.begin(),
                       std::back_inserter(result),
                       [](double x, double y) { return x + y; });
    }
    return result;
}

std::vector<double> JN(
    int N, 
    int l, 
    double h,
    double tM, 
    double tr, 
    double ts, 
    double nu,
    bool LS,
    int maxIter,
    const BlackScholes& bs
) 
{
    std::vector<double> xVec = computeNodes(N, tM);
    std::vector<double> tVec = computeCollocation(xVec);

    std::vector<double> Bvs(N) = QD(tVec, 
                                  bs.getRate(), 
                                  bs.getDividend(),
                                  bs.getVolatility(), 
                                  bs.getStrike()
                                  );
    std::vector<double> coeffs(N + 1), H(N), Bvs_(N), Fvs_(N);
    
    auto [n, w] = quadNodes(l, h);
    const double X = bs.getStrike() * std::min(1.0, bs.getRate() / bs.getDividend();

    int iter = 0;
    do {
        for (size_t i = 0; i < N; ++i) {
            H[i] = std::pow(std::log(Bvs[i] / X), 2);
        }
        coeffs = cWeights(H); 
        std::function<double(double)> B = [&](double t) {
            double z = (2 * std::sqrt(t) / std::sqrt(tM)) - 1.0;
            return X * std::exp(std::sqrt(qC(z, coeffs)));
        };
        
        for(int i = 0; i < N; i++){
            double F = f(tVec[i], n, w, B, bs);
            double F_ = f_(tVec[i], n, w, B, bs);
            Bvs_[i] = Bvs[i] + nu * (Bvs[i] - F) / (F_ - 1);
        }
          
        H.clear(); coeffs.clear();
        for (size_t i = 0; i < N; ++i) {
            H[i] = std::pow(std::log(Bvs_[i] / X), 2);
        }
        coeffs = cWeights(H); 
        std::function<double(double)> B_ = [&](double t) {
            double z = (2 * std::sqrt(t) / std::sqrt(tM)) - 1.0;
            return X * std::exp(std::sqrt(qC(z, coeffs)));
        };
        for(int i = 0; i < N; i++){
            double F_ = f(tVec[i], n, w, B_, bs);
            Fvs_[i] = F_;
        } 
          
        if (L2N(operation(Bvs_, Bvs, '-')) <= ts && 
            L2N(operation(Bvs_, Fvs_, '-')) <= tr) {
            return Bvs_;
        }
        Bvs = Bvs_;
        Bvs_.clear();
    } while (++iter < maxIter);
    return Bvs;
}