#include "FastAmericanOptionPricing.h"
#include <vector> 
#include <cmath> 

/* STEP 1: Compute Chebyshev Nodes */
std::vector<std::vector<double>> computeNodes(int n, int tMax) { 
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

std::vector<double> computeCollocation(std::vector<double> xVec) 
{
    std::vector<double> tau;
    for (int i = 0; i <= xVec.size() - 1; i++) {
        double t = std::pow(xVec[i], 2);
        tau.push_back(t);
    }
    return tau;
}


/* STEP 2: Tanh-Sinh Quadrature*/
std::vector<std::vector<double>> quadrature(int l, double h) 
{
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
