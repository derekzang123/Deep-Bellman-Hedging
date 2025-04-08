#include <vector> 
#include <cmath> 

/*
STEP 1: Compute Chebyshev Nodes
*/
std::vector<double> computeNodes(int n, int tMax)
{ 
    std::vector<double> zVec, xVec;
    for (int i = 0; i < n; i++) {
        double z = -1 * std::cos((i*M_PI)/n);
        double x = std::sqrt(tMax) / 2;
        x *= 1 + z;
        xVec.push_back(x);
    }
    return xVec;
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
        double n = std::tanh(0.5 * M_PI * std::sinh(k*h));
        double w = (0.5 * h * std::cosh(k * h)) / std::pow(std::cosh(0.5 * M_PI * std::sinh(k * h)), 2);
        nodes.push_back(n);
        weights.push_back(w);
        if (k > 0 || l % 2 == 0) {
            nodes.push_back(-n);
            weights.push_back(w);
        }
    }
    return {nodes, weights};
}

/* STEP 3: QD+ approximation */
struct HestonParams { 
    double strike;
    double volatility;
};

std::vector<double> qd_plus (std::vector<double> tau, const HestonParams& params) 
{
    return {};
}