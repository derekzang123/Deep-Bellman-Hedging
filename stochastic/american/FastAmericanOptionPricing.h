#ifndef FAST_AMERICAN_OPTION_PRICING_H
#define FAST_AMERICAN_OPTION_PRICING_H

#include <vector>
#include <numeric>
#include <functional>
#include "../BlackScholes.h"

const double PI = atan(1) * 4;

std::pair<std::vector<double>, std::vector<double>> computeNodes(int n, int tMax);
std::vector<double> computeCollocation(std::vector<double> xVec);
std::pair<std::vector<double>, std::vector<double>> quadrature(int l, double h);
std::vector<double> cWeights(std::vector<double> H);
std::vector<double> cBasis(double z, int N, bool F);
double qC(double z, const std::vector<double>& a);
double quadSum (std::function<double(double)>& f, std::vector<double> n, std::vector<double> w);

// Helper integrands for K1-K3
std::function<double(double)> K1I(double t, double q, double sigma,
    std::function<double(double)>& B, const BlackScholes& bs
);

std::function<double(double)> K2I(double t, double q, double sigma,
    std::function<double(double)>& B, const BlackScholes& bs
);
 
std::function<double(double)> K3I(double t, double r, double sigma,
    std::function<double(double)>& B, const BlackScholes& bs
);


// Compute K1-K3
double K1(double t, double q, double sigma,
    std::vector<double>& nodes,
    std::vector<double>& weights,
    std::function<double(double)>& B,
    const BlackScholes& bs
);

double K2(double t, double q, double sigma,
    std::vector<double>& nodes,
    std::vector<double>& weights,
    std::function<double(double)>& B,
    const BlackScholes& bs
);

double K3(double t, double r, double sigma,
    std::vector<double>& nodes,
    std::vector<double>& weights,
    std::function<double(double)>& B, 
    const BlackScholes& bs
);

// N and D terms 
double N(double t, double r, double s, double K,
    std::vector<double>& n, std::vector<double>& w,
	std::function<double(double)>& B,
    const BlackScholes& bs
);

double D(double t, double q, double s,double K, std::vector<double>& n,
    std::vector<double>& w, std::function<double(double)>& B,
    const BlackScholes& bs
);

// K star
double K_(double t, double r, double q, double K);

// N and D helper integrands for N_ and D_
std::function<double(double)> N_I (double t, double r, double s,
    std::function<double(double)>& B, const BlackScholes& bs
); 

std::function<double(double)> D_I(double t, double r, double s, double K, 
    std::function<double(double)>& B, const BlackScholes& bs
);

// N and D prime
double N_ (double t, double r, double s, double K, std::vector<double>& n,
    std::vector<double>& w, std::function<double(double)>& B,
    const BlackScholes& bs
);

double D_ (double t, double r, double q, double s, double K,
    std::vector<double>& n, std::vector<double>& w,
    std::function<double(double)>& B, const BlackScholes& bs
);

// F and F prime
double f(double t, double r, double q,double s,double K, std::vector<double>& n,
    std::vector<double>& w, std::function<double(double)>& B, const BlackScholes& bs
);

double f_ (double t, double r, double q, double s, double K,
    std::function<double(double)>& B, std::vector<double> w, 
    std::vector<double> n, const BlackScholes& bs
); 




#endif