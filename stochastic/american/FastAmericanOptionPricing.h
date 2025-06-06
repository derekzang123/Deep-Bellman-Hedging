#ifndef FAST_AMERICAN_OPTION_PRICING_H
#define FAST_AMERICAN_OPTION_PRICING_H

#include <vector>
#include <numeric>
#include <functional>
#include "../BlackScholes.h"

const double PI = atan(1) * 4;

std::vector<double> computeNodes(int N, int tM);

std::vector<double> computeCollocation(std::vector<double>& xVec);

std::vector<double> cWeights(std::vector<double> H);

double qC(double z, const std::vector<double>& a);

std::pair<std::vector<double>, std::vector<double>> quadNodes(int l, double h);

double quadSum (std::function<double(double)>& f, std::vector<double> n, std::vector<double> w);

// K1, K2, K3 Integrands + Expressions
std::function<double(double)> K1I(
    double t, 
    const std::function<double(double)>& B,
    const BlackScholes& bs
);

std::function<double(double)> K2I(
    double t, 
    const std::function<double(double)>& B,
    const BlackScholes& bs
); 
 
std::function<double(double)> K3I(
    double t, 
    const std::function<double(double)>& B,
    const BlackScholes& bs
);

double K1(
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
); 

double K2(
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
);

double K3(
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
); 

// N and D Integrands + Expressions
std::function<double(double)> N_I (
    double t, 
    std::function<double(double)>& B,
    const BlackScholes& bs
); 

double N(
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
);

std::function<double(double)> D_I(
    double t,
    std::function<double(double)>& B,
    const BlackScholes& bs
);

double D(
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
);

// N and D derivatives
double N_ (
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
); 

double D_ (
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
); 

// F and F derivatives 
double f(
    double t, 
    std::vector<double>& n,
    std::vector<double>& w,
    std::function<double(double)>& B,
    const BlackScholes& bs
); 

double f_ (
    double t, 
    std::vector<double> n,
    std::vector<double> w, 
    std::function<double(double)>& B,
    const BlackScholes& bs
);

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
); 

double V(const BlackScholes &bs, 
         double t, 
         std::function<double(double)> &B, 
         int l, 
         double h);

std::function<double(double)> V0_I(
    double t,
    const BlackScholes &bs, 
    std::function<double(double)> &B
);

std::function<double(double)> V1_I(
    double t,
    const BlackScholes &bs, 
    std::function<double(double)> &B
); 
    

// Utils
double L2N (const std::vector<double>& vec);
std::vector<double> operation (std::vector<double> a, std::vector<double> b, char op);



#endif