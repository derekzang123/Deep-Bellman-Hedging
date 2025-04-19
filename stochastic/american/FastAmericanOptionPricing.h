#ifndef FAST_AMERICAN_OPTION_PRICING_H
#define FAST_AMERICAN_OPTION_PRICING_H

#include <vector>

const double PI = atan(1) * 4;

enum class OptionType { Call, Put };

struct BlackScholesParams {
    double K;
    double vol;
    double r;
    double q;
    OptionType type;
};

std::vector<double> computeNodes(int n, int tMax);
std::vector<double> computeCollocation(std::vector<double> xVec);
std::vector<std::vector<double>> quadrature(int l, double h);

#endif