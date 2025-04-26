#ifndef FAST_AMERICAN_OPTION_PRICING_H
#define FAST_AMERICAN_OPTION_PRICING_H

#include <vector>
#include <numeric>
#include <functional>

const double PI = atan(1) * 4;

std::pair<std::vector<double>, std::vector<double>> computeNodes(int n, int tMax);
std::vector<double> computeCollocation(std::vector<double> xVec);
std::pair<std::vector<double>, std::vector<double>> quadrature(int l, double h);
std::vector<double> cWeights(std::vector<double> H);
std::vector<double> cBasis(double z, int N, bool F);
double qC(double z, const std::vector<double>& a);
double quadSum (std::function<double(double)>& f, std::vector<double> n, std::vector<double> w);

auto K1I(double t, double q, double sigma,
         std::function<double(double)>& B);

auto K2I(double t, double q, double sigma,
         std::function<double(double)>& B);

auto K3I(double t, double r, double sigma,
         std::function<double(double)>& B);

double computeK1(double tau, double q, double sigma,
                 std::vector<double>& nodes,
                 std::vector<double>& weights,
                 std::function<double(double)>& B);

double computeK2(double tau, double q, double sigma,
                 std::vector<double>& nodes,
                 std::vector<double>& weights,
                 std::function<double(double)>& B);

double computeK3(double tau, double r, double sigma,
                 std::vector<double>& nodes,
                 std::vector<double>& weights,
                 std::function<double(double)>& B);

#endif