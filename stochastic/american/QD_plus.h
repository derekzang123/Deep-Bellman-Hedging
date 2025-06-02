#ifndef QD_PLUS_H
#define QD_PLUS_H

#include "../Utils.h"
#include "../BlackScholes.h"
#include <vector>

double p(double S, double t, double K, const BlackScholes &bs);

double d1(double S, double t, double K, const BlackScholes &bs);

double d2(double S, double t, double K, const BlackScholes &bs);

std::vector<double> QD_(const BlackScholes &bs, std::vector<double> &tVec,
    int maxIter, double ts, double tf);

#endif 