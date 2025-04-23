#ifndef NORMAL_DISTRIBUTION_H
#define NORMAL_DISTRIBUTION_H

namespace Normal {
    double PDF(double x, double mean = 0.0, double stddev = 1.0);
    double CDF(double x, double mean = 0.0, double stddev = 1.0);
}

#endif