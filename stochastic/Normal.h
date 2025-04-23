#ifndef NORMAL_DISTRIBUTION_H
#define NORMAL_DISTRIBUTION_H

namespace Normal {
    double PDF(double x, double mean, double stddev);
    double CDF(double x, double mean, double stddev);
}

#endif