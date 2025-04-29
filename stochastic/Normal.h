#ifndef NORMAL_DISTRIBUTION_H
#define NORMAL_DISTRIBUTION_H

namespace Normal {
    double NPDF(double x, double mean, double stddev);
    double NCDF(double x, double mean, double stddev);
}

#endif