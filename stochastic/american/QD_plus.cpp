#include "QD_plus.h"
#include <cmath>

QDPlus::QDPlus(const BlackScholes& bs_, int n_, int m_, int l_, double tauMax_)
    : bs(bs_),
      n(n_),
      m(m_),
      l(l_),
      tauMax(tauMax_)
{
 // placeholder 
}


void QDPlus::initBoundary() { 
    boundary.resize(n + 1);
    
    if (bs.getType() == OptionType::Call) { 
        boundary[0] = bs.getStrike() * std::max(1.0, bs.getRate() / bs.getDividend());
    }
    else {
        boundary[0] = bs.getStrike() * std::min(1.0, bs.getRate() / bs.getDividend());
    }

    for(size_t i = 1; i <= n; i++) {
        double tau = tauNodes[i];
        boundary[i] = qdPlusApprox(tau);
    }
}

double QDPlus::qdPlusApprox(double tau){
    return tau; // placeholder for now
}

