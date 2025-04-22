#include "BlackScholes.h"
#include "american/FastAmericanOptionPricing.h"
#include "american/QD_plus.h"


BlackScholes::BlackScholes(double S_, double K_, double r_, double q_, double T_, double vol_, 
    int N_, int steps_, OptionType type_, ExerciseStyle style_)
: S(S_), K(K_), r(r_), q(q_), T(T_), vol(vol_), N(N_), steps(steps_), type(type_), style(style_),
rng(std::random_device{}()), dist(0.0, 1.0) {
}

double BlackScholes::price() const {
    if (style == ExerciseStyle::American) {
        return priceAmerican(1,2,3,4.0); // Placeholder values -- must determine n , m, l, taumax
    }
    return priceEuropean();
}

double BlackScholes::priceAmerican(int n, int m, int l, double tauMax) const {
    auto xVec = computeNodes(n + 1, tauMax); // Step 1
    auto tauNodes = computeCollocation(xVec);

    QDPlus qd(*this, n, m, l, tauMax); // init QD object
    qd.initBoundary();

    auto B = qd.getBoundary();

    double X = q > r ? (K * (r/q)) : K;

    
    
    // Remainder of Fast American logic
    return 0.0;
}
