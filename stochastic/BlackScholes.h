#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

#include <string>
#include <random>

enum class OptionType { Call, Put };
enum class ExerciseStyle { European, American };

class BlackScholes { 
private:
    double S; // Spot price
    double K; // Strike price
    double r; // Risk-free rate
    double q; // Dividend yield
    double T; // Time to maturity
    double vol;
    int N; // Number of simulations
    int steps; // Number of time steps; 
    OptionType type; // Call or Put
    ExerciseStyle style;

    std::mt19937 rng;
    std::normal_distribution<double> dist;

public: 
    BlackScholes(double S_, double K_, double r_, double q_, double T_, double vol_, 
        int N_, int steps_, OptionType type_, ExerciseStyle style_);

    double getSpot() const {
        return S;
    }

    double getStrike() const {
        return K;
    }

    double getRate() const {
        return r;
    }

    double getDividend() const {
        return q;
    }

    double getVolatility() const {
        return vol;
    }
    
    double getKStar(double t) const { 
        return K * std::exp(-(r - q) * t);
    }

    double getDminus(double tau, double z) const;

    double getDplus(double tau, double z) const;

    OptionType getType() { return type; }

    ExerciseStyle getStyle() { return style; }

    double price() const;

    double priceEuropean() const;
    
    double putPriceEuropean(double B, double t) const;

    double putThetaPrice(double t, double Bt) const;

    double priceAmerican(int n, int m, int l, double tauMax) const;
};

#endif