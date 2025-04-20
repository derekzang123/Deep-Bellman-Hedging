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

    double getSpot() {
        return S;
    }

    double getStrike() {
        return K;
    }

    double getRate() {
        return r;
    }

    double getDividend() {
        return q;
    }

    double getVolatility() {
        return vol;
    }

    OptionType getType() { return type; }

    ExerciseStyle getStyle() { return style; }

    double price() const;
    double priceEuropean() const; 
    double priceAmerican(int n, int m, int l, double tauMax) const;
};

#endif