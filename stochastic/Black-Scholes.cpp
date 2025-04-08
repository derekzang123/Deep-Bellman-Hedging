#include <string>

enum class OptionType { Call, Put };

class BlackScholes { 
    private:
        double S; // Spot price
        double K; // Strike price
        double r; // Risk-free rate
        double q; // Dividend yield
        double T; // Time to maturity
        int N; // Number of simulations
        int steps; // Number of time steps; 
        OptionType type; // Call or Put

        // TODO: initialize some randome number generator and normal distribution
    
    public:
};