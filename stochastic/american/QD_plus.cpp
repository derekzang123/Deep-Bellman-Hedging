#include <vector>

/* STEP 3: QD+ approximation */
enum class OptionType {Call, Put};

struct BlackScholesParams { 
    double K;
    double vol;
    double r;
    double q;
    OptionType type;
}

//  = params.type == OptionType::call ? (params.K * std::max(1, params.r / params.q)) : (params.K * std::min(1, params.r / params.q));

class QDPlus {
private:
    BlackScholesParams params;
    std::vector<double> boundary;
    std::vector<double> tauNodes;
    double X;
    double tauMax;
    double n, m, l;
    
public: 
    QDPlus(const BlackScholesParams& p, int n_, int m_, int l_, double tauMax_)
        : params(p), n(n_), m(m_), l(l_), tauMax(tauMax_);
        auto xVec = computeNodes(n )
    
     
        }

        void CInterp(std::<vector> double a, std::<vector> ) {
            std::vector<double> H(n+1);
            
            for (int i = 0; i < n+1; i++){
                H[i] = std::pow()  Sum(a_k * T_k)
            }
        }
        
        
        void initBoundary() { 
            boundary.resize(n + 1);
            if (params.type = OptionType::call) { 
                boundary[0] = params.K * std::max(1, params.r / params.q);
            }
            else {
                boundary[0] = params.K * std::min(1, params.r / params.q);
            }
            
            for(size_t i = 1; i <= n; i++) {
                double tau = tauNodes[i];
                boundary[i] = qdPlusApprox(tau);
            }
        }
        
        double qdPlusApprox(double tau) {
            
        }
        
};


}

