from Portfolio import Portfolio

class PortfolioManager:
    def __init__(self, market, instruments, weights=None):
        self.market = market
        self.portfolio = Portfolio(instruments, weights)

    def aggregate_cashflows(self, t):
        """
        Aggregate the cashflows of the portfolio at time `t`.
        """
        return self.portfolio.aggregate_cashflows(t)

    def update_portfolio_features(self, t):



