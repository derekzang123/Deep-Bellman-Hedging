
class Market:
    def __init__(self, market_data):
        self.market_data = market_data  # This is the historical market data over time

    def get_market_state(self, t):
        """
        Retrieve the market state (e.g., risk metrics, market features) at time `t`.

        interest rates, etc
        """
        return self.market_data[t]
