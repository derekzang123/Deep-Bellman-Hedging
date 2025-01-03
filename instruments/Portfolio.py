import torch
from BaseInstrument import BaseInstrument

class Portfolio:
    def __init__(self, instruments, weights):
        self.instruments = instruments
        self.weights = [1.0] # placeholder

    def simulate_cashflows(self, instrument, t):
        """

        Args:
            instrument: financial instrument
            t: time

        Returns:
            Simulates the cashflows of an instrument at time t

        """
        return instrument.calculate_cashflows(t)

    def aggregate_cashflows(self, t):
        """

        Args:
            t: time

        Returns:
            Cashflows of the entire portfolio, with weights applied, at time t
        """
        total_cashflows = torch.zeros_like(self.instruments[0].cashflows)

        for i, instrument in enumerate(self.instruments):
            total_cashflows += self.weights[i] * self.simulate_cashflows(instrument, t)

        return total_cashflows
