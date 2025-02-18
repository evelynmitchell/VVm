import pytest
from scipy.stats import norm
import numpy as np
from src.vvm.main import black_scholes_call

class TestBlackScholesCall:
    def test_black_scholes_call_valid_inputs(self):
        S = 100  # underlying price
        K = 100  # strike price
        T = 1    # time to maturity (1 year)
        r = 0.05 # risk-free rate
        sigma = 0.2 # volatility
        expected_price = S * norm.cdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) - K * np.exp(-r * T) * norm.cdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)) - sigma * np.sqrt(T))
        assert black_scholes_call(S, K, T, r, sigma) == pytest.approx(expected_price, rel=1e-9)

    def test_black_scholes_call_zero_or_negative_T(self):
        with pytest.raises(ValueError, match="T and sigma must be positive."):
            black_scholes_call(100, 100, 0, 0.05, 0.2)
        with pytest.raises(ValueError, match="T and sigma must be positive."):
            black_scholes_call(100, 100, -1, 0.05, 0.2)

    def test_black_scholes_call_zero_or_negative_sigma(self):
        with pytest.raises(ValueError, match="T and sigma must be positive."):
            black_scholes_call(100, 100, 1, 0.05, 0)
        with pytest.raises(ValueError, match="T and sigma must be positive."):
            black_scholes_call(100, 100, 1, 0.05, -0.2)