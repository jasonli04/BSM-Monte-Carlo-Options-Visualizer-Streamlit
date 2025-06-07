import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(
        self, 
        vol: float, 
        underlyingPrice: float, 
        strikePrice: float, 
        timeToExp: int,
        riskFreeRate: float):

        self.volatility = vol
        self.underlying_Price = underlyingPrice
        self.strike_Price = strikePrice
        self.time_to_expiration = timeToExp
        self.risk_free_rate = riskFreeRate
    
    def calcPrice(self, option_type='call'):
        S, K, T, r, sigma = self.underlying_Price, self.strike_Price, self.time_to_expiration, self.risk_free_rate, self.volatility
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def monte_carlo_simulation(self, num_simulations=500):
        num_steps = 252
        dt = self.time_to_expiration / num_steps
        S = np.zeros((num_simulations, num_steps + 1))
        S[:, 0] = self.underlying_Price
        for t in range(1, num_steps + 1):
            Z = np.random.standard_normal(num_simulations)
            S[:, t] = S[:, t-1] * np.exp((self.risk_free_rate - 0.5 * self.volatility**2) * dt + self.volatility * np.sqrt(dt) * Z)
        
        # Calculate payoffs for both call and put options using the final stock prices
        final_stock_prices = S[:, -1]
        call_payoffs = np.maximum(final_stock_prices - self.strike_Price, 0)
        put_payoffs = np.maximum(self.strike_Price - final_stock_prices, 0)

        # Calculate discounted option prices
        call_option_price = np.exp(-self.risk_free_rate * self.time_to_expiration) * np.mean(call_payoffs)
        put_option_price = np.exp(-self.risk_free_rate * self.time_to_expiration) * np.mean(put_payoffs)

        return call_option_price, put_option_price, S
