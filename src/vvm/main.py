"""
Vanna–Volga Implied Volatility Surface Builder

This program computes the smile‐consistent implied volatility for equity index options 
using the Vanna–Volga method. It accepts market option quotes (bid/ask prices) and uses 
iterative procedures to derive the ATM strike and the 25Δ call/put strikes. It then computes 
the full Vanna–Volga price as well as first‐ and second‐order approximations for implied volatility.
Finally, it builds and visualizes the full implied volatility surface and allows the user 
to query the surface for a specific strike and expiry.

Requirements:
    - Python 3.10+
    - NumPy, pandas, SciPy, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq

###############################################################################
# Black-Scholes Pricing and Implied Volatility Functions
###############################################################################

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute the Black-Scholes call option price.
    
    Parameters:
        S: underlying price
        K: strike price
        T: time to maturity (in years)
        r: risk-free rate
        sigma: volatility
        
    Returns:
        Call option price.
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

def implied_volatility(target_price: float, S: float, K: float, T: float, r: float, 
                       initial_guess: float = 0.2) -> float:
    """
    Compute the implied volatility given a target call price by minimizing the absolute error.
    
    Parameters:
        target_price: observed option price
        S: underlying price
        K: strike
        T: time to maturity
        r: risk-free rate
        initial_guess: starting volatility for the solver
        
    Returns:
        Implied volatility.
    """
    def objective(sigma):
        price = black_scholes_call(S, K, T, r, sigma)
        return abs(price - target_price)
    
    res = minimize_scalar(objective, bounds=(1e-6, 5.0), method='bounded')
    if res.success:
        return res.x
    else:
        raise RuntimeError("Implied volatility optimization did not converge.")

def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Compute the Black-Scholes Vega (sensitivity of price to volatility).
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

###############################################################################
# Vanna-Volga Weight and Pricing Functions
###############################################################################

def compute_weights(K: float, K1: float, K2: float, K3: float,
                    sigma: float, sigma1: float, sigma2: float, sigma3: float,
                    S: float, T: float, r: float) -> tuple[float, float, float]:
    """
    Compute the weights x1, x2, and x3 used in the Vanna–Volga pricing formula.
    
    Based on the formulas:
      x1 = [vega(K)/vega(K1)] * [ln(K2/K)*ln(K3/K)] / [ln(K2/K1)*ln(K3/K1)]
      x2 = [vega(K)/vega(K2)] * [ln(K/K1)*ln(K3/K)] / [ln(K2/K1)*ln(K3/K2)]
      x3 = [vega(K)/vega(K3)] * [ln(K/K1)*ln(K/K2)] / [ln(K3/K1)*ln(K3/K2)]
    """
    vega = bs_vega(S, K, T, r, sigma)
    vega1 = bs_vega(S, K1, T, r, sigma1)
    vega2 = bs_vega(S, K2, T, r, sigma2)
    vega3 = bs_vega(S, K3, T, r, sigma3)
    # Avoid division by zero
    if vega1 == 0 or vega2 == 0 or vega3 == 0:
        raise ValueError("One of the computed vegas is zero; check inputs.")
    
    term1 = np.log(K2/K) * np.log(K3/K) / (np.log(K2/K1) * np.log(K3/K1))
    term2 = np.log(K/K1) * np.log(K3/K) / (np.log(K2/K1) * np.log(K3/K2))
    term3 = np.log(K/K1) * np.log(K/K2) / (np.log(K3/K1) * np.log(K3/K2))
    
    x1 = (vega / vega1) * term1
    x2 = (vega / vega2) * term2
    x3 = (vega / vega3) * term3
    
    return x1, x2, x3

def full_vanna_volga_call(S: float, K: float, T: float, r: float, sigma: float,
                          K1: float, sigma1: float, K2: float, sigma2: float,
                          K3: float, sigma3: float, market_prices: dict) -> float:
    """
    Compute the Vanna–Volga call option price using the full pricing formula:
    
       CVV(K) = CBS(K) + sum_{i=1}^{3} x_i(K) * (CM(K_i) - CBS(K_i))
       
    where CBS(K) is the Black-Scholes price using volatility sigma (or sigma_i at K_i) and 
    market_prices[K_i] is the observed market price for strike K_i.
    
    Parameters:
        S, T, r: underlying price, time to expiry, risk-free rate.
        sigma: volatility used for strike K.
        (K1, sigma1), (K2, sigma2), (K3, sigma3): reference strikes and their implied volatilities.
        market_prices: dictionary mapping each K_i to its market observed call price.
    
    Returns:
        The Vanna–Volga adjusted call price.
    """
    CBS_K  = black_scholes_call(S, K,  T, r, sigma)
    CBS_K1 = black_scholes_call(S, K1, T, r, sigma1)
    CBS_K2 = black_scholes_call(S, K2, T, r, sigma2)
    CBS_K3 = black_scholes_call(S, K3, T, r, sigma3)
    
    x1, x2, x3 = compute_weights(K, K1, K2, K3, sigma, sigma1, sigma2, sigma3, S, T, r)
    
    CM_K1 = market_prices[K1]
    CM_K2 = market_prices[K2]
    CM_K3 = market_prices[K3]
    
    CVV = CBS_K + x1 * (CM_K1 - CBS_K1) + x2 * (CM_K2 - CBS_K2) + x3 * (CM_K3 - CBS_K3)
    return CVV

###############################################################################
# Approximations for Implied Volatility
###############################################################################

def first_order_approximation(K: float, K1: float, K2: float, K3: float,
                              sigma1: float, sigma2: float, sigma3: float) -> float:
    """
    Compute the first-order approximation of the implied volatility as a weighted average:
    
         sigma_first ≈ X1(K)*sigma1 + X2(K)*sigma2 + X3(K)*sigma3
        
    where the weights are defined by:
         X1(K) = ln(K2/K)*ln(K3/K) / [ln(K2/K1)*ln(K3/K1)]
         X2(K) = ln(K/K1)*ln(K3/K) / [ln(K2/K1)*ln(K3/K2)]
         X3(K) = ln(K/K1)*ln(K/K2) / [ln(K3/K1)*ln(K3/K2)]
    """
    X1 = (np.log(K2/K) * np.log(K3/K)) / (np.log(K2/K1) * np.log(K3/K1))
    X2 = (np.log(K/K1) * np.log(K3/K)) / (np.log(K2/K1) * np.log(K3/K2))
    X3 = (np.log(K/K1) * np.log(K/K2)) / (np.log(K3/K1) * np.log(K3/K2))
    sigma_first = X1 * sigma1 + X2 * sigma2 + X3 * sigma3
    return sigma_first

def second_order_approximation(K: float, K1: float, K2: float, K3: float,
                               sigma_atm: float, sigma1: float, sigma2: float, sigma3: float,
                               S: float, T: float, r: float) -> float:
    """
    Compute the second-order approximation of the implied volatility using the formula:
    
         sigma_second ≈ sigma_atm + (-sigma_atm + sqrt( sigma_atm^2 + d1*d2*(2*sigma_atm*D1 + D2) ))
                         / (d1*d2)
                         
    where:
         d1 = (ln(S/K) + (r + 0.5*sigma_atm^2)*T)/(sigma_atm*sqrt(T))
         d2 = d1 - sigma_atm*sqrt(T)
         D1 = X1*sigma1 + X2*sigma2 + X3*sigma3 - sigma_atm
         D2 = X1*d1(K1)*d2(K1)*(sigma1-sigma2)**2 + X3*d1(K3)*d2(K3)*(sigma3-sigma2)**2
         and the weights X1, X2, X3 are as defined in first_order_approximation.
         
    Note: This implementation is sensitive to d1*d2 (avoid division by zero) and the radicand must be non-negative.
    """
    # Compute d1 and d2 at strike K using ATM volatility sigma_atm
    d1 = (np.log(S/K) + (r + 0.5*sigma_atm**2)*T) / (sigma_atm * np.sqrt(T))
    d2 = d1 - sigma_atm * np.sqrt(T)
    
    # Compute weights X1, X2, X3 for strike K
    X1 = (np.log(K2/K) * np.log(K3/K)) / (np.log(K2/K1) * np.log(K3/K1))
    X2 = (np.log(K/K1) * np.log(K3/K)) / (np.log(K2/K1) * np.log(K3/K2))
    X3 = (np.log(K/K1) * np.log(K/K2)) / (np.log(K3/K1) * np.log(K3/K2))
    
    D1 = X1 * sigma1 + X2 * sigma2 + X3 * sigma3 - sigma_atm

    # Compute d1 and d2 for K1 and K3 using sigma1 and sigma3 respectively
    d1_K1 = (np.log(S/K1) + (r + 0.5*sigma1**2)*T) / (sigma1 * np.sqrt(T))
    d2_K1 = d1_K1 - sigma1 * np.sqrt(T)
    d1_K3 = (np.log(S/K3) + (r + 0.5*sigma3**2)*T) / (sigma3 * np.sqrt(T))
    d2_K3 = d1_K3 - sigma3 * np.sqrt(T)
    
    D2 = X1 * d1_K1 * d2_K1 * (sigma1 - sigma2)**2 + X3 * d1_K3 * d2_K3 * (sigma3 - sigma2)**2
    
    denom = d1 * d2
    if np.abs(denom) < 1e-8:
        raise ZeroDivisionError("d1*d2 is too small; check inputs.")
    
    radicand = sigma_atm**2 + d1 * d2 * (2 * sigma_atm * D1 + D2)
    if radicand < 0:
        radicand = 0  # enforce non-negative
    
    sigma_second = sigma_atm + (-sigma_atm + np.sqrt(radicand)) / (d1 * d2)
    return sigma_second

###############################################################################
# Iterative Procedures to Determine ATM and 25Δ Strikes
###############################################################################

def compute_atm_strike_and_vol(S: float, T: float, r: float, initial_sigma: float) -> tuple[float, float]:
    """
    Compute the ATM strike (zero-delta straddle) and its implied volatility via an iterative procedure.
    
    Here we use the relation K_ATM = S * exp((r + 0.5*sigma^2)*T) and assume q = 0.
    In practice one would adjust q from the forward price via put-call parity.
    
    Returns:
        (K_atm, sigma_atm)
    """
    tol = 1e-6
    max_iter = 100
    sigma = initial_sigma
    K_old = S * np.exp((r + 0.5 * sigma**2) * T)
    
    for i in range(max_iter):
        # Compute Black-Scholes call price for strike K_old
        price = black_scholes_call(S, K_old, T, r, sigma)
        # Invert the price to get an updated volatility (here we simply reuse sigma from the market)
        sigma_new = implied_volatility(price, S, K_old, T, r, initial_guess=sigma)
        K_new = S * np.exp((r + 0.5 * sigma_new**2) * T)
        if abs(K_new - K_old) < tol:
            return K_new, sigma_new
        K_old = K_new
        sigma = sigma_new
    raise RuntimeError("ATM strike iteration did not converge.")

def compute_delta_strike(S: float, T: float, r: float, sigma: float, delta: float, option_type: str = 'call') -> float:
    """
    Compute the strike corresponding to a given option delta by solving:
    
         delta = exp(-r*T)*N(d1)   (for call)
         delta = -exp(-r*T)*N(-d1) (for put)
         
    Parameters:
         delta: target delta (positive for call, negative for put)
         option_type: 'call' or 'put'
    """
    def objective(K):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            calc_delta = np.exp(-r*T) * norm.cdf(d1)
        else:
            calc_delta = -np.exp(-r*T) * norm.cdf(-d1)
        return calc_delta - delta

    # Provide a reasonable bracket (this may be adjusted)
    K_low = 0.5 * S
    K_high = 1.5 * S
    try:
        K_sol = brentq(objective, K_low, K_high)
    except ValueError as e:
        raise ValueError("Could not find a strike for the given delta.") from e
    return K_sol

###############################################################################
# Volatility Surface Construction and Visualization
###############################################################################

def build_vol_surface(S: float, expiries: np.ndarray, strike_grid: np.ndarray, r: float,
                      market_data: dict) -> pd.DataFrame:
    """
    Build an implied volatility surface using the Vanna–Volga methods.
    
    Parameters:
         S: underlying price.
         expiries: 1D numpy array of expiries (in years).
         strike_grid: 1D numpy array of strikes.
         r: risk-free rate.
         market_data: dictionary keyed by expiry; each value is a dict containing
                      the market quotes for that expiry (including ATM, 25Δ call/put strikes
                      and their corresponding market prices and implied vols).
                      For example:
                      {
                        T1: {'K1': ..., 'sigma1': ..., 'K2': ..., 'sigma2': ...,
                             'K3': ..., 'sigma3': ..., 'market_prices': {K1:..., K2:..., K3:...},
                             'atm_initial_sigma': ...},
                        ...
                      }
    
    Returns:
         A DataFrame with rows indexed by expiry and columns by strike, containing the implied vol.
         (Here we demonstrate the full Vanna–Volga method; one could also store the approximations.)
    """
    vol_surface = np.zeros((len(expiries), len(strike_grid)))
    
    for i, T in enumerate(expiries):
        try:
            # For each expiry, use the market data to compute the key strikes and vols.
            data = market_data[T]
            # Determine the ATM strike and vol via iteration.
            K_atm, sigma_atm = compute_atm_strike_and_vol(S, T, r, data['atm_initial_sigma'])
            # Compute the 25Δ strikes (call and put)
            K_25call = compute_delta_strike(S, T, r, data['sigma_call_guess'], 0.25, option_type='call')
            K_25put  = compute_delta_strike(S, T, r, data['sigma_put_guess'], -0.25, option_type='put')
            
            # Set K1, K2, K3 for Vanna-Volga (ensure K1 < K_atm < K3)
            K1, K2, K3 = K_25put, K_atm, K_25call
            sigma1, sigma2, sigma3 = data['sigma_put'], sigma_atm, data['sigma_call']
            market_prices = data['market_prices']  # should contain keys: K1, K2, K3
            
            for j, K in enumerate(strike_grid):
                # Compute full Vanna-Volga price for strike K using our computed parameters.
                try:
                    CVV = full_vanna_volga_call(S, K, T, r, sigma2, K1, sigma1, K2, sigma2, K3, sigma3, market_prices)
                    # Invert the Black-Scholes price to get the Vanna-Volga implied volatility.
                    vol = implied_volatility(CVV, S, K, T, r, initial_guess=sigma2)
                except Exception as ex:
                    vol = np.nan  # if error, assign NaN
                vol_surface[i, j] = vol
        except Exception as e:
            print(f"Warning: Could not compute vol surface for expiry T={T}: {e}")
            vol_surface[i, :] = np.nan
            
    df_surface = pd.DataFrame(vol_surface, index=expiries, columns=strike_grid)
    return df_surface

def plot_vol_surface(vol_surface: pd.DataFrame) -> None:
    """
    Plot the implied volatility surface as a contour plot.
    """
    expiries = vol_surface.index.values
    strikes = vol_surface.columns.values
    X, Y = np.meshgrid(strikes, expiries)
    Z = vol_surface.values

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel("Strike")
    ax.set_ylabel("Expiry (years)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("Implied Volatility Surface (Vanna–Volga)")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

###############################################################################
# Interactive Query Interface
###############################################################################

def query_surface(vol_surface: pd.DataFrame) -> None:
    """
    Allow the user to query the implied volatility surface for a specific strike and expiry.
    """
    try:
        strike_input = float(input("Enter strike: "))
        expiry_input = float(input("Enter expiry (in years): "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return
    
    # Find the nearest grid points
    strikes = vol_surface.columns.values.astype(float)
    expiries = vol_surface.index.values.astype(float)
    idx = (np.abs(expiries - expiry_input)).argmin()
    jdx = (np.abs(strikes - strike_input)).argmin()
    vol = vol_surface.iloc[idx, jdx]
    print(f"Nearest grid: Expiry={expiries[idx]:.4f} years, Strike={strikes[jdx]:.4f}")
    print(f"Implied volatility = {vol:.4f}")

###############################################################################
# Main Routine
###############################################################################

def main():
    # Example inputs:
    # Underlying price, risk-free rate, and an array of expiries (in years)
    S = 2091.58
    r = 0.02
    expiries = np.array([0.02, 0.08, 0.25, 0.5, 1.0])  # e.g., 7 days, 1 month, 3 months, 6 months, 1 year

    # Define a grid of strikes; for simplicity, we use a relative range around S
    strike_grid = np.linspace(0.8 * S, 1.2 * S, 50)
    
    # Market data: for each expiry, we need a dictionary containing:
    #  - 'atm_initial_sigma': an initial guess for the ATM volatility,
    #  - 'sigma_call_guess' and 'sigma_put_guess': initial guesses for the 25Δ call and put vol,
    #  - 'sigma_call' and 'sigma_put': observed implied vols for the 25Δ call and put,
    #  - 'market_prices': a dictionary mapping the reference strikes (K1, K2, K3) to observed call prices.
    # For demonstration purposes we use dummy numbers.
    market_data = {}
    for T in expiries:
        market_data[T] = {
            'atm_initial_sigma': 0.15,  # initial guess
            'sigma_call_guess': 0.16,
            'sigma_put_guess': 0.14,
            'sigma_call': 0.16,
            'sigma_put': 0.14,
            # In practice, these strikes and prices would be derived from the market quotes.
            # Here we use the following dummy values:
            # Assume K2 (ATM) is computed via iteration; for now we take it as S*exp((r+0.5*0.15^2)*T)
            'market_prices': {}
        }
        # For the sake of example, we set the reference strikes as:
        K_atm_dummy = S * np.exp((r + 0.5 * 0.15**2) * T)
        K_25call_dummy = S * 1.01
        K_25put_dummy = S * 0.99
        market_data[T]['market_prices'][K_25put_dummy] = black_scholes_call(S, K_25put_dummy, T, r, 0.14) * 1.02
        market_data[T]['market_prices'][K_atm_dummy] = black_scholes_call(S, K_atm_dummy, T, r, 0.15) * 1.01
        market_data[T]['market_prices'][K_25call_dummy] = black_scholes_call(S, K_25call_dummy, T, r, 0.16) * 1.02

    # Build the volatility surface
    vol_surface = build_vol_surface(S, expiries, strike_grid, r, market_data)
    
    # Visualize the surface
    plot_vol_surface(vol_surface)
    
    # Allow user to query the surface interactively
    query_surface(vol_surface)

if __name__ == '__main__':
    obb.user.preferences.output_type = "dataframe"
    options = obb.derivatives.options.chains("AAPL", provider="cboe")
    """
    underlying_symbol	underlying_price	contract_symbol	expiration	dte	strike	option_type	open_interest	volume	theoretical_price	...	low	prev_close	change	change_percent	implied_volatility	delta	gamma	theta	vega	rho
    """
    print(options.head())

    main()
