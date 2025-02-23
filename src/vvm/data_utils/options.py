
from openbb import obb
import pandas as pd

# Set OpenBB preferences to return a pandas DataFrame
obb.user.preferences.output_type = "dataframe"

def get_options_data(symbol: str = "VIX", provider: str = "cboe") -> pd.DataFrame:
    """
    Retrieve the options chain data for a given symbol using OpenBB.
    
    Parameters:
        symbol (str): The underlying symbol for options data (default is "VIX").
        provider (str): The data provider (default is "cboe").
    
    Returns:
        options_df: A DataFrame containing combined options data for calls and puts.
                      Expected columns include:
                      - underlying_symbol, underlying_price, contract_symbol,
                      - expiration, dte, strike, option_type, open_interest,
                      - volume, theoretical_price, low, prev_close, change,
                      - change_percent, implied_volatility, delta, gamma, theta,
                      - vega, and rho.
    """
    options_df = obb.derivatives.options.chains(symbol, provider=provider)
    if not isinstance(options_df, pd.DataFrame):
         options_df = pd.DataFrame(options_df)
    return options_df
   
if __name__ == "__main__":
    # Example usage
    df_options = get_options_data()
    print(df_options.head())