from openbb import obb

#print(obb.user.credentials)
obb.user.preferences.output_type = "dataframe"
options = obb.derivatives.options.chains("AAPL", provider="cboe")

print(options.head())
output = obb.equity.price.historical("AAPL")

print(output.head())