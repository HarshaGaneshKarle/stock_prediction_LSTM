import yfinance as yf

msft = yf.Ticker("MSFT")
df = msft.history(period="max")
df.to_csv("MSFT_data.csv")

print("Data saved as MSFT_data.csv")
