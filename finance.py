import pandas as pd
import yfinance as yf

msft = yf.Ticker('FB')
history = msft.history(start="2020-01-01", interval='1d')
print(history.head())