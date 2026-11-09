# LSTM Stock Price Forecasting, Tested Against a Naive Baseline

**Timeline:** 2025

A two-layer LSTM that predicts Tesla's next-day closing price from the previous 60 days, evaluated against the forecast that tomorrow's price equals today's.

## Why this project is framed around a baseline

Stock prices are highly persistent, so a model that predicts price levels will show a high R² and predictions that hug the actual price line, even if it has learned nothing useful. My first version of this project reported an R² of 0.97 and looked impressive. It also fit the scaler on the full dataset, which leaks information about the test period into training.

This version fixes both problems:

- **Train-only scaling:** the scaler is fit on training prices only
- **Naive baseline:** the LSTM is compared with a persistence forecast (tomorrow = today)
- **Directional accuracy:** how often the model calls the direction of the next move, compared with the share of up days

## Results (test period: last 20% of days, March 2024 to March 2025)

| Model | RMSE ($) | MAE ($) | MAPE | R² |
|---|---|---|---|---|
| LSTM | 14.66 | 11.86 | 4.8% | 0.972 |
| Naive (tomorrow = today) | 11.08 | 7.91 | 3.0% | 0.984 |

**The LSTM loses to the naive forecast on every metric.** Its R² of 0.97 looks strong until you see that simply repeating yesterday's price scores 0.98. Its directional accuracy is 49.2%, about the same as always guessing "up" (48.4% of test days were up days), so it has no real ability to call the next move. On the zoomed chart, the LSTM's predictions trail the actual price by about a day, which shows it mostly learned to echo recent prices.

This is a common result for price-level forecasting, and the main lesson of the project: always test against a naive baseline before trusting a forecasting model.

## Next steps

- Predict daily returns instead of price levels
- Walk-forward evaluation across several periods
- Add features beyond past prices (volume, volatility, market returns)

## How to run

Open `lstm_stock_forecast.ipynb` in Google Colab (TensorFlow and yfinance are preinstalled) and run all cells. The notebook downloads prices from Yahoo Finance, or reads a CSV exported from nasdaq.com if one is saved at `data/TSLA_nasdaq.csv`. Results here come from the Nasdaq export, which matches Yahoo's adjusted closes for TSLA.

## Tools

Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, yfinance, Matplotlib
