### Experiment 3

**Short description:**  
Multi-input LSTM for predicting the weekly stock price change by integrating macroeconomic factors.

**Data acquisition:**  
- Daily stock prices of 10 of the biggest DAX stocks in terms of market capitalization from 11/2022 to 11/2024  
- Macroeconomic indicators: ECB central interest rate, inflation, GDP, unemployment rate, oil price, gold price, EUR/USD currency rate

**Features:**  
- Stock prices from the last 20 timesteps  
- Oil and gold price from the last 5 timesteps
- other factors are provided as single values

**Target:** Open stock price in the next 2 hours  

**Modeling architecture:**  
Multi-input LSTM with an attention layer and multiple DNNs

**Performance criteria:**
- MSE of true vs. predicted stock price increase for test data  

**Baseline:**  
- MSE of true vs. average price increase over all stocks for test data
