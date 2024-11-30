### Experiment 2

**Short description:**  
Predicting stock price increase two hours ahead by using a MI-LSTM model on technical indicators.

**Data acquisition:**  
- Daily stock prices of 10 of the biggest DAX stocks in terms of market capitalization from 11/2022 to 11/2024  
- Technical indicators: SMA, EMA, RSI, MACD, MACDSignal, MACDHist, Bollinger Bands for each stock  

**Features:**  
- Stock prices from the last 20 timesteps  
- Technical indicators from the last 20 timesteps (technical indicators themselves are calculated with different time ranges)

**Target:** Open stock price in the next 2 hours  

**Modeling architecture:**  
Multi-input LSTM followed by DNN layer for prediction.

**Performance criteria:**
- MSE of true vs. predicted stock price increase for test data  

**Baseline:**  
- MSE of true vs. average price increase over all stocks for test data
