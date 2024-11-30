### Experiment 0

**Short description:**  
Using multi-input LSTM with DNN-Layers for predicting the price increase of a stock in the next 2 hours ahead.

**Data acquisition:**  
- Daily stock prices of 10 of the biggest DAX stocks in terms of market capitalization from 11/2022 to 11/2024 according to finanzen.net

**Features:**  
- Stock prices from the last 20 timesteps

**Target:** Open stock price in the next 2 hours

**Modeling architecture:**  
- Multi-input LSTM followed by DNN layer for prediction.

**Performance criteria:**
- MSE of true vs. predicted stock price increase for test data  

**Baseline:**  
- MSE of true vs. average price increase over all stocks for test data
