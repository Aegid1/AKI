### Experiment 1

**Short description:**  
Using multi-input LSTM with DNN-Layers for predicting the price increase of a stock in the next 2 hours ahead by using the short-term and long-term impact of news articles, defined by GPT-3.5.

**Data acquisition:**  
- Daily stock prices of 10 of the biggest DAX stocks in terms of market capitalization from 11/2022 to 11/2024 according to finanzen.net
- news articles for each given stock
- general economy related news articles

**Features:**  
- Stock prices from the last 20 timesteps  
- mean of the short-term impact (10 positive impact, 0 neutral impact, -10 negative impact) from last 20 time steps
- mean of the long-term impact from last 60 time steps

**Target:** Open stock price in the next 2 hours

**Modeling architecture:**  
Multi-input LSTM followed by DNN layer for prediction.

**Performance criteria:**
- MSE of true vs. predicted stock price increase for test data  

**Baseline:**  
- MSE of true vs. average price increase over all stocks for test data

-> only the code for the data retrieval could be finished, due to an openai bug, where batches cant be further finished because of a token limit of batches that are in process. 
However even when no batches are processed, this token limit sometimes still appears.
