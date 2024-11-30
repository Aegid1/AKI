**Normalization Strategy:**  
used a different scaler for every company for every feature (stock_prices, ema, sma, etc.)

**Learning Rate:** 0.0001

**Batch Size:** 20

**Activation functions:** replaced every tanh and sigmoid activation function with ReLu

-> encountered vanishing gradients problem, due to calculating the derivates of the sigmoid and tanh
