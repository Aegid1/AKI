**Normalization Strategy:**  
used batch normalization, so a new scaler is used for every batch during training

**Learning Rate:** 0.0001

**Batch Size:** 20

**Activation functions:** replaced every tanh and sigmoid activation function with ReLu

-> encountered vanishing gradients problem, due to calculating the derivates of the sigmoid and tanh