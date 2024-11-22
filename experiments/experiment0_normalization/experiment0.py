import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler
from torch import optim
from torch.utils.data import random_split, DataLoader
from Datasets.StocksDataSetNoNormalization import StocksDataSet
import matplotlib.pyplot as plt

from experiments.experiment0_normalization.Model import Model

def start_training_normalization():
    start_time = time.time()

    seq_size = 25
    batch_size = 50
    input_size = 1
    hidden_size = 200

    dataset = StocksDataSet(seq_size)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    net = Model(input_size, hidden_size)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_loss_vals = list()
    test_loss_vals = list()

    for epoch in range(100):
        print(f"EPOCH: {epoch}")
        running_training_loss = 0
        running_test_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            #labels = labels.unsqueeze(1) #necessary in order to have the same dimension as inputs
            inputs_reshaped = inputs.squeeze(-1).numpy()  # [50, 25]
            labels_reshaped = labels.squeeze().numpy()  # [50]

            # Combine and normalize
            combined_data = np.hstack((inputs_reshaped, labels_reshaped[:, None]))  # [50, 26]
            scaler = MinMaxScaler()
            combined_data_scaled = scaler.fit_transform(combined_data)

            # Split back into inputs and labels
            inputs_scaled = combined_data_scaled[:, :-1]  # [50, 25]
            labels_scaled = combined_data_scaled[:, -1]  # [50]

            # Convert back to tensors
            inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32).view(inputs.size(0), -1, 1)  # [50, 25, 1]
            labels_tensor = torch.tensor(labels_scaled, dtype=torch.float32).view(labels.size(0), 1, 1)  # [50, 1, 1]
            outputs = net.forward(inputs_tensor)
            loss = criterion(outputs.squeeze(), labels_tensor.squeeze())
            loss.backward()
            optimizer.step()
            running_training_loss += loss.item()

        torch.set_grad_enabled(False)
        for inputs, labels in test_loader:
            #normalize test values
            inputs_reshaped = inputs.squeeze(-1).numpy()  # [50, 25]
            labels_reshaped = labels.squeeze().numpy()  # [50]

            # Combine and normalize
            combined_data = np.hstack((inputs_reshaped, labels_reshaped[:, None]))  # [50, 26]
            scaler = MinMaxScaler()
            combined_data_scaled = scaler.fit_transform(combined_data)

            # Split back into inputs and labels
            inputs_scaled = combined_data_scaled[:, :-1]  # [50, 25]
            labels_scaled = combined_data_scaled[:, -1]  # [50]

            # Convert back to tensors
            inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32).view(inputs.size(0), -1, 1)  # [50, 25, 1]
            labels_tensor = torch.tensor(labels_scaled, dtype=torch.float32).view(labels.size(0), 1, 1)  # [50, 1, 1]
            outputs = net.forward(inputs_tensor)

            loss = criterion(outputs.squeeze(), labels_tensor.squeeze())
            running_test_loss += loss.item()

        print(running_training_loss/len(train_loader))
        print(running_test_loss/len(test_loader))
        train_loss_vals.append(running_training_loss/len(train_loader))
        test_loss_vals.append(running_test_loss/len(test_loader))
        torch.set_grad_enabled(True)

    plt.plot(range(100), train_loss_vals, color='blue', label='Training Loss')
    plt.plot(range(100), test_loss_vals, color='orange', label='Test Loss')

    final_train_loss = f"Final Train Loss: {train_loss_vals[-1]:.4f}"
    final_test_loss = f"Final Test Loss: {test_loss_vals[-1]:.4f}"
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label="Training Loss"),
        Line2D([0], [0], color='orange', lw=2, label="Test Loss"),
        Line2D([0], [0], color='none', label=final_train_loss),
        Line2D([0], [0], color='none', label=final_test_loss)
    ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Das Training hat {duration:.2f} Sekunden gebraucht.")
    torch.save(net.state_dict(), "trained_model_experiment0.pth")

    plt.savefig("training_loss_plot.png", format='png', dpi=300)
    plt.show()

    print(f"FIRST TRAIN LOSS: {train_loss_vals[0]}")
    print(f"FINAL TRAIN LOSS: {train_loss_vals[-1]}\n")

    print(f"FIRST TEST LOSS: {test_loss_vals[0]}")
    print(f"FINAL TEST LOSS: {test_loss_vals[-1]}")

start_training_normalization()