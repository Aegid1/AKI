import time

import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from torch import nn, optim
from torch.utils.data import random_split, DataLoader

from Datasets.StocksDataSetNoNormalization import StocksDataSet
from experiments.experiment0_normalization.Model import Model


def start_training_no_normalization():
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
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9) #learning rate is adjusted from 0.001 to 0.0001 - otherwise gradient explosion
    train_loss_vals = list()
    test_loss_vals = list()

    for epoch in range(100):
        print(f"EPOCH: {epoch}")
        running_training_loss = 0
        running_test_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels.squeeze(-1))
            loss.backward()
            optimizer.step()
            running_training_loss += loss.item()

        torch.set_grad_enabled(False)
        for inputs, labels in test_loader:

            outputs = net.forward(inputs)
            loss = criterion(outputs, labels.squeeze(-1))
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

#start_training_no_normalization