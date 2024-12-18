import time

import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from torch import optim
from torch.utils.data import random_split, DataLoader
from Datasets.TechnicalIndicatorsDataSet import TechnicalIndicatorsDataSet
import matplotlib.pyplot as plt

from experiments.experiment2.Model import Model

def start_training(batch_size, hidden_size, learning_rate):
    start_time = time.time()
    stocks_seq_size = 25
    batch_size = batch_size
    input_size = 1
    hidden_size = hidden_size #200 -> 0.28, 400 ->

    dataset = TechnicalIndicatorsDataSet(stocks_seq_size)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = Model(input_size, hidden_size)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    train_loss_vals = list()
    test_loss_vals = list()

    for epoch in range(100):
        print(f"EPOCH: {epoch}")
        running_training_loss = 0
        running_test_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            labels = labels.unsqueeze(1)
            features = [
                inputs["stock_prices"].unsqueeze(-1),
                inputs["rsi"],
                inputs["macd"],
                inputs["macdsignal"],
                inputs["macdhist"],
                inputs["sma"],
                inputs["ema"],
                inputs["upperband"],
                inputs["lowerband"],
                inputs["middleband"]
            ]

            outputs = model.forward(features)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f'{name}: {param.grad.norm()}')

            optimizer.step()
            running_training_loss += loss.item()

        torch.set_grad_enabled(False)
        for inputs, labels in test_loader:
            features = [
                inputs["stock_prices"].unsqueeze(-1),
                inputs["rsi"],
                inputs["macd"],
                inputs["macdsignal"],
                inputs["macdhist"],
                inputs["sma"],
                inputs["ema"],
                inputs["upperband"],
                inputs["lowerband"],
                inputs["middleband"]
            ]
            labels = labels.unsqueeze(1)
            outputs = model.forward(features)
            print("---------------OUTPUTS---------------")
            print(outputs)
            print("---------------LABELS---------------")
            print(labels)

            loss = criterion(outputs, labels)
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
    torch.save(model.state_dict(), "trained_model_experiment3.pth")

    plt.savefig("training_loss_plot.png", format='png', dpi=300)
    plt.show()

    print(f"FIRST TRAIN LOSS: {train_loss_vals[0]}")
    print(f"FINAL TRAIN LOSS: {train_loss_vals[-1]}\n")

    print(f"FIRST TEST LOSS: {test_loss_vals[0]}")
    print(f"FINAL TEST LOSS: {test_loss_vals[-1]}")

#start_training(20, 100, 0.0001) # -> 0.07