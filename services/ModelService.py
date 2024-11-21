import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from experiments.experiment0.experiment0 import Model

class ModelService:

    def test_model_prediction_normalization(self, experiment:str, model_path:str):
        model_path = f"./experiments/{experiment}/{model_path}"
        input_size = 1
        hidden_size = 200
        model = Model(input_size=input_size, hidden_size=hidden_size)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()

        data_path = "./data/Samples/TestSamples/bmw_test_data.csv"
        data = pd.read_csv(data_path)
        data['Datetime'] = pd.to_datetime(data['Datetime'])

        predictions_df = pd.DataFrame(columns=["Date", "Prediction", "Actual"])
        for i in range(25, len(data)):
            print(i)
            if i + 1 >= len(data):
                break
            scaler = MinMaxScaler()
            # use last 25 values as input
            inputs = data['Open'].iloc[i - 25:i].values
            inputs_scaled = scaler.fit_transform(inputs.reshape(1, -1))
            inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32).unsqueeze(2)
            print(inputs_tensor.shape)
            with torch.no_grad():
                prediction_scaled = model(inputs_tensor)

            #padding is needed as the scaler can only denormalize values in the same shape it normalized in the start
            prediction_padded = np.pad(prediction_scaled.detach().numpy(), ((0, 0), (0, 24)), mode='constant',
                                       constant_values=0)
            prediction = scaler.inverse_transform(prediction_padded)
            denormalized_prediction = prediction[0, 0]

            actual_value = data['Open'].iloc[i + 1]

            #save the date of the predicted value
            prediction_date = data['Datetime'].iloc[i + 1]
            new_row = pd.DataFrame({
                "Date": [prediction_date],
                "Prediction": [denormalized_prediction],
                "Actual": [actual_value]
            })
            predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)
        plt.figure(figsize=(10, 6))
        plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Prediction', color='blue')
        plt.plot(predictions_df['Date'], predictions_df['Actual'], label='Actual', color='orange')
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Model Predictions vs Actual Values")
        plt.legend()
        save_path = f"./experiments/{experiment}/prediction_plot_{experiment}.png"
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()


    def test_model_prediction_no_normalization(self, experiment:str, model_path:str):
        model_path = f"./experiments/{experiment}/{model_path}"
        input_size = 1
        hidden_size = 200
        model = Model(input_size=input_size, hidden_size=hidden_size)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()

        data_path = "./data/Samples/TestSamples/bmw_test_data.csv"
        data = pd.read_csv(data_path)
        data['Datetime'] = pd.to_datetime(data['Datetime'])

        predictions_df = pd.DataFrame(columns=["Date", "Prediction", "Actual"])
        for i in range(25, len(data)):
            if i + 1 >= len(data):
                break
            # use last 25 values as input
            inputs = data['Open'].iloc[i - 25:i].values
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
            inputs_tensor = inputs_tensor.unsqueeze(2)

            with torch.no_grad():
                prediction = model(inputs_tensor)
                print(prediction)

            actual_value = data['Open'].iloc[i + 1]

            #save the date of the predicted value
            prediction_date = data['Datetime'].iloc[i + 1]
            new_row = pd.DataFrame({
                "Date": [prediction_date],
                "Prediction": [prediction],
                "Actual": [actual_value]
            })
            predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)
        plt.figure(figsize=(10, 6))
        plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Prediction', color='blue')
        plt.plot(predictions_df['Date'], predictions_df['Actual'], label='Actual', color='orange')
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Model Predictions vs Actual Values")
        plt.legend()
        save_path = f"./experiments/{experiment}/prediction_plot_{experiment}.png"
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()