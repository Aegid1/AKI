import os
import pickle

import joblib
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from experiments.experiment0_normalization.experiment0 import Model as Model0
from experiments.experiment2.experiment2 import Model as Model2
from experiments.experiment3.experiment3 import Model as Model3
from utils.macro_factors_utils import get_last_n_valid_values

class ModelService:

    def test_model_prediction_normalization_experiment0(self, experiment:str, model_path:str):
        model_path = f"./experiments/{experiment}/{model_path}"
        input_size = 1
        hidden_size = 200
        model = Model0(input_size=input_size, hidden_size=hidden_size)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()

        data_path = "./data/Samples/TestSamples/experiment0/bmw_test_data.csv"
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
        model = Model0(input_size=input_size, hidden_size=hidden_size)

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


    def test_model_prediction_normalization_experiment3_new_scaler(self, experiment:str, model_path:str):
        model_path = f"./experiments/{experiment}/{model_path}"
        input_size = 1
        hidden_size = 200
        model = Model3(input_size=input_size, hidden_size=hidden_size)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        pickle_directory = os.path.join("data", "Samples", "TestSamples", "experiment3")
        pkl_files = [f for f in os.listdir(pickle_directory) if f.endswith('.pkl')]

        predictions_df = pd.DataFrame(columns=["Date", "Prediction", "Actual"])
        for file in pkl_files:
            file_path = os.path.join(pickle_directory, file)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            features_to_normalize = ["gdp", "unemployment_rate", "inflation", "interest_rates", "oil_prices", "currency_rates"]
            features = {key: [] for key in features_to_normalize}

            for key in features_to_normalize:
                inputs = data["features"][key]
                #load the scaler from the training for better performance
                scaler = joblib.load(os.path.join("experiments", "experiment3", "scalers", f"{key}_scaler.pkl"))
                scaled_inputs = []
                if inputs.squeeze().size != 1:
                    for value in inputs.squeeze():
                        scaled_value = scaler.transform(value.reshape(1, -1))
                        scaled_inputs.append(scaled_value.flatten())

                    inputs_tensor = torch.tensor(scaled_inputs, dtype=torch.float32).unsqueeze(0)
                else:
                    scaled_value = scaler.transform(inputs.reshape(1, -1))
                    inputs_tensor = torch.tensor(scaled_value, dtype=torch.float32).squeeze(1)

                features[key].append(inputs_tensor)

            stock_prices = data["features"]["stock_prices"]
            stock_scaler = MinMaxScaler()
            scaled_stock_prices = stock_scaler.fit_transform(stock_prices.reshape(1, -1))
            stocks_tensor = torch.tensor(scaled_stock_prices, dtype=torch.float32).unsqueeze(-1)

            features["stock_prices"] = scaled_stock_prices
            with torch.no_grad():
                prediction_scaled = model.forward(
                    stock_price=stocks_tensor,
                    oil_price=features["oil_prices"][0],
                    currency_rate=features["currency_rates"][0],
                    interest_rate=features["interest_rates"][0],
                    gdp=features["gdp"][0],
                    unemployment_rate=features["unemployment_rate"][0],
                    inflation_rate=features["inflation"][0]
                )
            prediction_scaled = prediction_scaled.unsqueeze(1)
            #padding is needed as the scaler can only denormalize values in the same shape it normalized in the start
            prediction_padded = np.pad(prediction_scaled.detach().numpy(), ((0, 0), (0, 24)), mode='constant',
                                       constant_values=0)
            prediction = stock_scaler.inverse_transform(prediction_padded)
            denormalized_prediction = prediction[0, 0]

            actual_value = data['target']
            prediction_date = data['datetime']
            new_row = pd.DataFrame({
                "Date": [prediction_date],
                "Prediction": [denormalized_prediction],
                "Actual": [actual_value]
            })
            predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)
        print(predictions_df)
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

    # def test_model_prediction_normalization_experiment3(self, experiment: str, model_path: str):
    #     model_path = f"./experiments/{experiment}/{model_path}"
    #     input_size = 1
    #     hidden_size = 200
    #     model = Model3(input_size=input_size, hidden_size=hidden_size)
    #
    #     state_dict = torch.load(model_path)
    #     model.load_state_dict(state_dict)
    #     model.eval()
    #     pickle_directory = os.path.join("data", "Samples", "TestSamples", "experiment3")
    #     pkl_files = [f for f in os.listdir(pickle_directory) if f.endswith('.pkl')]
    #
    #     predictions_df = pd.DataFrame(columns=["Date", "Prediction", "Actual"])
    #     for file in pkl_files:
    #         file_path = os.path.join(pickle_directory, file)
    #         with open(file_path, 'rb') as f:
    #             data = pickle.load(f)
    #
    #         features_to_normalize = ["gdp", "unemployment_rate", "inflation", "interest_rates", "stock_prices",
    #                                  "oil_prices", "currency_rates"]
    #         features = {key: [] for key in features_to_normalize}
    #
    #         for key in features_to_normalize:
    #             inputs = data["features"][key]
    #             # load the scaler from the training for better performance
    #             scaler = joblib.load(os.path.join("experiments", "experiment3", "scalers", f"{key}_scaler.pkl"))
    #             scaled_inputs = []
    #             if inputs.squeeze().size != 1:
    #                 for value in inputs.squeeze():
    #                     scaled_value = scaler.transform(value.reshape(1, -1))
    #                     scaled_inputs.append(scaled_value.flatten())
    #
    #                 inputs_tensor = torch.tensor(scaled_inputs, dtype=torch.float32).unsqueeze(0)
    #             else:
    #                 scaled_value = scaler.transform(inputs.reshape(1, -1))
    #                 inputs_tensor = torch.tensor(scaled_value, dtype=torch.float32).squeeze(1)
    #
    #             features[key].append(inputs_tensor)
    #
    #         with torch.no_grad():
    #             prediction_scaled = model.forward(
    #                 stock_price=features["stock_prices"][0],
    #                 oil_price=features["oil_prices"][0],
    #                 currency_rate=features["currency_rates"][0],
    #                 interest_rate=features["interest_rates"][0],
    #                 gdp=features["gdp"][0],
    #                 unemployment_rate=features["unemployment_rate"][0],
    #                 inflation_rate=features["inflation"][0]
    #             )
    #         prediction_scaled = prediction_scaled.unsqueeze(1)
    #         stock_prices_scaler = joblib.load(
    #             os.path.join("experiments", "experiment3", "scalers", f"stock_prices_scaler.pkl"))
    #         # padding is needed as the scaler can only denormalize values in the same shape it normalized in the start
    #         prediction_padded = np.pad(prediction_scaled.detach().numpy(), ((0, 0), (0, 24)), mode='constant',
    #                                    constant_values=0)
    #         prediction = stock_prices_scaler.inverse_transform(prediction_padded)
    #         denormalized_prediction = prediction[0, 0]
    #
    #         actual_value = data['target']
    #         prediction_date = data['datetime']
    #         new_row = pd.DataFrame({
    #             "Date": [prediction_date],
    #             "Prediction": [denormalized_prediction],
    #             "Actual": [actual_value]
    #         })
    #         predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)
    #     print(predictions_df)
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Prediction', color='blue')
    #     plt.plot(predictions_df['Date'], predictions_df['Actual'], label='Actual', color='orange')
    #     plt.xlabel("Date")
    #     plt.ylabel("Value")
    #     plt.title("Model Predictions vs Actual Values")
    #     plt.legend()
    #     save_path = f"./experiments/{experiment}/prediction_plot_{experiment}.png"
    #     plt.savefig(save_path, format='png', dpi=300)
    #     plt.show()