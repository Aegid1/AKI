import os
import pickle

import joblib
import pandas as pd
import talib
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from experiments.experiment0_normalization.experiment0 import Model as Model0
from experiments.experiment2.experiment2 import Model as Model2
from experiments.experiment3.experiment3 import Model as Model3

class ModelService:

    def test_model_prediction_normalization_experiment0(self, experiment:str, model_path:str):
        """
            Tests a model's prediction accuracy for Experiment 0 and visualizes the results.

            Parameters:
            - experiment (str): Name of the experiment folder containing the model.
            - model_path (str): Path to the trained model file within the experiment folder.

            Workflow:
            1. Loads the specified model and sets it to evaluation mode.
            2. Reads test data from a predefined CSV file.
            3. Processes the test data using a rolling window of the last 25 values to generate inputs for the model.
            4. Normalizes input data using MinMaxScaler for each prediction step.
            5. Uses the model to predict the next value, scales it back to the original range, and compares it to the actual value.
            6. Stores predictions, actual values, and their differences in a DataFrame.
            7. Calculates the average difference between predictions and actual values.
            8. Generates and saves a plot comparing predictions to actual values over time, annotated with the average difference.

            Data Format:
            - Input CSV file must have columns:
              - 'Datetime': Timestamp of the data points.
              - 'Open': Values used for predictions.
            - Output CSV includes:
              - 'Date': Dates of predictions.
              - 'Prediction': Predicted values by the model.
              - 'Actual': Actual observed values.
              - 'Difference': Absolute difference between prediction and actual values.

            Outputs:
            - A PNG file with a line plot comparing predictions to actual values.

            Notes:
            - The MinMaxScaler is re-fitted for each prediction window, ensuring normalization respects the local context of the input values.
            - Padding is applied to match scaler input shapes during denormalization.
            - The function is tailored for Experiment 0 and assumes a specific model structure and data format.

            Visualization:
            - X-axis: Dates of predictions.
            - Y-axis: Predicted and actual values.
            - Legend includes the average difference for easy evaluation.
            """
        model_path = os.path.join("experiments", f"{experiment}",f"{model_path}")
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

        predictions_df['Difference'] = abs(predictions_df['Prediction'] - predictions_df['Actual'])
        average_difference = predictions_df['Difference'].mean()
        print(average_difference)
        plt.figure(figsize=(10, 6))
        plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Prediction', color='blue')
        plt.plot(predictions_df['Date'], predictions_df['Actual'], label='Actual', color='orange')
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Model Predictions vs Actual Values")
        plt.legend(title=f"Average Difference: {average_difference:.4f}")
        save_path = f"./experiments/{experiment}/prediction_plot_{experiment}.png"
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()


    def test_model_prediction_no_normalization_experiment0(self, experiment:str, model_path:str):
        """
            Tests a model's prediction accuracy for Experiment 0 without any data normalization.

            Parameters:
            - experiment (str): Name of the experiment folder containing the model.
            - model_path (str): Path to the trained model file within the experiment folder.

            Workflow:
            1. Loads the specified model and sets it to evaluation mode.
            2. Reads test data from a predefined CSV file (using 'Open' column for prediction).
            3. Processes the test data using a rolling window of the last 25 values to generate inputs for the model.
            4. Uses the model to predict the next value without normalizing the inputs or outputs.
            5. Compares the predicted value with the actual value from the dataset.
            6. Stores predictions, actual values, and their differences in a DataFrame.
            7. Calculates the average difference between predictions and actual values.
            8. Generates and saves a plot comparing predictions to actual values over time, annotated with the average difference.

            Data Format:
            - Input CSV file must have columns:
              - 'Datetime': Timestamp of the data points.
              - 'Open': Values used for predictions.
            - Output CSV includes:
              - 'Date': Dates of predictions.
              - 'Prediction': Predicted values by the model.
              - 'Actual': Actual observed values.
              - 'Difference': Absolute difference between prediction and actual values.

            Outputs:
            - A PNG file with a line plot comparing predictions to actual values.

            Notes:
            - This function tests the model with raw (non-normalized) data.
            - Unlike the previous experiment that normalized inputs using a scaler, this function uses the raw values directly.
            - Padding is not necessary since the inputs are not normalized.
            - The `inputs_tensor` is directly passed to the model for predictions.
            """
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

        predictions_df['Difference'] = abs(predictions_df['Prediction'] - predictions_df['Actual'])
        average_difference = predictions_df['Difference'].mean()
        print(average_difference)
        plt.figure(figsize=(10, 6))
        plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Prediction', color='blue')
        plt.plot(predictions_df['Date'], predictions_df['Actual'], label='Actual', color='orange')
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Model Predictions vs Actual Values")
        plt.legend(title=f"Average Difference: {average_difference:.4f}")
        save_path = f"./experiments/{experiment}/prediction_plot_{experiment}.png"
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()


    def test_model_prediction_normalization_experiment3_new_scaler(self, experiment:str, model_path:str):
        """
            Tests a model's prediction accuracy for Experiment 3 using normalization with new scalers for each feature.

            Parameters:
            - experiment (str): Name of the experiment folder containing the model.
            - model_path (str): Path to the trained model file within the experiment folder.

            Workflow:
            1. Loads the specified model (`Model3`) and sets it to evaluation mode.
            2. Iterates over each pickle file in the test data directory for the experiment.
            3. Loads the data from each pickle file, including various economic indicators and stock prices.
            4. Normalizes the features (such as GDP, inflation, oil prices, etc.) using scalers stored during training.
            5. Creates a tensor for each feature and normalizes it based on the scaler from the training phase.
            6. The model then predicts stock prices based on these normalized features.
            7. Denormalizes the predicted stock prices back to the original scale using the `MinMaxScaler`.
            8. Stores the predictions, actual values, and their differences in a DataFrame.
            9. Calculates the average difference between predictions and actual values.
            10. Generates and saves a plot comparing predictions to actual values over time, annotated with the average difference.

            Data Format:
            - Input pickle files should contain the following:
              - 'features': A dictionary with keys representing economic indicators and stock prices:
                - 'gdp', 'unemployment_rate', 'inflation', 'interest_rates', 'oil_prices', 'currency_rates': Normalized features.
                - 'stock_prices': Stock prices, which are normalized using `MinMaxScaler`.
              - 'target': Actual stock price to be predicted.
              - 'datetime': Timestamps corresponding to each data entry.

            Notes:
            - The model uses individual scalers for each economic indicator, loaded from a training directory.
            - A `MinMaxScaler` is used specifically for stock prices to ensure proper normalization.
            - The scaler is applied to each feature independently before passing them into the model for predictions.
            - The prediction is padded with zeros to match the shape used during training before denormalization.
            """
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

        predictions_df['Difference'] = abs(predictions_df['Prediction'] - predictions_df['Actual'])
        average_difference = predictions_df['Difference'].mean()
        print(average_difference)
        plt.figure(figsize=(10, 6))
        plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Prediction', color='blue')
        plt.plot(predictions_df['Date'], predictions_df['Actual'], label='Actual', color='orange')
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Model Predictions vs Actual Values")
        plt.legend(title=f"Average Difference: {average_difference:.4f}")
        save_path = f"./experiments/{experiment}/prediction_plot_{experiment}.png"
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()


    def test_model_prediction_experiment2_batch_normalization(self, model_path):
        """
        Tests the model with batch normalization for Experiment 2 by calculating predictions based on
        normalized input values and comparing the results to the actual values.

        This function loads a pre-trained model (Model2) that uses batch normalization and uses it to
        generate predictions for test data. The input features are normalized before being passed to the model.
        The predictions are then denormalized, and the difference between the predictions and the actual values
        is computed. A plot comparing the predictions and actual values is generated and saved.

        Parameters:
        - model_path (str): The path to the file containing the pre-trained model weights, relative to the experiment directory.

        Workflow:
            1. Loads the pre-trained model (`Model2`) from the given `model_path`.
            2. Puts the model in evaluation mode (`model.eval()`) to disable batch normalization and dropout.
            3. Loads the pickle files containing the test data for Experiment 2 from a predefined directory.
            4. Normalizes the input values (including stock prices and other features) using the `MinMaxScaler`.
            5. Passes the normalized features to the model to compute the prediction.
            6. Denormalizes the prediction to bring the value back to its original scale.
            7. Calculates the difference between the predicted and actual values.
            8. Generates and saves a plot comparing the predictions with the actual values.
            9. Computes and outputs the average difference between the predictions and the actual values.

            Return Values:
            - This function does not return any values. It generates a plot and saves it in the specified directory.

            Data Format:
            - The pickle files should have the following structure:
            - `features`: A dictionary containing the features, including "stock_prices" and other relevant economic indicators (e.g., unemployment rate, oil prices, etc.).
            - `target`: The actual value that the model is supposed to predict (e.g., the stock price).
            - `datetime`: The date of the actual observation (used for the X-axis of the plot).
        """
        model_path = os.path.join("experiments", "experiment2", f"{model_path}")

        input_size = 1
        hidden_size = 100
        model = Model2(input_size=input_size, hidden_size=hidden_size)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        pickle_directory = os.path.join("data", "Samples", "TestSamples", "experiment2")
        pkl_files = [f for f in os.listdir(pickle_directory) if f.endswith('.pkl')]

        predictions_df = pd.DataFrame(columns=["Date", "Prediction", "Actual"])
        for file in pkl_files:
            file_path = os.path.join(pickle_directory, file)
            with open(file_path, 'rb') as f:
                item = pickle.load(f)
                features = []

                stock_prices = item["features"]["stock_prices"]
                stock_scaler = MinMaxScaler()
                stock_tensor = torch.tensor(stock_prices, dtype=torch.float32)
                stock_prices_scaled = stock_scaler.fit_transform(stock_tensor.reshape(-1,1))
                features.append(torch.tensor(stock_prices_scaled, dtype=torch.float32).unsqueeze(-1))
                for key, value in item["features"].items():
                    if key != "stock_prices":
                        scaler = MinMaxScaler()
                        input_tensor = torch.tensor(item["features"][key], dtype=torch.float32)
                        feature_scaled = scaler.fit_transform(input_tensor.reshape(-1,1))
                        features.append(torch.tensor(feature_scaled, dtype=torch.float32).unsqueeze(-1))

                prediction_normalized = model.forward(features)
                prediction_padded = np.pad(prediction_normalized.detach().numpy(), ((0, 0), (0, 24)), mode='constant',
                                           constant_values=0)
                prediction = stock_scaler.inverse_transform(prediction_padded)
                denormalized_prediction = prediction[0, 0]

            actual_value = item['target']
            prediction_date = item['datetime']
            print(actual_value)
            print(denormalized_prediction)
            new_row = pd.DataFrame({
                "Date": [prediction_date],
                "Prediction": [denormalized_prediction],
                "Actual": [actual_value]
            })
            predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)

        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        predictions_df = predictions_df.sort_values(by="Date").reset_index(drop=True)
        predictions_df['Difference'] = abs(predictions_df['Prediction'] - predictions_df['Actual'])
        average_difference = predictions_df['Difference'].mean()
        print(average_difference)
        plt.figure(figsize=(10, 6))
        plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Prediction', color='blue')
        plt.plot(predictions_df['Date'], predictions_df['Actual'], label='Actual', color='orange')
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Model Predictions vs Actual Values")
        plt.legend(title=f"Average Difference: {average_difference:.4f}")
        save_path = os.path.join("experiments", "experiment2", "prediction_plot_experiment2.png")
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()


    def test_moving_average_model(self):
        """
        Tests the moving average model by generating predictions based on a 25-period simple moving average
        and comparing them to the actual values.

        Workflow:
        1. Loads the test data from a CSV file (`bmw_test_data.csv`) containing stock prices and dates.
        2. Adds a new column to the data frame that contains the 25-period simple moving average (SMA) of the "Open" prices.
        3. For each day in the data (starting from the 26th row), the function:
            - Uses the previous day's SMA as the predicted value.
            - Retrieves the actual stock price for the following day.
            - Records the prediction, actual value, and date.
        4. Computes the absolute difference between the predicted and actual values.
        5. Generates and saves a plot comparing the predicted values with the actual values.
        6. Computes and prints the average difference between predictions and actual values.
        """
        data_path = os.path.join("data", "Samples", "TestSamples", "experiment0", "bmw_test_data.csv")
        data = pd.read_csv(data_path)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data['MA_25'] = talib.SMA(data['Open'], timeperiod=25)

        predictions_df = pd.DataFrame(columns=["Date", "Prediction", "Actual"])
        for i in range(25, len(data) - 1):
            actual_value = data['Open'].iloc[i + 1]
            prediction_date = data['Datetime'].iloc[i + 1]
            prediction = data['MA_25'].iloc[i]

            new_row = pd.DataFrame({
                "Date": [prediction_date],
                "Prediction": [prediction],
                "Actual": [actual_value]
            })
            predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)

        predictions_df['Difference'] = abs(predictions_df['Prediction'] - predictions_df['Actual'])
        average_difference = predictions_df['Difference'].mean()
        print(average_difference)
        plt.figure(figsize=(10, 6))
        plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Prediction', color='blue')
        plt.plot(predictions_df['Date'], predictions_df['Actual'], label='Actual', color='orange')
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Model Predictions vs Actual Values")
        plt.legend(title=f"Average Difference: {average_difference:.4f}")
        save_path = f"./experiments/moving_average_prediction.png"
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()