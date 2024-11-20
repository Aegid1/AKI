import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from experiments.experiment0.Experiment0 import Model

class ModelService:

    def test_model_prediction(self, experiment:str, model_path:str):
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
        initial_input = data['Open'].iloc[:25].values

        actuals = []
        predictions = []
        scaler = MinMaxScaler()

        for i in range(25, len(data)):
            print(i)
            if i + 1 >= len(data):
                break

            # use last 25 values as input
            inputs = data['Open'].iloc[i - 25:i].values
            inputs_scaled = scaler.fit_transform(inputs)
            inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32).unsqueeze(0)  # Formatiere für das Modell

            with torch.no_grad():
                prediction_scaled = model(inputs_tensor).item()

            #denormalize the values of the prediction
            prediction = scaler.inverse_transform(prediction_scaled.detach().numpy().reshape(-1, 1))

            #get actual value of the test data
            actual_value = data['Open'].iloc[i + 1]
            predictions.append(prediction)
            actuals.append(actual_value)

            #save the date of the predicted value
            prediction_date = data['Datetime'].iloc[i + 1]

            predictions_df = predictions_df.append({
                "Date": prediction_date,
                "Prediction": prediction,
                "Actual": actual_value
            }, ignore_index=True)

        plt.figure(figsize=(10, 6))
        plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Prediction', color='blue')
        plt.plot(predictions_df['Date'], predictions_df['Actual'], label='Actual', color='orange')
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Model Predictions vs Actual Values")
        plt.legend()
        plt.grid()
        plt.show()

        save_path = f"./experiments/{experiment}/prediction_plot_{model_path}.png"
        plt.savefig(save_path, format='png', dpi=300)
        plt.show()

        return predictions_df