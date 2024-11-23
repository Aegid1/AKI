import pandas as pd
import yfinance as yf

def create_test_data_stock_pickle_file():

    data = yf.download(
        "BMW.DE",
        start="2023-11-19",
        end="2024-11-19",
        interval="1h"
    )
    data.index = data.index.tz_convert("Europe/Berlin")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    desired_times = ["09:00:00", "11:00:00", "13:00:00", "15:00:00", "17:00:00"]
    filtered_data = data[data.index.strftime('%H:%M:%S').isin(desired_times)]
    filtered_data['Datetime'] = filtered_data.index

    final_data = filtered_data[['Datetime', 'Open']]
    #final_data = final_data.sort_values(by='Datetime')
    #final_data = final_data.reset_index(drop=True)

    save_path = f"../data/Samples/TestSamples/bmw_test_data.csv"

    final_data.to_csv(save_path, index=False)
    print(f"Gespeichert: {save_path}")


create_test_data_stock_pickle_file()