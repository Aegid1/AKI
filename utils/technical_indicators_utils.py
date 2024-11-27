import os
import pickle
import pandas as pd

def create_technical_indicators_samples(company_names, training:bool, step_size:int):
    input_dir_stocks = os.path.join('..', 'data', 'technical_indicators')
    if training:
        output_dir = os.path.join('..', 'data', 'Samples', 'experiment2')
    else:
        output_dir = os.path.join('..', 'data', 'Samples', "TestSamples", 'experiment2')
    os.makedirs(output_dir, exist_ok=True)

    for company in company_names:
        file_path = os.path.join(input_dir_stocks, f"{company}.csv")
        all_data = pd.DataFrame()
        df = pd.read_csv(file_path, parse_dates=['Datetime'])
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True).dt.tz_localize(None)
        all_data = pd.concat([all_data, df])

        all_data.sort_values(by='Datetime', inplace=True)
        all_data.reset_index(drop=True, inplace=True)

        for i in range(0, len(all_data) - 25, step_size):
            if i + 25 <= len(all_data):
                sample = all_data.iloc[i:i + 26]
                stock_features = sample['Open'][:-1].values
                rsi = sample['RSI'][:-1].values
                macd = sample['MACD'][:-1].values
                macdsignal = sample['MACDSignal'][:-1].values
                macdhist = sample['MACDHist'][:-1].values
                sma = sample['SMA'][:-1].values
                ema = sample['EMA'][:-1].values
                upperband = sample['UpperBand'][:-1].values
                middleband = sample['MiddleBand'][:-1].values
                lowerband = sample['LowerBand'][:-1].values

                stock_target = sample['Open'].iloc[-1]
                datetime_end = sample['Datetime'].iloc[-1]

                item = {
                    "company_name": company,
                    'features': {
                        'stock_prices': stock_features,
                        'rsi': rsi,
                        'macd': macd,
                        'macdsignal': macdsignal,
                        'macdhist': macdhist,
                        'sma': sma,
                        'ema': ema,
                        "upperband": upperband,
                        "middleband": middleband,
                        "lowerband": lowerband
                    },
                    'target': stock_target,
                    'datetime': datetime_end
                }

                output_filename = os.path.join(output_dir, f'{company}_{datetime_end.strftime("%Y-%m-%d_%H-%M-%S")}.pkl')
                with open(output_filename, 'wb') as f:
                    pickle.dump(item, f)
                print(f"Sample saved: {output_filename}")


#create_technical_indicators_samples(["Airbus", "Allianz", "Deutsche Telekom", "Mercedes-Benz", "Volkswagen", "Porsche", "SAP", "Siemens", "Siemens Healthineers", "Deutsche Bank"], True, 1) #this creates the training samples
#create_technical_indicators_samples(["BMW"], False, 1) #this creates the test samples