import os
import pickle
import pandas as pd

def create_stock_pickle_files(experiment: str):
    company_names = ["Airbus", "Allianz", "Deutsche Telekom", "Mercedes-Benz", "Volkswagen", "Porsche", "SAP",
                     "Siemens", "Siemens Healthineers", "Deutsche Bank"]
    input_dir = os.path.join('..', 'data', 'stocks')
    output_dir = os.path.join('..', 'data', 'Samples', f'{experiment}')

    for company in company_names:
        company_path = os.path.join(input_dir, company)
        csv_files = sorted([f for f in os.listdir(company_path) if f.endswith('.csv')])
        all_data = pd.DataFrame()
        for file in csv_files:
            file_path = os.path.join(company_path, file)
            df = pd.read_csv(file_path, parse_dates=['Datetime'])
            all_data = pd.concat([all_data, df])

        all_data.sort_values(by='Datetime', inplace=True)
        all_data.reset_index(drop=True, inplace=True)

        for i in range(0, len(all_data), 25):
            sample = all_data.iloc[i:i + 26]
            x = sample['Open'][:-1].values
            y = sample['Open'].iloc[-1]

            item = {'features': x, 'target': y}
            pickle_filename = f"{company}_{sample['Datetime'].iloc[0].strftime('%Y%m%d%H%M')}.pkl"
            with open(os.path.join(output_dir, pickle_filename), 'wb') as f:
                pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)


create_stock_pickle_files("experiment0")