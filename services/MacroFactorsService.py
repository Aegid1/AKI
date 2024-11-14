from datetime import datetime
import pandas as pd
import yaml
from alpha_vantage.foreignexchange import ForeignExchange
from ecbdata import ecbdata
from alpha_vantage.timeseries import TimeSeries

class MacroFactorsService:
    config = yaml.safe_load(open("openai_config.yaml"))
    api_key = config['KEYS']['alpha_vantage']

    def save_central_interest_rate(self, start_period: str, end_period: str):
        # start_period = 2022-11 and end_period
        df = ecbdata.get_series('FM.B.U2.EUR.4F.KR.MRR_FR.LEV', start=start_period, end=end_period)
        df = df[["TIME_PERIOD", "OBS_VALUE"]]
        df = df.rename(columns={"TIME_PERIOD": "Date", "OBS_VALUE": "Value"})
        self.save_as_csv("central_interest_rate", df)


    def save_inflation_rate(self, start_period: str, end_period: str):
        df = ecbdata.get_series('ICP.M.DE.N.000000.4.ANR', start=start_period, end=end_period)
        df = df[["TIME_PERIOD", "OBS_VALUE"]]
        df = df.rename(columns={"TIME_PERIOD": "Date", "OBS_VALUE": "Value"})
        self.save_as_csv("inflation_rate", df)


    def save_unemployment_rate(self, start_period: str, end_period: str):
        df = ecbdata.get_series('LFSI.M.DE.S.UNEHRT.TOTAL0.15_74.T', start=start_period, end=end_period)
        df = df[["TIME_PERIOD", "OBS_VALUE"]]
        df = df.rename(columns={"TIME_PERIOD": "Date", "OBS_VALUE": "Value"})
        self.save_as_csv("unemployment_rate", df)


    def save_gdp(self, start_period: str, end_period: str):
        df = ecbdata.get_series('MNA.Q.N.DE.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N', start=start_period, end=end_period)
        df = df[["TIME_PERIOD", "OBS_VALUE"]]

        quarter_to_month = {
            "Q1": "01",
            "Q2": "04",
            "Q3": "07",
            "Q4": "10"
        }
        df['TIME_PERIOD'] = df['TIME_PERIOD'].apply(lambda x: f"{x[:4]}-{quarter_to_month[x[5:]]}-01")
        df = df.rename(columns={"TIME_PERIOD": "Date", "OBS_VALUE": "Value"})
        self.save_as_csv("gdp", df)


    def save_currency_euro_dollar(self, start_period: str, end_period: str):
        fx = ForeignExchange(key=self.api_key, output_format='json')
        data, meta_data = fx.get_currency_exchange_daily(from_symbol='EUR', to_symbol='USD', outputsize='full')

        start_date = datetime.strptime(start_period, '%Y-%m-%d')
        end_date = datetime.strptime(end_period, '%Y-%m-%d')

        filtered_data = []
        for date_str, values in data.items():
            date = datetime.strptime(date_str, '%Y-%m-%d')

            if start_date <= date <= end_date:
                open_value = values['1. open']
                filtered_data.append([date_str, open_value])

        df = pd.DataFrame(filtered_data, columns=["Date", "Open"])
        df = df.rename(columns={"Open": "Value"})
        df = df.sort_values(by="Date", ascending=True)

        self.save_as_csv("currency_euro_dollar", df)


    #only takes the WTI oil price and not the BRENT
    def save_oil_prices(self, start_period: str, end_period: str):
        ts = TimeSeries(key=self.api_key, output_format='json')
        data, meta_data = ts.get_daily(symbol="WTI", outputsize='full')

        start_date = datetime.strptime(start_period, '%Y-%m-%d')
        end_date = datetime.strptime(end_period, '%Y-%m-%d')

        filtered_data = []
        for date_str, values in data.items():
            date = datetime.strptime(date_str, '%Y-%m-%d')

            if start_date <= date <= end_date:
                open_value = values['1. open']
                filtered_data.append([date_str, open_value])

        df = pd.DataFrame(filtered_data, columns=["Date", "Open"])
        df = df.rename(columns={"Open": "Value"})
        df = df.sort_values(by="Date", ascending=True)

        self.save_as_csv("oil", df)


    def save_as_csv(self, filename: str, dataframe):
        filepath = f"data/macro_factors/{filename}.csv"
        dataframe.to_csv(filepath, index=False)

        print(f"DataFrame wurde als {filepath} gespeichert.")