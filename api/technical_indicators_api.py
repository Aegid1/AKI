import os

import pandas as pd
from fastapi import APIRouter, Depends
from services.TechnicalIndicatorsService import TechnicalindicatorsService

router = APIRouter()

@router.get("/technical/indicators")
def save_technical_indicators(technical_indicator_service: TechnicalindicatorsService = Depends()):
    """
        Calculate and save technical indicators for multiple companies.

        This endpoint processes stock data for a predefined list of companies,
        calculates various technical indicators, and saves the enriched data as
        CSV files. The technical indicators include RSI, Bollinger Bands, SMA,
        EMA, and MACD.

        Parameters:
        - technical_indicator_service (TechnicalindicatorsService): A service that
          provides methods for reading stock data and calculating technical indicators.

        Technical Indicators Calculated:
        1. **RSI (Relative Strength Index):**
           - Calculated using a 14-day period.
        2. **Bollinger Bands:**
           - Calculated using a 20-day period and provides `UpperBand`, `MiddleBand`,
             and `LowerBand`.
        3. **SMA (Simple Moving Average):**
           - Calculated using a 20-day period.
        4. **EMA (Exponential Moving Average):**
           - Calculated using a 20-day period.
        5. **MACD (Moving Average Convergence Divergence):**
           - Calculated using periods of 12, 26, and 9 days, resulting in `MACD`,
             `MACDSignal`, and `MACDHist`.

        Processing Steps:
        1. For each company in the predefined list:
           - Read stock data from the corresponding file directory.
           - Calculate the technical indicators and add them as new columns to the
             stock data DataFrame.
        2. Handle missing values by replacing them with 0.
        3. Sort the DataFrame by the `Datetime` column in ascending order.
        4. Save the processed DataFrame to a CSV file in the `data/technical_indicators/` directory.
        """

    company_names = ["Airbus", "Allianz", "Deutsche Telekom", "Mercedes-Benz", "Volkswagen", "Porsche", "SAP",
                     "Siemens", "Siemens Healthineers", "Deutsche Bank", "BMW"]
    for company in company_names:
        file_path = os.path.join("data", "stocks", f"{company}")
        output_dir = os.path.join("data", "technical_indicators", f"{company}.csv")

        df = technical_indicator_service.get_stock_data_as_df(file_path)
        df['RSI'] = technical_indicator_service.calculate_rsi(company, 14)

        upperband, middleband, lowerband = technical_indicator_service.calculate_bollinger_baender(company, 20)
        df['UpperBand'] = upperband
        df['MiddleBand'] = middleband
        df['LowerBand'] = lowerband

        df['SMA'] = technical_indicator_service.calculate_simple_moving_average(company, 20)
        df['EMA'] = technical_indicator_service.calculate_exponential_moving_average(company, 20)

        macd, macdsignal, macdhist = technical_indicator_service.calculate_moving_average_convergence_divergence(company, 12,26, 9)
        df['MACD'] = macd
        df['MACDSignal'] = macdsignal
        df['MACDHist'] = macdhist

        df.fillna(0, inplace=True)

        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.sort_values(by='Datetime', ascending=True, inplace=True)

        df.to_csv(output_dir, index=False)