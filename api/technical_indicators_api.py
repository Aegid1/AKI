import os

from fastapi import APIRouter, Depends
from services.TechnicalIndicatorsService import TechnicalindicatorsService

router = APIRouter()

@router.get("/technical/indicators")
def save_technical_indicators(technical_indicator_service: TechnicalindicatorsService = Depends()):
    company_names = ["Airbus", "Allianz", "Deutsche Telekom", "Mercedes-Benz", "Volkswagen", "Porsche", "SAP",
                     "Siemens", "Siemens Healthineers", "Deutsche Bank", "BMW"]
    for company in company_names:
        file_path = os.path.join("data", "stocks", f"{company}")
        output_dir = os.path.join("data", "technical_indicators", f"{company}.csv")

        df = technical_indicator_service.get_stock_data_as_df(file_path)
        df['RSI'] = technical_indicator_service.calculate_rsi(df, 14)

        upperband, middleband, lowerband = technical_indicator_service.calculate_bollinger_baender(df, 20)
        df['UpperBand'] = upperband
        df['MiddleBand'] = middleband
        df['LowerBand'] = lowerband

        df['SMA'] = technical_indicator_service.calculate_simple_moving_average(df, 20)
        df['EMA'] = technical_indicator_service.calculate_exponential_moving_average(df, 20)

        macd, macdsignal, macdhist = technical_indicator_service.calculate_moving_average_convergence_divergence(df, 12,26, 9)
        df['MACD'] = macd
        df['MACDSignal'] = macdsignal
        df['MACDHist'] = macdhist

        df.fillna(0, inplace=True)

        df.to_csv(output_dir, index=False)

