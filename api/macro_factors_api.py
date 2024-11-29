
from fastapi import APIRouter, Depends
from services.MacroFactorsService import MacroFactorsService

router = APIRouter()

@router.get("/macro/factors")
def get_macro_factors(macro_factors_service: MacroFactorsService = Depends()):
    """
        Fetch and save macroeconomic factors for a specified date range.

        This endpoint triggers the retrieval and storage of several macroeconomic
        indicators, including central interest rates, GDP, oil prices, inflation rates,
        currency exchange rates (Euro to Dollar), and unemployment rates. The data
        is fetched for the period from November 2022 to November 2024.

        Parameters:
        - macro_factors_service (MacroFactorsService): A service for handling
          macroeconomic data operations, provided via dependency injection.

        """
    macro_factors_service.save_central_interest_rate("2022-11", "2024-11")
    macro_factors_service.save_gdp("2022-11", "2024-11")
    macro_factors_service.save_oil_prices("2022-11-01", "2024-11-01")
    macro_factors_service.save_inflation_rate("2022-11", "2024-11")
    macro_factors_service.save_currency_euro_dollar("2022-11-01", "2024-11-01")
    macro_factors_service.save_unemployment_rate("2022-11", "2024-11")
