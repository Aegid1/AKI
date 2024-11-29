from fastapi import APIRouter, Depends
from services.StockService import StockService
from basemodel.StockMetadata import StockMetadata

router = APIRouter()

@router.post("/stocks/retrieve/all")
def retrieve_stock_data(stock_metadata: StockMetadata, stocks_service: StockService = Depends()):
    """
        Retrieve and save stock data for a specific company.

        This endpoint fetches historical stock data for the given company based on
        its ticker symbol and a specified date range. The data is processed and saved
        for further analysis or use.

        Parameters:
        - stock_metadata (StockMetadata): Metadata containing the following information:
            - `ticker_symbol` (str): The stock's ticker symbol.
            - `start_date` (str): The start date for the data retrieval in "YYYY-MM-DD" format.
            - `end_date` (str): The end date for the data retrieval in "YYYY-MM-DD" format.
            - `company_name` (str): The name of the company for which stock data is retrieved.
        - stocks_service (StockService): Service responsible for handling stock data retrieval
          and storage, provided via dependency injection.
        """
    stocks_service.save_stock_data(stock_metadata.ticker_symbol, stock_metadata.start_date, stock_metadata.end_date, stock_metadata.company_name)