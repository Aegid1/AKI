from fastapi import APIRouter, Depends
from services.StockService import StockService
from basemodel.StockMetadata import StockMetadata

router = APIRouter()

@router.post("/stocks/retrieve/all")
def retrieve_stock_data(stock_metadata: StockMetadata, stocks_service: StockService = Depends()):
    stocks_service.save_stock_data(stock_metadata.ticker_symbol, stock_metadata.start_date, stock_metadata.end_date, stock_metadata.company_name)