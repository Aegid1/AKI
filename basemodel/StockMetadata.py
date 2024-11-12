from pydantic import BaseModel
class StockMetadata(BaseModel):
    ticker_symbol: str
    start_date: str
    end_date: str
    company_name: str