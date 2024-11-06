from pydantic import BaseModel
class NewsApiRequest(BaseModel):
    start_date: str
    end_date: str
    up_to_page_number: int
    company_name: str
    is_company_given: bool