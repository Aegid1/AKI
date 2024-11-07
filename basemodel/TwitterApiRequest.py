from pydantic import BaseModel
class TwitterApiRequest(BaseModel):
    start_date: str
    end_date: str
    company_name: str
    excluded_tag: str