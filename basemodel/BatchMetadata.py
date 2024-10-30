from pydantic import BaseModel
class BatchMetadata(BaseModel):
    company_name: str
    batch_size: int #50 is recommended
