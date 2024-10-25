from fastapi import APIRouter, Depends, HTTPException
from services.OpenAIBatchService import OpenAIBatchService

router = APIRouter()


@router.post("/batch/{batch_name}")
def send_batch(batch_name: str, batch_api_service: OpenAIBatchService = Depends()):
    """
    Send a batch of documents to the OpenAI service.

    Parameters:
        batch_name (str): The name of the batch to be sent.
        batch_api_service (OpenAIService): The OpenAI service instance.

    Returns:
        None
    """
    batch_api_service.send_batch(batch_name)

@router.get("/batch/retrieval/{batch_type}")
def retrieve_all_batches(company_name:str, batch_api_service: OpenAIBatchService = Depends()):
    """
    Retrieve the content of batches and delete the batch files, where the ids of the batches are stored

    Parameters:
        batch_api_service (OpenAIService): The OpenAI service instance.

    Returns:
        None
    """
    batch_ids = batch_api_service.get_batch_ids(batch_type)
    for batch_id in batch_ids:
        batch_api_service.retrieve_batch_results(batch_id, company_name)

    batch_api_service.delete_batch_file(batch_type)
