import os

from fastapi import APIRouter, Depends
from basemodel.BatchMetadata import BatchMetadata
from services.OpenAIBatchService import OpenAIBatchService

router = APIRouter()

@router.post("/batch/create")
def create_batch(batch_metadata: BatchMetadata, batch_api_service: OpenAIBatchService = Depends()):
    #hier durch alle files in dem gegebenen Ordner ../data/news/[company_name] durchgehen und durch jede zeile in jedem file iterieren
    folder_path = "../data/news/" + batch_metadata.company_name
    file_count_in_batch=1
    count_of_batches=1
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as file:
            for line in file:
                #wenn batch_size erreicht, dann muss anderer batch_namen verwendet werden, 50 is recommended
                if file_count_in_batch > batch_metadata.batch_size:
                    count_of_batches+=1

                document_id, date, text=line.split(",", 2)
                #der name der batch file kann bspw. später so aussehen: batch_Siemens_1.jsonl, batch_Siemens_2.jsonl, batch_SAP_1.jsonl
                batch_api_service.create_one_prompt_for_batch(document_id, text, batch_metadata.company_name+str(count_of_batches))
                file_count_in_batch+=1

@router.post("/batch/{batch_file_name}")
def send_batch(company_name: str, batch_api_service: OpenAIBatchService = Depends()):
    """
    Send a batch of documents to the OpenAI service.

    Parameters:
        batch_name (str): The name of the batch to be sent.
        batch_api_service (OpenAIService): The OpenAI service instance.

    Returns:
        None
    """
    #iterate through all batches of a given company and send them to openai
    folder_path = "../data/batches/" + company_name
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        batch_api_service.send_batch(file_path)

@router.get("/batch/retrieval/{batch_type}")
def retrieve_all_batches(company_name:str, batch_api_service: OpenAIBatchService = Depends()):
    """
    Retrieve the content of batches and delete the batch files, where the ids of the batches are stored

    Parameters:
        batch_api_service (OpenAIService): The OpenAI service instance.

    Returns:
        None
    """
    batch_ids = batch_api_service.get_batch_ids(company_name)
    for batch_id in batch_ids:
        batch_api_service.retrieve_batch_results(batch_id, company_name)

    batch_api_service.delete_batch_ids_file(company_name)
