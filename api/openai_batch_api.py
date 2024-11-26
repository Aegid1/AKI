import os
import time
from datetime import datetime
from itertools import count

from fastapi import APIRouter, Depends
from basemodel.BatchMetadata import BatchMetadata
from services.OpenAIBatchService import OpenAIBatchService

router = APIRouter()

@router.post("/batch/create")
def create_batch(batch_metadata: BatchMetadata, batch_api_service: OpenAIBatchService = Depends()):
    #hier durch alle files in dem gegebenen Ordner data/news/[company_name] durchgehen und durch jede zeile in jedem file iterieren
    folder_path = "data/news/" + batch_metadata.company_name
    current_year, current_week = None, None
    batch_size = 10
    batch_number = 1
    for filename in os.listdir(folder_path):
        if not "merged" in filename: continue
        file_path = os.path.join(folder_path, filename)
        count_of_lines = 0
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                count_of_lines += 1
                if count_of_lines > 10:
                    batch_number += 1
                    count_of_lines = 0
                document_id, date, text=line.split(",", 2)

                date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                year, week, _ = date.isocalendar()

                if current_year != year or current_week != week:
                    current_year, current_week = year, week
                    count_of_lines += 1

                #der name der batch file kann bspw. später so aussehen: batch_Siemens_1.jsonl, batch_Siemens_2.jsonl, batch_SAP_1.jsonl
                batch_filename = f"batch_{batch_metadata.company_name}_{current_year}_W{current_week}_{batch_number}.jsonl"
                batch_api_service.create_one_prompt_for_batch(document_id,
                                                              text,
                                                              batch_filename,
                                                              batch_metadata.company_name)


@router.post("/batch/{company_name}")
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
    folder_path = "data/batches/" + company_name
    batch_id = None
    for filename in os.listdir(folder_path):

        if batch_id: batch_api_service.check_batch_status(batch_id)

        file_path = os.path.join(folder_path, filename)
        batch_id = batch_api_service.send_batch(file_path, company_name)

        # try:
        #     os.remove(file_path)
        #     print(f"deleting file {file_path}")
        # except Exception as e:
        #     print(f"Error deleting file {file_path}: {e}")

@router.get("/batch/retrieval/{company_name}")
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

    #batch_api_service.delete_batch_ids_file(company_name)
