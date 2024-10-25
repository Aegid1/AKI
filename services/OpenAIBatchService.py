import json
import os
from datetime import datetime

import pandas as pd
from app.services.OpenAIService import OpenAIService
class OpenAIBatchService:

    openaiservice = OpenAIService()

    def create_prompt_for_batch(self, document_id: str, text: str, company_name: str, time: str):
        """
                Analyze the sentiment of a document with respect to its short-term and long-term impact on the economy

                Parameters:
                    text (str): The text of the document

                Returns:
                    prompt (str): the prompt that will be sent to the batch-api
        """
        prompt = (
                "Think step by step. Analyze the sentiment of the following economic news article about the mentioned company "
                "based on its short-term and long-term impact on the stock price of the mentioned company."
                "Rate each on a scale from -5 to +5, where -5 indicates a very negative impact, 0 indicates a neutral impact, "
                "and +5 indicates a very positive impact. "
                "Respond only with a tuple in the format ([short-term sentiment], [long-term sentiment]).\n\n"
                "Article:\n" + text
        )
        request = {
            "custom_id": document_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial analyst specializing in sentiment analysis of economic news articles"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 3
            }
        }
        return self.__add_to_batch(request, "batch_sentiments.jsonl" + company_name)

    def send_batch(self, batch_name: str):
        """
                Send a batch of requests and store the id of the sent batch into a file.

                Parameters:
                    batch_name (str): The name of the batch file.

                Returns:
                    None
        """
        batch_input_file = self.openaiservice.client.files.create(
            file=open(batch_name, "rb"),
            purpose="batch"
        )
        batch = self.openaiservice.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "nightly eval job"
            }
        )

        #bsp. "Siemens_1_ids"
        with open(batch_name+"_ids", 'a') as file:
            file.write('\n' + batch.id)

    def retrieve_batch_results(self, batch_id: str, company_name: str):
        """
                Retrieve the results of a sent batch.

                Parameters:
                    batch_id (str): The ID of the batch.

        """
        batch = self.openaiservice.client.batches.retrieve(batch_id)
        if batch.status != "completed":
            return "Your batch is currently not ready and in the state: " + batch.status

        output_file = self.openaiservice.client.files.content(batch.output_file_id)

        content = output_file.content
        content_str = content.decode('utf-8')
        lines = content_str.splitlines()
        result_list = [json.loads(line) for line in lines]
        self.__save_batch_results_in_pickle_file(result_list, company_name)

    def check_batch_status(self, batch_id: str):
        """
                Check the status of a batch.

                Parameters:
                    batch_id (str): The ID of the batch.

                Returns:
                    str: The status of the batch.
        """
        batch = self.openaiservice.client.batches.retrieve(batch_id)
        return batch.status

    def delete_batch_file(self, batch_name):
        """
                Delete a batch file which contains the ids of the batches.

                Parameters:
                    batch_name (str): The name of the batch

                Returns:
                    None
        """

        try:
            os.remove(batch_name)
            print(f"File '{batch_name}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting file '{batch_name}': {e}")

    def get_batch_ids(self, batch_name):
        """
                Get batch IDs from a file.

                Parameters:
                    batch_type (str): The type of document (SUMMARY or KEYWORDS).

                Returns:
                    list: A list of batch IDs.
        """
        batch_ids = []

        with open(batch_name, "r") as file:
            for line in file:
                batch_id = line.strip()
                if batch_id:
                    batch_ids.append(batch_id)
        return batch_ids

    def __add_to_batch(self, request: dict, file_name: str):
        """
                Add a request to a batch file.

                Parameters:
                    request (dict): The request to add.
                    file_name (str): The name of the batch file.

                Returns:
                    None
        """
        with open(file_name, 'a') as file:
            json_str = json.dumps(request)
            file.write(json_str + '\n')

    def __save_batch_results_in_pickle_file(self, results, company_name:str):
        data = []
        for result in results:
            document_id = result.get("custom_id")
            response = result.get("response").get("body").get("choices")[0].get("message").get("content")  # gets the actual result
            # hier alles in einen dataframe schreiben und zu jedem ergebnis mit der uuid das entsprechende datum hinzufügen-> uuid | datum | sentiment

            short_term_sentiment, long_term_sentiment = eval(response)
            #für jedes document noch das datum hinzufügen
            file_path = f"../data/news/{company_name}/{document_id}"
            with open(file_path, 'r') as file:
                date_str = file.readline().split(',')[0]
                date = datetime.strptime(date_str, "%d.%m.%Y")

            data.append({
                "document_id": document_id,
                "date": date,
                "short_term_sentiment": short_term_sentiment,
                "long_term_sentiment": long_term_sentiment
            })

        df = pd.DataFrame(data, columns=["document_id", "short_term_sentiment", "long_term_sentiment"])
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        pickle_filename = f"{min_date}_to_{max_date}.pkl"
        df.to_pickle(pickle_filename)
