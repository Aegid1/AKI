import csv
import json
import os
from datetime import datetime
import pandas as pd
import yaml
from openai import OpenAI

class OpenAIBatchService:

    config = yaml.safe_load(open("openai_config.yaml"))
    client = OpenAI(api_key=config['KEYS']['openai'])

    def create_one_prompt_for_batch(self, document_id: str, text: str, file_name: str, company_name: str):
        """
                Analyze the sentiment of a document with respect to its short-term and long-term impact on the economy

                Parameters:
                    text (str): The text of the document

                Returns:
                    prompt (str): the prompt that will be sent to the batch-api
        """
        prompt = (
                f"Think step by step. Analyze the sentiment of the following economic news article about {company_name}"
                f"based on its short-term and long-term impact on the stock price of {company_name}."
                "Rate each on a scale from -10 to +10, where -10 indicates a very negative impact, 0 indicates a neutral impact, "
                f"and +10 indicates a very positive impact. Figure out which relevancy the article has to the stock price of {company_name}"
                "and determine a relevancy_factor between 1 and 0. 1 indicates a high relevancy and 0 indicates a low relevancy."
                "Respond only with a triple in the format ([short-term sentiment], [long-term sentiment], [relevancy_factor]).\n\n"
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
                "max_tokens": 10
            }
        }
        return self.__add_to_batch(request, file_name, company_name)

    def send_batch(self, batch_file_name: str, company_name: str):
        """
                Send a batch of requests and store the id of the sent batch into a file.

                Parameters:
                    batch_file_name (str): The name of the batch file.

                Returns:
                    None
        """
        batch_input_file = self.client.files.create(
            file=open(batch_file_name, "rb"),
            purpose="batch"
        )
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "nightly eval job"
            }
        )

        #bsp. "data/batch_ids/Siemens"
        with open("data/batch_ids/"+ company_name + "_batch_ids", 'a') as file:
            file.write('\n' + batch.id)

    def retrieve_batch_results(self, batch_id: str, company_name: str):
        """
                Retrieve the results of a sent batch.

                Parameters:
                    batch_id (str): The ID of the batch.

        """
        batch = self.client.batches.retrieve(batch_id)
        if batch.status != "completed":
            return "Your batch is currently not ready and in the state: " + batch.status

        output_file = self.client.files.content(batch.output_file_id)

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
        batch = self.client.batches.retrieve(batch_id)
        return batch.status

    def delete_batch_ids_file(self, batch_name):
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

    def get_batch_ids(self, company_name:str):
        """
                Get batch IDs from a file.

                Parameters:
                    batch_type (str): The type of document (SUMMARY or KEYWORDS).

                Returns:
                    list: A list of batch IDs.
        """
        batch_ids = []

        with open("data/batch_ids/" + company_name + "_batch_ids", "r") as file:
            for line in file:
                batch_id = line.strip()
                if batch_id:
                    batch_ids.append(batch_id)
        return batch_ids

    def __add_to_batch(self, request: dict, file_name: str, company_name: str):
        """
                Add a request to a batch file in data/batches/[company_name].

                Parameters:
                    request (dict): The request to add.
                    company_name (str): The name of the company

                Returns:
                    None
        """
        with open("data/batches/" + company_name + "/" + file_name, 'a') as file:
            json_str = json.dumps(request)
            file.write(json_str + '\n')

    def __save_batch_results_in_pickle_file(self, results, company_name:str):
        data = []
        #iterate through all results
        for result in results:
            document_id = result.get("custom_id")
            response = result.get("response").get("body").get("choices")[0].get("message").get("content")  # gets the actual result

            #get the determined long-term and short-term sentiment of the result
            try:
                short_term_sentiment, long_term_sentiment, relevancy = eval(response)
            except (SyntaxError, ValueError) as e:
                print(f"Fehler beim Verarbeiten von document_id {document_id} mit dem response: {response}: {e}")
                continue

            #search for the date of the document, this is needed for structuring the data in the later pickle files
            date = self.__get_date_of_document(document_id, company_name)
            data.append({
                "document_id": document_id,
                "date": date,
                "short_term_sentiment": short_term_sentiment,
                "long_term_sentiment": long_term_sentiment,
                "relevancy_factor": relevancy
            })

        df = pd.DataFrame(data, columns=["document_id", "date", "short_term_sentiment", "long_term_sentiment"])
        #date needs to be formatted so min and max can be determined -> however we want to see h,m,s of the timestamp in the dataframe
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.strftime("%Y-%m-%d %H:%M:%S")

        min_date = df["date"].min()
        max_date = df["date"].max()
        pickle_filename = f"data/sentiments_news/{company_name}/{min_date}_to_{max_date}.pkl"
        df.to_pickle(pickle_filename)

    def __get_date_of_document(self, document_id, company_name):
        directory = f"data/news/{company_name}/"
        for filename in os.listdir(directory):
            if filename.endswith(".csv") and "merged" in filename:
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        uuid = row[0]
                        if uuid == "e704db97-2ceb-4b3b-a7d8-5b36eb3243cf":
                            date_str = row[1]

                            date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S%z")
                            return date