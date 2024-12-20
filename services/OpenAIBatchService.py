import csv
import json
import os
import time
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
        text = self.__truncate_to_char_limit(text, 5000)

        prompt = (
                f"Think step by step. Analyze the sentiment of the following economic news article about {company_name}"
                f"based on its short-term and long-term impact on the stock price of {company_name}."
                "Rate each on a scale from -10 to +10, where -10 indicates a very negative impact, 0 indicates a neutral impact, "
                f"and +10 indicates a very positive impact. Figure out which relevancy the article has to the stock price of {company_name}"
                "and determine a relevancy_factor between 1 and 0. 1 indicates a high relevancy and 0 indicates a low relevancy."
                "Respond only with a triple in the format ([short-term sentiment], [long-term sentiment], [relevancy_factor]). "
                "Example Output1: (9, 2, 0.85)\n\n"
                "Example Output2: (-7, -3, 0.7)\n\n"
                "Article:\n" + text
        )
        request = {
            "custom_id": document_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo-0125",
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
                "max_tokens": 15
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

        return batch.id

    def retrieve_batch_results(self, batch_id: str, company_name: str):
        """
                Retrieve the results of a sent batch.

                Parameters:
                    batch_id (str): The ID of the batch.

        """
        batch = self.client.batches.retrieve(batch_id)
        input_file_name = self.__get_input_file_name_from_batch(batch)
        print(input_file_name)

        if batch.status != "completed":
            return "Your batch is currently not ready and in the state: " + batch.status

        output_file = self.client.files.content(batch.output_file_id)
        content = output_file.content
        content_str = content.decode('utf-8')
        lines = content_str.splitlines()
        result_list = [json.loads(line) for line in lines]
        self.__save_batch_results_in_csv_file(result_list, company_name)
        self.__delete_batch_file(input_file_name, company_name)

    def check_batch_status(self, batch_id: str):
        """
                Check the status of a batch.

                Parameters:
                    batch_id (str): The ID of the batch.

                Returns:
                    str: The status of the batch.
        """
        batch_status = self.client.batches.retrieve(batch_id).status
        epoch = 0
        while batch_status != "completed":
            print("EPOCH: " + str(epoch))
            print(batch_status)
            time.sleep(10)
            batch_status = self.client.batches.retrieve(batch_id).status
            epoch += 1

        return batch_status

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
        json_str = json.dumps(request, ensure_ascii=False)
        with open("data/batches/" + company_name + "/" + file_name, 'a',  encoding='utf-8') as file:
            file.write(json_str + '\n')

    def __save_batch_results_in_csv_file(self, results, company_name: str):
        """
            Saves batch sentiment analysis results to a CSV file.

            This function processes the batch of results containing sentiment analysis data for different documents. It
            extracts the relevant information (document ID, sentiment analysis, and date), organizes it into a structured
            DataFrame, and saves the results as a CSV file.

            Parameters:
            - results (list): A list of result dictionaries, each containing the sentiment analysis data for a document.
            - company_name (str): The name of the company to be used in the output CSV file's path and filename.

            Workflow:
            1. Iterates through the provided results to extract document IDs, sentiment analysis responses, and their corresponding
               dates.
            2. Extracts short-term sentiment, long-term sentiment, and relevancy from the response, handling any issues with
               incorrect formatting.
            3. Retrieves the date of the document based on the document ID and company name.
            4. Creates a DataFrame containing the extracted data.
            5. Converts the date column to a proper datetime format.
            6. Determines the minimum and maximum date in the dataset for naming the CSV file.
            7. Saves the DataFrame as a CSV file with the name corresponding to the date range.

            Example:
            - The resulting CSV file will be saved in the "data/sentiments_news/{company_name}/{min_date}_to_{max_date}.csv" format,
              where `{min_date}` and `{max_date}` represent the date range of the sentiment analysis results.
            """
        data = []
        # Iterate through all results
        for result in results:
            document_id = result.get("custom_id")
            response = result.get("response").get("body").get("choices")[0].get("message").get(
                "content")  # gets the actual result

            # Get the determined long-term and short-term sentiment of the result
            try:
                open_brackets = response.count("(")
                close_brackets = response.count(")")
                if open_brackets + close_brackets != 2:
                    response = response.replace(")", "")
                    response = response.replace("(", "")
                    response = response.replace("[", "")
                    response = response.replace("]", "")
                    response_splitted = response.split(",")
                    short_term_sentiment, long_term_sentiment, relevancy = response_splitted[0], response_splitted[1], response_splitted[2]
                else:
                    short_term_sentiment, long_term_sentiment, relevancy = eval(response)
            except (SyntaxError, ValueError, IndexError) as e:
                print(f"Fehler beim Verarbeiten von document_id {document_id} mit dem response: {response}: {e}")
                continue

            # Search for the date of the document
            date = self.__get_date_of_document(document_id, company_name)
            data.append({
                "document_id": document_id,
                "date": date,
                "short_term_sentiment": short_term_sentiment,
                "long_term_sentiment": long_term_sentiment,
                "relevancy_factor": relevancy
            })

        df = pd.DataFrame(data, columns=["document_id", "date", "short_term_sentiment", "long_term_sentiment",
                                         "relevancy_factor"])

        df['date'] = pd.to_datetime(df['date'])

        # Determine min and max date
        min_date = str(df["date"].min().date())
        max_date = str(df["date"].max().date())

        df['date'] = df['date'].dt.strftime("%Y-%m-%d %H:%M:%S")

        csv_filename = os.path.join("data", "sentiments_news", f"{company_name}", f"{min_date}_to_{max_date}.csv")
        # Save DataFrame as CSV
        df.to_csv(csv_filename, index=False)

    def __truncate_to_char_limit(self, text, char_limit):
        """
            Truncates the input text to a specified character limit.

            This function checks if the length of the text exceeds the given character limit and truncates the text
            accordingly. If the length is within the limit, it returns the original text.

            Parameters:
            - text (str): The text to be truncated.
            - char_limit (int): The maximum allowed character length.

            Returns:
            - str: The truncated text if the length exceeds the limit, or the original text if not.
        """
        if len(text) > char_limit:
            return text[:char_limit]
        return text


    def __delete_batch_file(self, filename: str, company_name:str):
        """
            Deletes a specified batch file from the filesystem.

            This function attempts to delete a batch file from the 'batches' directory. If the file exists, it is removed.
            Otherwise, an error message is displayed.

            Parameters:
            - filename (str): The name of the batch file to be deleted.
            - company_name (str): The name of the company (used to locate the file within the company's directory).

            Returns:
            - None: This function does not return any value.
        """
        file_path = os.path.join("data", "batches", f"{company_name}", f"{filename}")
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Die Datei {file_path} wurde gelöscht.")
            else:
                print(f"Die Datei {file_path} existiert nicht.")
        except Exception as e:
            print(f"Fehler beim Löschen der Datei {file_path}: {e}")


    def __get_date_of_document(self, document_id, company_name):
        """
            Retrieves the publication date of a document from its associated CSV file.

            This function searches through the company's merged news CSV files to find a document with the specified
            `document_id` and extracts its publication date.

            Parameters:
            - document_id (str): The ID of the document to retrieve the date for.
            - company_name (str): The name of the company (used to locate the directory containing the CSV files).

            Returns:
            - datetime: The publication date of the document.
        """
        directory = f"data/news/{company_name}/"
        for filename in os.listdir(directory):
            if filename.endswith(".csv") and "merged" in filename:
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        uuid = row[0]
                        if uuid == document_id:
                            date_str = row[1]

                            date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                            return date


    def __get_input_file_name_from_batch(self, batch):
        """
            Retrieves the input file name from a batch object.

            This function uses the `input_file_id` from the provided batch object to retrieve the associated file's name
            from the client. If the file name is found, it is returned. Otherwise, a message is printed indicating that
            the name could not be found.

            Parameters:
            - batch: The batch object containing the input file ID.

            Returns:
            - str: The name of the input file, or None if the name cannot be found.
        """
        input_file_id = batch.input_file_id
        input_file = self.client.files.retrieve(input_file_id)
        input_file_name = input_file.filename

        if input_file_name:
            return input_file_name
        else:
            print("Der Name der Input-Datei konnte nicht gefunden werden.")
            return None