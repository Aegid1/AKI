import csv

import openai
import requests
import yaml
import re
from openai import OpenAI

class NewsService:
    url = "https://newsnow.p.rapidapi.com/newsv2"
    config = yaml.safe_load(open("openai_config.yaml"))
    client = OpenAI(api_key=config['KEYS']['openai'])

    def get_articles(self, topic: str, page_number: int, start_date: str, end_date: str):
        """
                Get articles from the News API based on the provided topic and date range.

                Parameters:
                    topic (str): The topic to search for.
                    page_number (int): The page number to retrieve.
                    start_date (str): The start date for the search (DD/MM/YYYY).
                    end_date (str): The end date for the search (DD/MM/YYYY).

                Returns:
                    dict: The response from the News API as a JSON object.
        """

        headers = {
            "x-rapidapi-key": self.config["KEYS"]["rapid-api"],
            "x-rapidapi-host": "newsnow.p.rapidapi.com",
            "Content-Type": "application/json"
        }

        #maybe test with variable amount of pages
        payload = {
            "query": topic,
            "time_bounded": True,
            "from_date": start_date,
            "to_date": end_date,
            "location": "de",
            "language": "de",
            "page": page_number
        }
        response = requests.post(self.url, json=payload, headers=headers)
        return response.json()

    def get_relevant_related_topics(self, company_name: str):
        prompt = (f"Denke Schritt für Schritt. Suche mir zum Unternehmen {company_name} die aktuellen Informationen zu folgenden Kontexten. "
                  "Beachte dabei, dass du die Informationen mit einem Komma trennst und in einem Tupel wiedergibst. "
                  "Gib mir nur die präzisen Antworten darauf. Verfolge folgendes Schema:\n"
                  "Branchen: (Dienstleistungsbranche, Gesundheitssektor, Verkehrswesen, ...)\n"
                  "Kunden: (Staat, BeispielAG, ....)\n"
                  "Wettbewerber: (MustermannGmbH, Max e.K., ...)\n\n"
                  "Fragen:\n\n"
                  f"5 Branchen in denen {company_name} am meisten tätig ist:\n"
                  f"5 größten Kunden von {company_name}:\n"
                  f"5 größten Wettbewerber von {company_name}:"
                  )

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    def store_relevant_related_topics(self, company_name):
        result = self.get_relevant_related_topics(company_name)
        branches_values, customers_values, competitors_values = self.__extract_relevant_results(result)

        with open(f"data/news/{company_name}/{company_name}_relevant_topics.csv", "w") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['Unternehmen', 'Branchen', 'Kunden', 'Wettbewerber'])
            csv_writer.writerow([company_name, branches_values, customers_values, competitors_values])

    def get_relevant_topics_as_list(self, company_name):
        file_path = f"data/news/{company_name}/{company_name}_relevant_topics.csv"
        relevant_topics = []
        with open(file_path, "r") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                for value in row:
                    relevant_topics.extend(value.split(','))

        return relevant_topics

    def __extract_relevant_results(self, result:str):
        branches = re.search(r'Branchen:\s*\((.*?)\)', result)
        customers = re.search(r'Kunden:\s*\((.*?)\)', result)
        competitors = re.search(r'Wettbewerber:\s*\((.*?)\)', result)

        branches_values = branches.group(1).strip() if branches else ""
        customers_values = customers.group(1).strip() if customers else ""
        competitors_values = competitors.group(1).strip() if competitors else ""

        return branches_values, customers_values, competitors_values