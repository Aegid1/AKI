import csv
import uuid

import pandas as pd
import requests
import yaml

class NewsService:
    url = "https://newsnow.p.rapidapi.com/newsv2"
    config = yaml.safe_load(open("openai_config.yaml"))

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