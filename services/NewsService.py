import csv
import re
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd
import pytz
import requests
import yaml
from dateutil.relativedelta import relativedelta
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from bs4 import BeautifulSoup

class NewsService:

    config = yaml.safe_load(open("openai_config.yaml"))
    client = OpenAI(api_key=config['KEYS']['openai'])

    tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
    model = AutoModel.from_pretrained('bert-base-german-cased')

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
        url = "https://newsnow.p.rapidapi.com/newsv2"
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
        response = requests.post(url, json=payload, headers=headers)
        return response.json()


    # def get_relevant_related_topics(self, company_name: str):
    #     prompt = (f"Denke Schritt für Schritt. Suche mir zum Unternehmen {company_name} die aktuellen Informationen zu folgenden Kontexten. "
    #               "Beachte dabei, dass du die Informationen mit einem Komma trennst und in einem Tupel wiedergibst. "
    #               "Gib mir nur die präzisen Antworten darauf. Verfolge folgendes Schema:\n"
    #               "Branchen: (Dienstleistungsbranche, Gesundheitssektor, Verkehrswesen, ...)\n"
    #               "Wettbewerber: (MustermannGmbH, Max e.K., ...)\n\n"
    #               "Fragen:\n\n"
    #               f"5 Branchen in denen {company_name} am meisten tätig ist:\n"
    #               f"5 größten Wettbewerber von {company_name}:"
    #               )
    #
    #     response = self.client.chat.completions.create(
    #         model="gpt-4",
    #         messages=[{"role": "user", "content": prompt}]
    #     )
    #
    #     return response.choices[0].message.content


    # def store_relevant_related_topics(self, company_name):
    #     result = self.get_relevant_related_topics(company_name)
    #     branches_values, competitors_values = self.__extract_relevant_results(result)
    #
    #     with open(f"data/news/{company_name}/{company_name}_relevant_topics.csv", "w") as file:
    #         csv_writer = csv.writer(file)
    #         csv_writer.writerow(['Unternehmen', 'Branchen', 'Kunden', 'Wettbewerber'])
    #         csv_writer.writerow([company_name, branches_values, f"{company_name} Kunde", competitors_values])


    # def get_relevant_topics_as_list(self, company_name):
    #     file_path = f"data/news/{company_name}/{company_name}_relevant_topics.csv"
    #     relevant_topics = []
    #     with open(file_path, "r") as file:
    #         csv_reader = csv.reader(file)
    #         next(csv_reader)
    #         for row in csv_reader:
    #             for value in row:
    #                 relevant_topics.extend(value.split(','))
    #
    #     return relevant_topics


    def check_if_article_is_relevant(self, topic: str, content: str, threshold: float, keyword: str):
        if keyword.lower() not in content.lower(): return False #check whether the content contains the topic
        return True
        #TODO the cosine similarity check is currently not used, due to the deadline of the project -> use later on

        # keywords = self.load_keywords_from_csv("data/embedding_comparison_keywords.csv")
        #
        # similarities = []
        #
        # for keyword in keywords:
        #     embedding_topic = self.get_embedding(f"{topic} {keyword}", self.tokenizer, self.model)
        #     embedding_content = self.get_embedding(content, self.tokenizer, self.model)
        #
        #     similarity = cosine_similarity(embedding_topic, embedding_content)[0][0]
        #     length_topic = len(topic.split())
        #     length_content = len(content.split())
        #     length_factor = self.determine_length_factor(length_topic, length_content)
        #
        #     adjusted_similarity = similarity * length_factor
        #     similarities.append(adjusted_similarity)
        #     print(f"Ähnlichkeit für Keyword '{keyword}': {similarity}")
        #
        # avg_similarity = sum(similarities) / len(similarities)
        # print(f"{content[:50]}: {avg_similarity}")
        # return avg_similarity >= threshold


    def get_embedding(self, text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings


    def determine_length_factor(self, length_topic, length_content):
        max_length_factor = 6
        raw_length_factor = np.log(1 + max(length_topic, length_content))
        length_factor = min(raw_length_factor, max_length_factor)  # Limit the length factor
        # the limit results in accepting articles with a minimum average cosine similarity of 0.55
        return length_factor


    def load_keywords_from_csv(self, csv_file):
        keywords = []
        with open(csv_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    keywords.append(row[0])
        return keywords


    def get_open_days(self, start_date_str, end_date_str):
        start_date = datetime.strptime(start_date_str, "%d/%m/%Y").date()
        end_date = datetime.strptime(end_date_str, "%d/%m/%Y").date()

        holidays = [
            date(start_date.year, 1, 1),  # Neujahrstag
            date(start_date.year, 4, 7),  # Karfreitag
            date(start_date.year, 4, 10),  # Ostermontag
            date(start_date.year, 5, 1),  # Tag der Arbeit
            date(start_date.year, 5, 18),  # Christi Himmelfahrt
            date(start_date.year, 5, 29),  # Pfingstmontag
            date(start_date.year, 10, 3),  # Tag der Deutschen Einheit
            date(start_date.year, 12, 25),  # Erster Weihnachtstag
            date(start_date.year, 12, 26)  # Zweiter Weihnachtstag
        ]

        if start_date.year != end_date.year:
            holidays.extend([
                date(end_date.year, 1, 1),
                date(end_date.year, 4, 7),
                date(end_date.year, 4, 10),
                date(end_date.year, 5, 1),
                date(end_date.year, 5, 18),
                date(end_date.year, 5, 29),
                date(end_date.year, 10, 3),
                date(end_date.year, 12, 25),
                date(end_date.year, 12, 26)
            ])

        open_days = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5 and current_date not in holidays:
                open_days.append(current_date.strftime("%d/%m/%Y"))
            current_date += timedelta(days=1)

        return open_days


    # def __extract_relevant_results(self, result:str):
    #     branches = re.search(r'Branchen:\s*\((.*?)\)', result)
    #     competitors = re.search(r'Wettbewerber:\s*\((.*?)\)', result)
    #
    #     branches_values = branches.group(1).strip() if branches else ""
    #     competitors_values = competitors.group(1).strip() if competitors else ""
    #
    #     return branches_values, competitors_values


    def get_actual_date_of_article(self, url:str, placeholder_date:str):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            raise ValueError("No timestamp found")

        date = None

        # try to extract the date from <meta> tag
        meta_date = soup.find('meta', {'property': 'article:published_time'})
        if meta_date:
            date = meta_date.get('content')

        if not date:
            meta_date_fallback = soup.find('meta', {'name': 'date'}) or soup.find('meta', {'property': 'datePublished'})
            if meta_date_fallback:
                date = meta_date_fallback.get('content')

        #try to extract the date from <time> tag
        if not date:
            time_tag = soup.find('time')
            if time_tag:
                date = time_tag.get('datetime')

        if not date:
            text = soup.get_text()
            time_match = re.search(r'\b(\d{2}:\d{2}(?::\d{2})?)\b', text)  #searches in text for time format 'HH:MM' oder 'HH:MM:SS'
            if time_match:
                time = time_match.group(0)
                placeholder_date_obj = datetime.fromisoformat(placeholder_date)
                placeholder_date_iso = placeholder_date_obj.strftime("%Y-%m-%d")
                date = f"{placeholder_date_iso} {time}"  # Format: 'YYYY-MM-DD HH:MM[:SS]'

        if not date or not re.search(r'\d{2}:\d{2}(:\d{2})?', date):
            #date = placeholder_date
            raise ValueError("No timestamp found")

        try:
            date_obj = datetime.fromisoformat(date)
            # transform in UTC, if necessary
            date_utc = date_obj.astimezone(pytz.UTC)
            formatted_date = date_utc.strftime("%a, %d %b %Y %H:%M:%S GMT")
            return formatted_date
        except ValueError:
            return date


    def merge_articles(self, company_name:str, month:str, year:str):
        df1 = pd.read_csv(f"data/news/Wirtschaft/Wirtschaft_sorted_{month}_{year}.csv", header=None, names=["ID", "Datum", "Text"])
        df2 = pd.read_csv(f"data/news/{company_name}/{company_name}_sorted_{month}_{year}.csv", header=None, names=["ID", "Datum", "Text"])

        merged_df = pd.concat([df1, df2], ignore_index=True)

        merged_df['Datum'] = pd.to_datetime(merged_df['Datum'], errors='coerce')
        merged_df = merged_df.sort_values(by='Datum').reset_index(drop=True)

        merged_df.to_csv(f"data/news/{company_name}/{company_name}_merged_{month}_{year}.csv", index=False, header=False)

    def create_monthly_intervals(self, start_date: str, end_date: str):
        start_date = datetime.strptime(start_date, "%d/%m/%Y")
        end_date = datetime.strptime(end_date, "%d/%m/%Y")

        dates = []
        current_start = end_date
        while current_start > start_date:
            current_end = current_start
            current_start = current_start - relativedelta(months=1)
            dates.append((current_start.strftime("%d/%m/%Y"), current_end.strftime("%d/%m/%Y")))

        return dates