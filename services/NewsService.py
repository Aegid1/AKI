import csv
from datetime import date, timedelta

import numpy as np
import requests
import yaml
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class NewsService:

    config = yaml.safe_load(open("openai_config.yaml"))
    client = OpenAI(api_key=config['KEYS']['openai'])

    # tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
    # model = AutoModel.from_pretrained('yiyanghkust/finbert-pretrain')
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


    def get_economy_news(self, page_number: int, start_date: str, end_date: str):
        url = "https://newsnow.p.rapidapi.com/newsv2"
        headers = {
            "x-rapidapi-key": self.config["KEYS"]["rapid-api"],
            "x-rapidapi-host": "newsnow.p.rapidapi.com",
            "Content-Type": "application/json"
        }

        #maybe test with variable amount of pages
        payload = {
            "query": "Wirtschaft",
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
        keywords = self.load_keywords_from_csv("data/embedding_comparison_keywords.csv")

        similarities = []

        for keyword in keywords:
            embedding_topic = self.get_embedding(f"{topic} {keyword}", self.tokenizer, self.model)
            embedding_content = self.get_embedding(content, self.tokenizer, self.model)

            similarity = cosine_similarity(embedding_topic, embedding_content)[0][0]
            length_topic = len(topic.split())
            length_content = len(content.split())
            length_factor = np.log(1 + max(length_topic, length_content))  # Beispiel für logarithmische Gewichtung

            adjusted_similarity = similarity * length_factor

            similarities.append(adjusted_similarity)
            print(f"Ähnlichkeit für Keyword '{keyword}': {similarity}")

        avg_similarity = sum(similarities) / len(similarities)
        print(f"{content[:50]}: {avg_similarity}")
        return avg_similarity >= threshold


    def load_keywords_from_csv(self, csv_file):
        keywords = []
        with open(csv_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:  # Falls die Zeile nicht leer ist
                    keywords.append(row[0])
        return keywords


    def get_open_days_2023(self, year):
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        # Feiertage 2023 in Deutschland, an denen die Börse geschlossen ist
        holidays = [
            date(year, 1, 1),  # Neujahrstag
            date(year, 4, 7),  # Karfreitag
            date(year, 4, 10),  # Ostermontag
            date(year, 5, 1),  # Tag der Arbeit
            date(year, 5, 18),  # Christi Himmelfahrt
            date(year, 5, 29),  # Pfingstmontag
            date(year, 10, 3),  # Tag der Deutschen Einheit
            date(year, 12, 25),  # Erster Weihnachtstag
            date(year, 12, 26)  # Zweiter Weihnachtstag
        ]
        open_days = []
        current_date = start_date
        while current_date <= end_date:

            if current_date.weekday() < 5 and current_date not in holidays:
                open_days.append(current_date.strftime("%d/%m/%Y"))
            current_date += timedelta(days=1)

        return open_days


    def get_embedding(self, text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings


    # def __extract_relevant_results(self, result:str):
    #     branches = re.search(r'Branchen:\s*\((.*?)\)', result)
    #     competitors = re.search(r'Wettbewerber:\s*\((.*?)\)', result)
    #
    #     branches_values = branches.group(1).strip() if branches else ""
    #     competitors_values = competitors.group(1).strip() if competitors else ""
    #
    #     return branches_values, competitors_values