import re

import requests
import yaml

class TwitterService:
    config = yaml.safe_load(open("openai_config.yaml"))

    def get_tweets_by_hashtag_and_date(self, company_name:str, start_date:str, end_date:str):
        url = "https://twitter-api-v1-1-enterprise.p.rapidapi.com/base/apitools/search"
        api_key = self.config["KEYS"]["rapid-api"]
        querystring = {
            "words": company_name,
            "until": start_date,
            "since": end_date,
            "tag": company_name,
            "resFormat": "json",
            "apiKey": api_key
        }

        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "twitter-api-v1-1-enterprise.p.rapidapi.com"
        }

        return requests.get(url, headers=headers, params=querystring)

    def extract_results_from_twitter_api(self, api_result):
        posts = api_result.get("data").get("data").get("search_by_raw_query").get("search_timeline").get("timeline").get("instructions").get("0").get("entries")
        for post in posts:
            content = post.get("content").get("itemContent").get("tweet_results").get("result").get("legacy").get("full_text")
            date = post.get("content").get("itemContent").get("tweet_results").get("result").get("legacy").get("created_at") #time format: Tue Oct 31 23:13:56 +0000 2023
            content = re.sub(r'\s+', ' ', content).strip()  # news texts contain unnecessary newlines

        return date, content