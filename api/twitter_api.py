import json
import re
import time
import uuid

import pandas as pd
from fastapi import APIRouter, Depends
from basemodel.TwitterApiRequest import TwitterApiRequest
from services.NewsService import NewsService
from services.TwitterService import TwitterService

router = APIRouter()

@router.get("/twitter/posts/all")
def store_all_relevant_articles_from_news_api(request: TwitterApiRequest, twitter_service: TwitterService = Depends(), news_service: NewsService = Depends()):
    start_time = time.time()
    articles_list = []
    days = twitter_service.get_open_days_twitter(request.start_date, request.end_date) #TODO get open days in utils
    print(days)
    for i in range(len(days)-1):
        print(f"DAY: {days[i]} to {days[i+1]}")
        response = twitter_service.get_tweets_by_hashtag_and_date(request.company_name, days[i], days[i+1]).content.decode("utf-8")
        response = json.loads(response)
        posts = twitter_service.extract_results_from_twitter_api(response)
        for post in posts:
            if post:
                content = post.get("content").get("itemContent").get("tweet_results").get("result").get("legacy").get("full_text")
                date = post.get("content").get("itemContent").get("tweet_results").get("result").get("legacy").get("created_at") #time format: Tue Oct 31 23:13:56 +0000 2023
                content = re.sub(r'\s+', ' ', content).strip()  # news texts contain unnecessary newlines
                print(content)
                print(date)
                converted_date = pd.to_datetime(date, format='%a %b %d %H:%M:%S %z %Y')
                articles_list.append((str(uuid.uuid4()), converted_date, content))

        i+=2

    articles_df = pd.DataFrame(articles_list, columns=['UUID', 'Date', 'Text'])
    articles_df = articles_df.drop_duplicates(subset='Text')
    articles_sorted = articles_df.sort_values(by='Date')
    articles_sorted.to_csv(f"data/twitter_posts/{request.company_name}/{request.company_name}_sorted.csv", index=False, header=False)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Die Funktion hat {duration:.2f} Sekunden gebraucht.")
