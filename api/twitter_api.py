import re
import time
import uuid

import pandas as pd
from fastapi import APIRouter, Depends
from basemodel.TwitterApiRequest import TwitterApiRequest
from services.NewsService import NewsService
from services.TwitterService import TwitterService

router = APIRouter()

@router.post("/twitter/posts/all")
def store_all_relevant_articles_from_news_api(request: TwitterApiRequest, twitter_service: TwitterService = Depends(), news_service: NewsService = Depends()):
    start_time = time.time()
    articles_list = []
    days = news_service.get_open_days(request.start_date, request.end_date) #TODO get open days in utils

    for i in range(len(days)-1):
        print(f"DAY: {days[i]} to {days[i+1]}")
        response = twitter_service.get_tweets_by_hashtag_and_date(request.company_name, days[i], days[i+1])
        date, content = twitter_service.extract_results_from_twitter_api(response)
        converted_date = pd.to_datetime(date, format='%a, %d %b %Y %H:%M:%S %Z')
        articles_list.append((str(uuid.uuid4()), converted_date, content))

        i+=2

    articles_df = pd.DataFrame(articles_list, columns=['UUID', 'Date', 'Text'])
    articles_df = articles_df.drop_duplicates(subset='Text')
    articles_sorted = articles_df.sort_values(by='Date')
    articles_sorted.to_csv(f"data/twitter_posts/{request.company_name}/{request.company_name}_sorted.csv", index=False, header=False)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Die Funktion hat {duration:.2f} Sekunden gebraucht.")
