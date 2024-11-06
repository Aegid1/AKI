import re
import uuid
import pandas as pd
from fastapi import APIRouter, Depends
from basemodel.NewsApiRequest import NewsApiRequest
from services.NewsService import NewsService

router = APIRouter()

@router.post("/articles/news/check")
def get_articles_from_news_api(request: NewsApiRequest, news_api_service: NewsService = Depends()):
    """
        Get articles from the News API.

        Parameters:
            request (NewsApiRequest): The request containing start_date, end_date, topic, and page_number.
            news_api_service (NewsApiService): The News API service instance.

        Returns:
            list: A list of articles from the News API.
        """
    articles = news_api_service.get_articles(request.company_name, request.page_number, request.start_date,
                                             request.end_date).get("news")
    return articles


@router.post("/articles/news")
def store_articles_from_news_api(request: NewsApiRequest,
                                 news_service: NewsService = Depends()):
    """
        Store articles from the News API locally.

        Parameters:
            request (NewsApiRequest): The request containing start_date, end_date, topic, and page_number.
            news_api_service (NewsApiService): The News API service instance.

        Returns:
            None
        """
    articles = news_service.get_articles(request.company_name, request.page_number, request.start_date,
                                             request.end_date).get("news")
    articles_list = []
    for article in articles:
        content = article.get("text")
        date = article.get("date")
        converted_date = pd.to_datetime(date, format='%d %b %Y %H:%M:%S %Z')
        articles_list.append((str(uuid.uuid4()), converted_date, content))

    articles_df = pd.DataFrame(articles_list, columns=['UUID', 'Date', 'Text'])
    articles_sorted = articles_df.sort_values(by='Date')
    articles_sorted.to_csv(f"data/news/{request.company_name}/{request.company_name}_sorted.csv", index=False, header=False)


@router.post("/articles/news/relevant_topics/all")
def store_all_relevant_articles_from_news_api(request: NewsApiRequest, news_api_service: NewsService = Depends()):
    requested_pages = request.page_number
    articles_list = []
    current_page = 1
    topic = f"{request.company_name} Unternehmen"
    response = news_api_service.get_articles(topic, current_page, request.start_date, request.end_date)
    articles = response.get("news")

    while response.get("count") != 0:
        print("PAGE: " + str(current_page))
        for article in articles:
            content = article.get("text")
            content = re.sub(r'\s+', ' ', content).strip() #news texts contain unnecessary newlines

            article_is_relevant = news_api_service.check_if_article_is_relevant(topic, content, 0.3, request.company_name) #sometimes articles are not relevant -> sort those out
            if not article_is_relevant: continue

            date = article.get("date")
            converted_date = pd.to_datetime(date, format='%a, %d %b %Y %H:%M:%S %Z')
            articles_list.append((str(uuid.uuid4()), converted_date, content))

        current_page += 1
        if current_page == requested_pages:
            break
        response = news_api_service.get_articles(f"{request.company_name} Unternehmen", current_page, request.start_date, request.end_date)
        articles = response.get("news")

    articles_df = pd.DataFrame(articles_list, columns=['UUID', 'Date', 'Text'])
    articles_sorted = articles_df.sort_values(by='Date')
    print(articles_sorted)
    articles_sorted.to_csv(f"data/news/{request.company_name}/{request.company_name}_sorted.csv", index=False, header=False)


@router.post("/articles/economy_news/all")
def store_all_relevant_economy_articles_from_news_api(news_api_service: NewsService = Depends()):
    articles_list = []
    #pick out all the relevant days and iterate through them
    response = news_api_service.get_economy_news(1, "01/11/2024", "02/11/2024")
    articles = response.get("news")

    for article in articles:
        content = article.get("text")
        content = re.sub(r'\s+', ' ', content).strip() #news texts contain unnecessary newlines
        article_is_relevant = news_api_service.check_if_article_is_relevant("Wirtschaft", content, 0.78, "Wirtschaft")  # sometimes articles are not relevant -> sort those out
        if not article_is_relevant: continue

        date = article.get("date")
        converted_date = pd.to_datetime(date, format='%a, %d %b %Y %H:%M:%S %Z')
        articles_list.append((str(uuid.uuid4()), converted_date, content))

    articles_df = pd.DataFrame(articles_list, columns=['UUID', 'Date', 'Text'])
    articles_sorted = articles_df.sort_values(by='Date')
    print(articles_sorted)
    articles_sorted.to_csv(f"data/news/top_economy_news_sorted.csv", index=False, header=False)


# @router.post("/articles/news/topics/{company_name}")
# def store_relevant_related_topics(company_name: str, news_service: NewsService = Depends()):
#     news_service.store_relevant_related_topics(company_name)