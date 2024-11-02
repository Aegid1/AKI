import uuid
import pandas as pd
from fastapi import APIRouter, Depends
from basemodel import NewsApiRequest
from services import NewsService

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
    articles = news_api_service.get_articles(request.topic, request.page_number, request.start_date,
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
    articles = news_service.get_articles(request.topic, request.page_number, request.start_date,
                                             request.end_date).get("news")
    articles_list = []
    for article in articles:
        content = article.get("text")
        date = article.get("date")
        converted_date = pd.to_datetime(date, format='%d %b %Y %H:%M:%S %Z')
        articles_list.append((str(uuid.uuid4()), converted_date, content))

    articles_df = pd.DataFrame(articles_list, columns=['UUID', 'Date', 'Text'])
    articles_sorted = articles_df.sort_values(by='Date')
    articles_sorted.to_csv(f"/data/news/{request.company_name}/{request.company_name}_sorted.csv", index=False, header=False)


@router.post("/articles/news/all")
def store_all_articles_from_news_api(request: NewsApiRequest, news_api_service: NewsService = Depends()):
    requested_pages = request.page_number
    current_page = 0
    response = news_api_service.get_articles(request.topic, current_page, request.start_date, request.end_date)
    articles = response.get("news")

    articles_list = []
    while response.get("count") != 0:
        print("PAGE: " + str(current_page))
        for article in articles:
            content = article.get("text")
            date = article.get("date")
            converted_date = pd.to_datetime(date, format='%d %b %Y %H:%M:%S %Z')
            articles_list.append((str(uuid.uuid4()), converted_date, content))

        current_page += 1
        if current_page == requested_pages:
            break
        response = news_api_service.get_articles(request.topic, current_page, request.start_date, request.end_date)
        articles = response.get("news")

    articles_df = pd.DataFrame(articles_list, columns=['UUID', 'Date', 'Text'])
    articles_sorted = articles_df.sort_values(by='Date')
    articles_sorted.to_csv(f"/data/news/{request.company_name}/{request.company_name}_sorted.csv", index=False, header=False)