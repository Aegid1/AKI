import re
import time
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
   articles = news_api_service.get_articles(request.company_name, request.up_to_page_number, request.start_date,
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
   articles = news_service.get_articles(request.company_name, request.up_to_page_number, request.start_date,
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


@router.post("/articles/news/all")
def store_all_relevant_articles_from_news_api(request: NewsApiRequest, news_api_service: NewsService = Depends()):
   """
        Fetch and store all relevant articles from a news API based on the given request.

        This endpoint retrieves news articles within a specified date range and filters
        them for relevance using predefined criteria. The relevant articles are stored
        in a CSV file, organized by month and year.

        Parameters:
        - request (NewsApiRequest): The request containing parameters such as the start
          and end dates, company name (if applicable), and the maximum number of pages
          to fetch from the news API.
        - news_api_service (NewsService): A service for interacting with the news API
          and performing operations like date conversion and article relevance checking.
          Provided via dependency injection.

        Processing Steps:
        1. The date range is divided into monthly intervals.
        2. For each interval, fetch articles using the API.
        3. Filter articles to keep only the relevant ones using a relevance threshold (3.3)
         -> this threshold is a score calculated with embeddings.
        4. Handle placeholder dates from the API and convert them to actual dates.
        5. Remove duplicates and sort the articles by date.
        6. Save the filtered articles into a CSV file in a structured format.

        Returns:
        None: The function performs file I/O but does not return a response.

        Raises:
        - Various exceptions can be encountered during date parsing or file operations,
          which are handled gracefully within the function to ensure smooth execution.
   """
   dates = news_api_service.create_monthly_intervals(request.start_date, request.end_date)
   print(dates)
   for date_tuple in dates:

       start_time = time.time()

       requested_pages = request.up_to_page_number
       articles_list = []
       days = news_api_service.get_open_days(date_tuple[0], date_tuple[1])
       _, month, year = date_tuple[0].split("/")

       if request.is_company_given:
           topic = f"{request.company_name} Unternehmen" #in case of company news, if this is added more precise results are given
       else:
           topic = request.company_name

       for i in range(len(days)-1):
           print(f"DAY: {days[i]} to {days[i+1]}")
           current_page = 1
           response = news_api_service.get_articles(topic, current_page, days[i], days[i+1])

           while response.get("count") != 0:
               articles = response.get("news")
               print("PAGE: " + str(current_page))
               if articles:
                   for article in articles:
                       content = article.get("text")
                       content = re.sub(r'\s+', ' ', content).strip() #news texts contain unnecessary newlines

                       #threshold of 3.3 is used, this was the result of several tests leading to more relevant articles
                       article_is_relevant = news_api_service.check_if_article_is_relevant(topic, content, 3.3, request.company_name) #sometimes articles are not relevant -> sort those out
                       if not article_is_relevant: continue

                       date = article.get("date")

                       try:
                           if "08:00:00" in date or "07:00:00" in date: #the used news api sometimes doesn't provide the actual date and uses 08:00:00 or 07:00:00 as a placeholder
                               date = news_api_service.get_actual_date_of_article(article.get("url"), date)

                           converted_date = pd.to_datetime(date, format='%a, %d %b %Y %H:%M:%S %Z')

                       except Exception: continue

                       articles_list.append((str(uuid.uuid4()), converted_date, content))

               current_page += 1
               if current_page > requested_pages: break
               response = news_api_service.get_articles(topic, current_page, days[i], days[i+1])

           i+=2

       articles_df = pd.DataFrame(articles_list, columns=['UUID', 'Date', 'Text'])
       articles_df = articles_df.drop_duplicates(subset='Text')
       articles_sorted = articles_df.sort_values(by='Date')
       articles_sorted.to_csv(f"data/news/{request.company_name}/{request.company_name}_sorted_{month}_{year}.csv", index=False, header=False)

       end_time = time.time()
       duration = end_time - start_time
       print(f"Die Funktion hat {duration:.2f} Sekunden gebraucht.")


@router.post("/articles/merge/news/{company_name}/{month}/{year}")
def add_economy_news_to_company_news(company_name:str, month:str, year:str, news_api_service: NewsService = Depends()):
   news_api_service.merge_articles(company_name, month, year)


# @router.post("/articles/news/topics/{company_name}")
# def store_relevant_related_topics(company_name: str, news_service: NewsService = Depends()):
#     news_service.store_relevant_related_topics(company_name)
