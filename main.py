import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.openai_batch_api import router as openai_router
from api.news_api import router as news_router
from api.twitter_api import router as twitter_router
from api.stocks_api import router as stocks_router
from api.macro_factors_api import router as macro_factors_router
from api.model_api import router as model_router
from api.technical_indicators_api import router as technical_indicator_router
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(stocks_router, prefix="/api/v1")
app.include_router(openai_router, prefix="/api/v1")
app.include_router(news_router, prefix="/api/v1")
app.include_router(twitter_router, prefix="/api/v1")
app.include_router(macro_factors_router, prefix="/api/v1")
app.include_router(model_router, prefix="/api/v1")
app.include_router(technical_indicator_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=4000)