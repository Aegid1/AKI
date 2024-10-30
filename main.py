import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.openai_batch_api import router as openai_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(openai_router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="localhost", port=4000)