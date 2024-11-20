from fastapi import APIRouter, Depends
from services.ModelService import ModelService

router = APIRouter()

@router.post("/model/test/{experiment}/{model_path}")
def test_model_output(experiment: str, model_path:str, model_service: ModelService = Depends()):
    model_service.test_model_prediction(experiment, model_path)