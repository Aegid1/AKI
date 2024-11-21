from fastapi import APIRouter, Depends
from services.ModelService import ModelService

router = APIRouter()

@router.post("/model/test/normalization/{experiment}/{model_path}")
def test_model_output_normalization(experiment: str, model_path:str, model_service: ModelService = Depends()):
    model_service.test_model_prediction_normalization(experiment, model_path)


@router.post("/model/test/{experiment}/{model_path}")
def test_model_output_no_normalization(experiment: str, model_path:str, model_service: ModelService = Depends()):
    model_service.test_model_prediction_no_normalization(experiment, model_path)