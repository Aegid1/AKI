import os

from fastapi import APIRouter, Depends
from services.ModelService import ModelService

router = APIRouter()

@router.post("/model/test/normalization/experiment0/{model_path}")
def test_model_output_normalization(model_path:str, model_service: ModelService = Depends()):
    model_service.test_model_prediction_normalization_experiment0("experiment0", model_path)


@router.post("/model/test/experiment0/{model_path}")
def test_model_output_no_normalization(model_path:str, model_service: ModelService = Depends()):
    model_service.test_model_prediction_no_normalization_experiment0("experiment0", model_path)

@router.post("/model/test/experiment3/{model_path}")
def test_model_output_no_normalization(model_path:str, model_service: ModelService = Depends()):
    model_service.test_model_prediction_normalization_experiment3_new_scaler("experiment3", model_path)

@router.post("/model/test/experiment2/{model_path}")
def test_model_output_no_normalization(model_path:str, model_service: ModelService = Depends()):
    model_service.test_model_prediction_experiment2_batch_normalization(model_path)

@router.post("/model/moving_average")
def test_moving_average_model(model_service: ModelService = Depends()):
    model_service.test_moving_average_model()


