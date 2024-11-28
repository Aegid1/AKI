import os

from fastapi import APIRouter, Depends
from services.ModelService import ModelService

router = APIRouter()

@router.post("/model/test/normalization/experiment0/{model_path}")
def test_model_output_normalization(model_path:str, model_service: ModelService = Depends()):
    model_service.test_model_prediction_normalization_experiment0("experiment0", model_path)


@router.post("/model/test/experiment0/{model_path}")
def test_model_output_no_normalization(model_path:str, model_service: ModelService = Depends()):
    model_service.test_model_prediction_no_normalization("experiment0", model_path)

@router.post("/model/test/experiment3/{model_path}")
def test_model_output_no_normalization(model_path:str, model_service: ModelService = Depends()):
    model_service.test_model_prediction_normalization_experiment3_new_scaler("experiment3", model_path)

@router.post("/model/attention_weights/{model_name}/{experiment}")
def plot_attention_weights_of_trained_model(model_name:str, experiment:str, model_service: ModelService = Depends()):
    model_path = os.path.join("experiments", experiment, model_name)
    model_service.save_model_attention_weights(model_path, experiment)
