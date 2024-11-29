import os

from fastapi import APIRouter, Depends
from services.ModelService import ModelService

router = APIRouter()

@router.post("/model/test/normalization/experiment0/{model_path}")
def test_model_output_normalization(model_path:str, model_service: ModelService = Depends()):
    """
        Test model output normalization for Experiment 0.

        This endpoint tests the normalization process applied to model predictions
        for Experiment 0. The results are evaluated using the "experiment0_normalization" configuration.

        Parameters:
        - model_path (str): Path to the model to be tested.
        - model_service (ModelService): Service for handling model-related operations,
          injected as a dependency.
        """
    model_service.test_model_prediction_normalization_experiment0("experiment0_normalization", model_path)


@router.post("/model/test/experiment0/{model_path}")
def test_model_output_no_normalization(model_path:str, model_service: ModelService = Depends()):
    """
    Test model output without normalization for Experiment 0.

    This endpoint evaluates model predictions for Experiment 0 without applying
    any normalization. The results are processed using the "experiment0" configuration.

    Parameters:
    - model_path (str): Path to the model to be tested.
    - model_service (ModelService): Service for handling model-related operations,
      injected as a dependency.
    """
    model_service.test_model_prediction_no_normalization_experiment0("experiment0", model_path)

@router.post("/model/test/experiment3/{model_path}")
def test_model_output_no_normalization(model_path:str, model_service: ModelService = Depends()):
    """
        Test model output normalization with a new scaler for Experiment 3.

        This endpoint evaluates model predictions for Experiment 3 using a new scaler
        during the normalization process. So scalers of the training are not used.
        The results are processed using the "experiment3" configuration.

        Parameters:
        - model_path (str): Path to the model to be tested.
        - model_service (ModelService): Service for handling model-related operations,
          injected as a dependency.
        """
    model_service.test_model_prediction_normalization_experiment3_new_scaler("experiment3", model_path)

@router.post("/model/test/experiment2/{model_path}")
def test_model_output_no_normalization(model_path:str, model_service: ModelService = Depends()):
    """
        Test model output with batch normalization for Experiment 2.

        This endpoint evaluates model predictions for Experiment 2, applying batch
        normalization techniques during the test process.

        Parameters:
        - model_path (str): Path to the model to be tested.
        - model_service (ModelService): Service for handling model-related operations,
          injected as a dependency.
        """
    model_service.test_model_prediction_experiment2_batch_normalization(model_path)

@router.post("/model/moving_average")
def test_moving_average_model(model_service: ModelService = Depends()):
    """
        Test a simple moving average model.

        This endpoint triggers the testing of a moving average model to evaluate its
        predictive capabilities.

        Parameters:
        - model_service (ModelService): Service for handling model-related operations,
          injected as a dependency.
        """
    model_service.test_moving_average_model()


