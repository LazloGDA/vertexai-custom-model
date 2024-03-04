
import joblib
import numpy as np
import json

from google.cloud import storage
from google.cloud.aiplatform.prediction.sklearn.predictor import SklearnPredictor


class CprPredictor(SklearnPredictor):

    def __init__(self):
        return

    def load(self, artifacts_uri: str) -> None:
        """Loads the sklearn pipeline and preprocessing artifact."""

        super().load(artifacts_uri)

        # open preprocessing artifact
        with open("preprocessor.json", "rb") as f:
            self._preprocessor = json.load(f)


    def preprocess(self, prediction_input: np.ndarray) -> np.ndarray:
        """Performs preprocessing by checking if clarity feature is in abbreviated form."""

        inputs = super().preprocess(prediction_input)

        for sample in inputs:
            if sample[3] not in self._preprocessor.values():
                sample[3] = self._preprocessor[sample[3]]
        return inputs

    def postprocess(self, prediction_results: np.ndarray) -> dict:
        """Performs postprocessing by rounding predictions and converting to str."""

        return {"predictions": [f"${value}" for value in np.round(prediction_results)]}
