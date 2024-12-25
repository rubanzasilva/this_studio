import os

import numpy as np
import xgboost as xgb

import bentoml

import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.tabular.all import *

from typing import Annotated  # Python 3.9 or above
from typing_extensions import Annotated  # Older than 3.9
import pandas as pd
from bentoml.validators import DataframeSchema
import bentoml


@bentoml.service(
    resources={"cpu": "8"},
    traffic={"timeout": 10},
)

class MentalHealthClassifier:
    #retrieve the latest version of the model from the BentoML model store
    bento_model = bentoml.models.get("mental_health_v1:latest")

    def __init__(self):
        self.model = bentoml.xgboost.load_model(self.bento_model)

       
    @bentoml.api
    def predict( self,
        input: Annotated[pd.DataFrame, DataframeSchema(orient="records", columns=[{
    "id": 140700,"Name": "Shivam","Gender": "Male","Age": 53,
    "City":"Visakhapatnam","Working Professional or Student": "Working Professional","Profession": "Judge", "Academic Pressure":5,
    "Work Pressure": 4,"CGPA": 6.84,"Study Satisfaction": 1,"Job Satisfaction": 3,
    "Sleep Duration": "Less than 5 hours","Dietary Habits": "Moderate","Degree": "B.Ed","Have you ever had suicidal thoughts ?": "No",
    "Work/Study Hours": 6,"Financial Stress": 4,"Family History of Mental Illness": "Yes"
}
])]
    ) -> int:

        return self.model.predict(input)

        #return self.predict(data)