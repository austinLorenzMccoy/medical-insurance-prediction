import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            # Ensure features align with preprocessing expectations
            features_scaled = preprocessor.transform(features)
            preds = model.predict(features_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 age: int, 
                 sex: str, 
                 bmi: float, 
                 children: int, 
                 smoker: str, 
                 region: str):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region

    def get_data_as_data_frame(self):
        try:
            # Include derived columns to match the model's expected features
            age_bmi = self.age * self.bmi  # Example derived feature
            smoker_bmi = 1 if self.smoker.lower() == 'yes' else 0

            # Create a dictionary for input data, now including the derived features
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "bmi": [self.bmi],
                "children": [self.children],
                "age_bmi": [age_bmi],  # Added derived feature
                "smoker_bmi": [smoker_bmi],  # Added derived feature
                "smoker": [self.smoker],
                "region": [self.region],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

