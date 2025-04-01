import sys
import os
import pandas as pd
from src.exception import customexception
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def Predict(self , features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise customexception(e,sys)
    
    
class CustomData:
    def __init__(self,
        location: str,
        size: str,
        total_sqft: int,
        price_per_sqft: int,
        bath: int,
        bhk: int):
        self.location = location

        self.size = size

        self.total_sqft = total_sqft

        self.price_per_sqft = price_per_sqft

        self.bhk = bhk

        self.bath = bath
        
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "location": [self.location],
                "size": [self.size],
                "total_sqft": [self.total_sqftn],
                "bhk": [self.bhk],
                "bath": [self.bath],
                "price_per_sqft": [self.price_per_sqft],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise customexception(e, sys)