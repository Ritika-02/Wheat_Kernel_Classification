import os
import sys
import pandas as pd
from src.WheatKernelClassification.exception import customexception
from src.WheatKernelClassification.logger import logging
from src.WheatKernelClassification.utils.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
        
        except Exception as e:
            raise customexception(e,sys)