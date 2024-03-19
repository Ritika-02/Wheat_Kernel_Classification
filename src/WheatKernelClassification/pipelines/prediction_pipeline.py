import os
import sys
import numpy as np
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
        
class CustomData:
    def __init__(self,
                 area : float,
                 perimeter : float,
                 compactness : float,
                 length_of_kernel : float,
                 width_of_kernel : float,
                 asymmetry_coefficient : float,
                 length_of_kernel_groove : float
                 ):
        
        self.area = area
        self.perimeter = perimeter
        self.compactness = compactness
        self.length_of_kernel = length_of_kernel
        self.width_of_kernel = width_of_kernel
        self.asymmetry_coefficient = asymmetry_coefficient
        self.length_of_kernel_groove = length_of_kernel_groove


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'area' : [self.area],
                'perimeter' : [self.perimeter],
                'compactness' : [self.compactness],
                'length_of_kernel' : [self.length_of_kernel],
                'width_of_kernel' : [self.width_of_kernel],
                'asymmetry_coefficient': [self.asymmetry_coefficient],
                'length_of_kernel_groove' : [self.length_of_kernel_groove],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
            return df
        
        except Exception as e:
            logging.info('Exception Occured in Prediction Pipeline')
            raise customexception(e,sys)