import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.WheatKernelClassification.logger import logging
from src.WheatKernelClassification.exception import customexception
from sklearn.metrics import accuracy_score, precision_score, recall_score



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e,sys)
    
def evaluate_model(X_train, y_train, X_test,y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')

            report[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }

        return report
    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occured in load_object function utils')
        raise customexception(e,sys)    
    

    
