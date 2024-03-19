from src.WheatKernelClassification.components.data_ingestion import DataIngestion

from src.WheatKernelClassification.components.data_transformation import DataTransformation

from src.WheatKernelClassification.components.model_trainer import ModelTrainer


import os
import sys
from src.WheatKernelClassification.logger import logging
from src.WheatKernelClassification.exception import customexception
import pandas as pd


# Data Ingestion Pipeline
obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()

# Data Transformation Pipeline
data_transformation = DataTransformation()
train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path,test_data_path)

# Model Trainer Pipeline
model_trainer_obj = ModelTrainer()
model_trainer_obj.initiate_model_training(train_arr,test_arr)