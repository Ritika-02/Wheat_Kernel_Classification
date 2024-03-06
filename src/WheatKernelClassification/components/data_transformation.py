import os
import sys
import numpy as np
import pandas as pd 

from dataclasses import dataclass
from src.WheatKernelClassification.logger import logging
from src.WheatKernelClassification.exception import customexception
from src.WheatKernelClassification.utils.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info('Data Transformation Initiated')

            numerical_cols = ['area', 'perimeter', 'compactness', 'length_of_kernel', 'width_of_kernel', 'asymmetry_coefficient', 'length_of_kernel_groove']

            logging.info('Pipeline Initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', MinMaxScaler())
                ]
            )


            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols)
                ],
                remainder='passthrough'
            )

            return preprocessor

        except Exception as e:
            logging.info('Exception occurred in initiate_data_transformation')
            raise customexception(e, sys)
        
    def initialize_data_transformation(self,train_path,test_path):
        
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data complete')
            logging.info(f'Train Dataframe Head : \n {train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n {test_df.head().to_string()}')  

            for df in [train_df, test_df]:
                df = df.replace({'variety': {
                    3 : 'Canadian',
                    2 : 'Rosa',
                    1 : 'Kama'
                }})

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'variety'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing datasets')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info('preprocessing pickle file saved')

            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info('Exception occured in initiate_datatransformation')

            raise customexception(e,sys)

