import os
import sys
import numpy as np
from dataclasses import dataclass
from src.WheatKernelClassification.logger import logging
from src.WheatKernelClassification.exception import customexception
from src.WheatKernelClassification.utils.utils import evaluate_model, save_object

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score, classification_report



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def tune_hyperparameters(self, model, params, X_train, y_train):
        clf = GridSearchCV(estimator=model, param_grid=params, cv=10, n_jobs=-1)
        clf.fit(X_train, y_train)

        print(f'Tuned hyperparameters: {clf.best_params_}')
        print(f'Accuracy: {clf.best_score_}')

        return clf.best_estimator_

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LogisticRegression': LogisticRegression(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                'RandomForestClassifier': RandomForestClassifier(),
                'GaussianNB': GaussianNB()
            }

            param_grids = {
                'LogisticRegression': {
                    'C': [0.001, 0.01, 0.1, 1.0, 10, 50],
                    'class_weight': ['balanced'],
                    'solver': ['liblinear']
                },
                'DecisionTreeClassifier': {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'splitter': ['best', 'random'],
                    'max_depth': list(np.arange(4, 30, 1))
                },
                'KNeighborsClassifier': {
                    'n_neighbors': list(np.arange(3, 50, 2)),
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2, 3, 4]
                },
                'RandomForestClassifier': {
                    'n_estimators': [50, 150, 500],
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2']
                },
            }

            model_reports = {}

            for model_name, model in models.items():
                param_grid = param_grids.get(model_name, {})
                tuned_model = self.tune_hyperparameters(model, param_grid, X_train, y_train)

                tuned_model.fit(X_train, y_train)
                y_pred = tuned_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                model_reports[model_name] = {
                    'model': tuned_model,
                    'accuracy': accuracy,
                    'classification_report': classification_report(y_test, y_pred)
                }

                print(f'{model_name} - Accuracy: {accuracy}')
                print(f'{model_name} - Classification Report:\n{classification_report(y_test, y_pred)}')
                print('\n====================================================================================\n')

            best_model_name = max(model_reports, key=lambda k: model_reports[k]['accuracy'])
            best_model = model_reports[best_model_name]['model']

            print(f'Best Model Found, Model Name: {best_model_name}, Accuracy: {model_reports[best_model_name]["accuracy"]}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, Accuracy: {model_reports[best_model_name]["accuracy"]}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise customexception(e, sys)