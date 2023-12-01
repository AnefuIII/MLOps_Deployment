import os
import sys
from dataclasses import dataclass

#from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
#import xgboost as xg
#from xgboost import XGBRegressor

from src.utils import evaluate_models
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                #'XGB Regressor': xg.XGBRegressor(),
               # 'CatBoosting Regressor': CatBoostRegressor(verbosr = False),
                'AdaBoost Regressor': AdaBoostRegressor(),
            }

            params = {
                'Random Forest': {
                    'n_estimators': [8,16,32,64,128,256],
                    #'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'splitters': ['best', 'random'],
                    #'max_features': ['sqrt','log2'],
                },

                'Decision Tree': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'splitters': ['best', 'random'],
                    #'max_features': ['sqrt','log2'],
                },

                'Gradient Boosting': {
                    'learning_rate': [.1, .01, .5, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8,16,32,64,128,256],
                    #'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    #'splitters': ['best', 'random'],
                    #'max_features': ['sqrt','log2'],                   
                }, 

                'Linear Regression': {},

                'K-Neighbors Regressor': {
                    'n_neighbors': [5,7,9,11],
                    #'weights': ['uniform','distance'],
                    #'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                }, 

                # 'XGB Regressor': {
                #     'learning_rate': [.1, .01, .5, .001],
                #     'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                # },

                # 'CatBoosting Regressor': {
                #     'learning_rate': [.1, .01, .5, .001],
                #     'depth': [6,8,10],
                #     'iterations': [30, 50, 100],
                # },

                'AdaBoost Regressor': {
                    'learning_rate': [.1, .01, .5, .001],
                    'n_estimators': [8,16,32,64,128,256],
                    #'loss': ['linear', 'square', 'exponential'],

                },

            }

            model_report: dict = evaluate_models(
                X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test,
                models = models,
                param=params
            )

            # to get best model model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No Best model found')
            
            logging.info('Best found model on both training and testing dataset')

            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj = best_model
            ) 

            predicted = best_model.predict(X_test)

            r2_scr = r2_score(y_test, predicted)

            return r2_scr

        except Exception as e:
            raise CustomException(e, sys)