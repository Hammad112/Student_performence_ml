import os,sys
from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import (
RandomForestRegressor,
AdaBoostRegressor,
GradientBoostingRegressor,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_object,load_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("Artifacts/model","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()
            }
            params={
                "Decision Tree":{
                    "criterion":["poisson"],
                    "max_depth":[5,20],
       
                },
                "Random Forest":{
                    "criterion":["squared_error"],
                    "max_depth":[5,10,20],

                },
                "Gradient Boosting":{
                    "learning_rate":[0.1,0.05],
                    "subsample":[0.6],
                    "n_estimators":[1000],

                },
                "Linear Regression":{},
                "XGBRegressor":{
                    "learning_rate":[0.1,0.01,0.05],
                    "n_estimators":[10],
                    "max_depth":[5,10,20],

 
                },  
                "CatBoosting Regressor":{
                    "depth":[6,8,10],


                },
                "AdaBoost Regressor":{
                    "learning_rate":[0.1,0.01,0.05],
                    "n_estimators":[10],
                            
                }
                }
            ## Report of Model with r2 score
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models=models,params=params)
            ## To get best model
            best_model_score=max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            ## To get best model
            best_model=models[best_model_name]
            

            if best_model_score<0.6:
                raise Exception("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2=r2_score(y_test,predicted)
            return r2       

        except Exception as e:
            raise CustomException(e,sys)
