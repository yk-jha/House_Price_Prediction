import sys
from dataclasses import dataclass
import scipy.sparse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import customexception
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
        
    def get_data_transformer_object(self):
        '''This Function is responsible for data Transformation'''
        try:
            numerical_columns = ['total_sqft', 'bath', 'bhk', 'price_per_sqft']
            categorical_columns = ['location', 'size']
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )  
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise customexception(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Ensure 'bhk' is numeric
            train_df["bhk"] = pd.to_numeric(train_df["bhk"], errors="coerce")
            test_df["bhk"] = pd.to_numeric(test_df["bhk"], errors="coerce")

            # Fill NaN values in 'bhk' with mode (most common value)
            bhk_mode = train_df["bhk"].mode()[0]
            train_df["bhk"] = train_df["bhk"].fillna(bhk_mode)
            test_df["bhk"] = test_df["bhk"].fillna(bhk_mode)

            logging.info("Ensured 'bhk' is numeric and handled missing values.")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "price"
            numerical_columns = ['total_sqft', 'bath', 'bhk', 'price_per_sqft']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            if scipy.sparse.issparse(input_feature_train_arr):
                input_feature_train_arr = input_feature_train_arr.toarray()

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            if scipy.sparse.issparse(input_feature_test_arr):
                input_feature_test_arr = input_feature_test_arr.toarray()
            
            if isinstance(target_feature_train_df, pd.Series):
                target_feature_train_df = target_feature_train_df.values
            if isinstance(target_feature_test_df, pd.Series):
                target_feature_test_df = target_feature_test_df.values

        # Ensure target is 1D
            target_feature_train_df = np.ravel(target_feature_train_df)
            target_feature_test_df = np.ravel(target_feature_test_df)

        # Debug shapes
            logging.info(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")
            logging.info(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
            logging.info(f"Shape of target_feature_test_df: {target_feature_test_df.shape}")
            print(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            print(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")
            # Ensure target arrays are 1D before concatenation
            # Debugging shapes before concatenation
            print("Shape of input_feature_train_arr just before concatenation:", input_feature_train_arr.shape)
            print("Shape of target_feature_train_df just before concatenation:", target_feature_train_df.shape)
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise customexception(e, sys)
