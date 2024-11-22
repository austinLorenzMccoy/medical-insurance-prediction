import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

    def preprocess_data(self, df):
        try:
            df_processed = df.copy()

            # Log transform the target variable (charges)
            df_processed['charges_log'] = np.log(df_processed['charges'])

            # Create interaction terms
            df_processed['age_bmi'] = df_processed['age'] * df_processed['bmi']
            df_processed['smoker_bmi'] = (
                (df_processed['smoker'] == 'yes').astype(int) * df_processed['bmi']
            )

            return df_processed

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        try:
            # Define numeric and categorical columns
            numeric_features = ['age', 'bmi', 'children', 'age_bmi', 'smoker_bmi']
            categorical_features = ['sex', 'smoker', 'region']

            # Create preprocessing pipelines
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(drop='first', sparse_output=False))
            ])

            # Combine transformers
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Preprocessing train and test datasets")
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            # Get preprocessor
            preprocessor = self.get_data_transformer_object()

            # Separate features and target
            target_column = 'charges_log'
            feature_columns = [col for col in train_df.columns if col != target_column]

            # Fit and transform data
            X_train = train_df[feature_columns]
            y_train = train_df[target_column]
            X_test = test_df[feature_columns]
            y_test = test_df[target_column]

            # Transform data
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Reconstruct DataFrames
            feature_names = preprocessor.get_feature_names_out()
            train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
            test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)

            train_transformed[target_column] = y_train.values
            test_transformed[target_column] = y_test.values

            # Save preprocessor
            save_object(
                file_path=self.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return train_transformed, test_transformed, self.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)