import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class ModelTrainer:
    def __init__(self):
        self.trained_model_file_path = os.path.join("artifacts", "model.pkl")

    def initiate_model_trainer(self, train_df, test_df):
        try:
            logging.info("Model Training Started")
            
            # Validate input
            if train_df is None or test_df is None:
                raise ValueError("Training or testing dataframe is None")

            # Log DataFrame details
            logging.info(f"Train DataFrame Columns: {train_df.columns}")
            logging.info(f"Train DataFrame Shape: {train_df.shape}")
            logging.info(f"Test DataFrame Columns: {test_df.columns}")
            logging.info(f"Test DataFrame Shape: {test_df.shape}")

            # Separate features and target
            target_column = 'charges_log'
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Create and train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate metrics
            metrics = {
                'R2 Score (Train)': r2_score(y_train, y_pred_train),
                'R2 Score (Test)': r2_score(y_test, y_pred_test),
                'RMSE (Train)': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'RMSE (Test)': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'MAE (Train)': mean_absolute_error(y_train, y_pred_train),
                'MAE (Test)': mean_absolute_error(y_test, y_pred_test)
            }

            # Log metrics
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value:.4f}")

            # Save model
            save_object(
                file_path=self.trained_model_file_path,
                obj=model
            )

            return metrics, metrics['R2 Score (Test)']

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)