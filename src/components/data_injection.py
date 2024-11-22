import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class DataIngestion:
    def __init__(self):
        self.train_data_path = os.path.join('artifacts', "train.csv")
        self.test_data_path = os.path.join('artifacts', "test.csv")
        self.raw_data_path = os.path.join('artifacts', "data.csv")

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")
            
            # Load dataset
            df = pd.read_csv('notebook/data/medical_insurance.csv')
            logging.info('Dataset loaded successfully')

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully")

            # Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test data
            train_set.to_csv(self.train_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")
            return self.train_data_path, self.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Data ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Data transformation
    data_transformation = DataTransformation()
    train_df, test_df, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Model trainer
    model_trainer = ModelTrainer()
    metrics, r2_score = model_trainer.initiate_model_trainer(train_df, test_df)
    print("Model Training Metrics:", metrics)