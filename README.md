# Medical Insurance Cost Prediction 🏥

## Overview 📊
This project implements a machine learning model to predict medical insurance costs based on demographic and health-related factors. Using a Linear Regression approach, the model analyzes features such as age, BMI, smoking status, and region to estimate individual insurance charges.

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange.svg)](https://scikit-learn.org/stable/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.3-green.svg)](https://pandas.pydata.org/)
[![Conda](https://img.shields.io/badge/conda-4.12.0-lightgreen.svg)](https://docs.conda.io/projects/conda/en/latest/index.html)

## Project Structure 🗂️
```
medical-insurance-prediction/
│
├── artifacts/               # Trained models and preprocessors
├── notebook/               # Jupyter notebooks for analysis
│   └── data/              # Dataset directory
├── src/                   # Source code
│   ├── components/        # Core components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/          # Prediction pipeline
│   ├── utils.py           # Utility functions
│   ├── logger.py          # Logging configuration
│   └── exception.py       # Custom exception handling
├── templates/             # Flask templates for web interface
├── setup.py               # Package setup configuration
├── requirements.txt       # Project dependencies
└── README.md
```

## Features 🌟
- Comprehensive data preprocessing and feature engineering
- Interactive web interface for predictions using Flask
- Detailed exploratory data analysis (EDA)
- Model performance evaluation with multiple metrics
- Feature importance analysis
- Custom exception handling and logging
- End-to-end ML pipeline for data ingestion, transformation, and training

## Key Features Analyzed 📈
- Age
- Sex
- BMI (Body Mass Index)
- Number of Children
- Smoking Status
- Region
- Derived Features (age_bmi, smoker_bmi)

## Model Performance 📊
The Linear Regression model achieves:
- R² Score (Test): 0.8171 (81.71% variance explained)
- RMSE (Test): 0.4055
- MAE (Test): 0.2531

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/austinLorenzMccoy/medical-insurance-prediction.git
cd medical-insurance-prediction
```

2. Create and activate a Conda environment:
```bash
# Create a new conda environment
conda create -n insurance-pred python=3.8 -y

# Activate the environment
conda activate insurance-pred
```

3. Install required packages:
```bash
# Install main dependencies using conda
conda install pandas=2.2.3 numpy=1.26.4 scikit-learn=1.5.2 flask=3.1.0 scipy=1.14.1 seaborn=0.13.2 matplotlib=3.9.2 -c conda-forge

# Install additional packages using pip
pip install catboost==1.2.7 xgboost==2.1.2 dill
```

### Dependencies
The project requires the following main packages:
- pandas==2.2.3
- numpy==1.26.4
- seaborn==0.13.2
- matplotlib==3.9.2
- scikit-learn==1.5.2
- catboost==1.2.7
- xgboost==2.1.2
- Flask==3.1.0
- scipy==1.14.1
- dill

### Package Configuration
The project uses `setup.py` for package configuration:
```python
from setuptools import find_packages, setup
from typing import List

# Package metadata
name="mlproject"
version="0.0.1"
author="Austin"
author_email="chibuezeaugustine23@gmail.com"
```

## Usage 💻

### Training the Model
```python
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Initialize and run data ingestion
ingestion = DataIngestion()
train_data, test_data = ingestion.initiate_data_ingestion()

# Transform data
transformation = DataTransformation()
train_arr, test_arr, _ = transformation.initiate_data_transformation(train_data, test_data)

# Train model
trainer = ModelTrainer()
metrics, r2_score = trainer.initiate_model_trainer(train_arr, test_arr)
```

### Making Predictions
```python
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Create sample data
data = CustomData(
    age=35,
    sex='male',
    bmi=27.5,
    children=2,
    smoker='no',
    region='southwest'
)

# Convert to DataFrame
df = data.get_data_as_data_frame()

# Make prediction
pipeline = PredictPipeline()
prediction = pipeline.predict(df)
```

### Running the Web Application
```bash
# Start the Flask application
python app.py
```
Then open your browser and navigate to `http://localhost:5000`

## Dataset Details 📖
The dataset includes information about medical insurance beneficiaries:
- 1338 samples
- 7 features
- Target variable: Insurance charges (USD)
- Source: Kaggle Medical Cost Personal Dataset

## Key Insights 🔍
1. Smoking status has the strongest impact on insurance charges
2. Age and BMI show significant positive correlations with charges
3. Regional differences have minimal impact on costs
4. Interaction between smoking status and BMI is crucial

## Error Handling and Logging 🔧
The project includes custom error handling and logging:
- Custom exception class for better error tracking
- Detailed logging configuration for debugging
- Error message details including file name and line number

## Contributing 🤝
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## Contact 📧
Austin Lorenz McCoy
- GitHub: [austinLorenzMccoy](https://github.com/austinLorenzMccoy)
- Email: chibuezeaugustine23@gmail.com

## Acknowledgments 🙏
- Dataset source: [Kaggle Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance/data)
