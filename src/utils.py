import os
import sys
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a Python object to the specified file path using pickle.

    Args:
        file_path (str): Path to save the object.
        obj (object): Object to save.

    Raises:
        CustomException: If an error occurs during saving.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a Python object from the specified file path using pickle.

    Args:
        file_path (str): Path to load the object from.

    Returns:
        object: The loaded object.

    Raises:
        CustomException: If an error occurs during loading.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_pipeline(model, X_train, y_train, X_test, y_test):
    """
    Evaluates a single pipeline model and returns performance metrics.

    Args:
        model (Pipeline): Trained pipeline model.
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target set.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Test target set.

    Returns:
        dict: Dictionary containing evaluation metrics for training and test data.
    """
    try:
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        metrics = {
            "R2 Score (Train)": r2_score(y_train, y_train_pred),
            "R2 Score (Test)": r2_score(y_test, y_test_pred),
            "RMSE (Train)": mean_squared_error(y_train, y_train_pred, squared=False),
            "RMSE (Test)": mean_squared_error(y_test, y_test_pred, squared=False),
            "MAE (Train)": mean_absolute_error(y_train, y_train_pred),
            "MAE (Test)": mean_absolute_error(y_test, y_test_pred),
        }
        return metrics

    except Exception as e:
        raise CustomException(e, sys)


def create_feature_pipeline():
    """
    Creates a preprocessing pipeline for feature engineering.

    Returns:
        Pipeline: A pipeline that preprocesses numerical and categorical features.
    """
    try:
        # Define numerical and categorical features
        numerical_features = ['age', 'bmi', 'children']  # Adjust as per your dataset
        categorical_features = ['sex', 'smoker', 'region']  # Adjust as per your dataset

        # Define transformations for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Define transformations for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine both transformers into a column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        return preprocessor
    except Exception as e:
        raise CustomException(e, sys)
