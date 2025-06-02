import pandas as pd
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_feature_columns(jsonPath):
    """ Loads pre-defined numeric and categorical features from JSON. """
    with open(jsonPath, 'r') as f:
        features = json.load(f)

    return features['numerical'], features['categorical']

def build_preprocessing_pipeline_from_json(json_path):
    """Builds a preprocessing pipeline using feature definitions from a JSON file."""    
    
    numeric_features, categorical_features = load_feature_columns(json_path)

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    pre_processor = ColumnTransformer(transformers = [
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return pre_processor