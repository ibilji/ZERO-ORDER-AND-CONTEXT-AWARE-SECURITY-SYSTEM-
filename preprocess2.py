import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.categorical_features = ['device_type', 'location', 'resource_requested', 'time_of_day']
        self.composite_features = ['location_resource', 'device_resource']
        self.freq_features = ['device_count', 'resource_count', 'location_count']
        self.numeric_features = ['user_risk', 'device_risk', 'location_risk', 'resource_risk']
        self.label_encoders = {}
        self.freq_maps = {}
        self.risk_maps = {}
        self.is_fitted = False

    def fit(self, data: pd.DataFrame):
        df = data.copy()

        # Composite features
        df['location_resource'] = df['location'] + "_" + df['resource_requested']
        df['device_resource'] = df['device_type'] + "_" + df['resource_requested']

        # Frequency maps
        self.freq_maps['device_count'] = df['device_type'].value_counts().to_dict()
        self.freq_maps['resource_count'] = df['resource_requested'].value_counts().to_dict()
        self.freq_maps['location_count'] = df['location'].value_counts().to_dict()

        # Doing label encoding 
        for feature in self.categorical_features + self.composite_features:
            le = LabelEncoder()
            le.fit(df[feature].astype(str).tolist() + ["unknown"])
            self.label_encoders[feature] = le

        # Risk maps 
        self.risk_maps['user_risk'] = (df.groupby('user_id')['access_granted']
                                        .apply(lambda x: 1 - x.mean()).to_dict())
        self.risk_maps['device_risk'] = (df.groupby('device_type')['access_granted']
                                        .apply(lambda x: 1 - x.mean()).to_dict())
        self.risk_maps['location_risk'] = (df.groupby('location')['access_granted']
                                          .apply(lambda x: 1 - x.mean()).to_dict())
        self.risk_maps['resource_risk'] = (df.groupby('resource_requested')['access_granted']
                                          .apply(lambda x: 1 - x.mean()).to_dict())

        self.is_fitted = True

    def transform(self, data: pd.DataFrame):
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        df = data.copy()

        
        df['location_resource'] = df['location'] + "_" + df['resource_requested']
        df['device_resource'] = df['device_type'] + "_" + df['resource_requested']

        
        df['device_count'] = df['device_type'].map(self.freq_maps['device_count']).fillna(0)
        df['resource_count'] = df['resource_requested'].map(self.freq_maps['resource_count']).fillna(0)
        df['location_count'] = df['location'].map(self.freq_maps['location_count']).fillna(0)

        
        for feature in self.categorical_features + self.composite_features:
            le = self.label_encoders[feature]
            df[feature] = df[feature].astype(str).apply(lambda x: x if x in le.classes_ else "unknown")
            df[feature] = le.transform(df[feature])

        
        df['user_risk'] = df['user_id'].map(self.risk_maps['user_risk']).fillna(0)
        df['device_risk'] = df['device_type'].map(self.risk_maps['device_risk']).fillna(0)
        df['location_risk'] = df['location'].map(self.risk_maps['location_risk']).fillna(0)
        df['resource_risk'] = df['resource_requested'].map(self.risk_maps['resource_risk']).fillna(0)

        feature_cols = self.categorical_features + self.composite_features + self.freq_features + self.numeric_features
        return df[feature_cols].values

def preprocess_data(df: pd.DataFrame, preprocessor=None, fit=True):
    df = df.copy()
    df['time_of_access'] = pd.to_datetime(df['time_of_access'])
    df['time_of_day'] = df['time_of_access'].dt.hour
    y = df['access_granted'].values

    if preprocessor is None:
        preprocessor = DataPreprocessor()

    if fit:
        preprocessor.fit(df)
    X = preprocessor.transform(df)

    feature_names = preprocessor.categorical_features + preprocessor.composite_features + preprocessor.freq_features + preprocessor.numeric_features
    return X, y, preprocessor, feature_names
