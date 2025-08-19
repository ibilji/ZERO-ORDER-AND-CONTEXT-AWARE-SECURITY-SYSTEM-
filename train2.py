import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from preprocess2 import DataPreprocessor

os.makedirs("models", exist_ok=True)


def preprocess_with_interactions(df, preprocessor=None, fit=True):
    df = df.copy()
    df['time_of_access'] = pd.to_datetime(df['time_of_access'])
    df['time_of_day'] = df['time_of_access'].dt.hour

    if preprocessor is None:
        preprocessor = DataPreprocessor()

    
    df_copy = df.copy()
    for col in ['user_id', 'device_type', 'location', 'resource_requested']:
        denials = df_copy.groupby(col)['access_granted'].apply(lambda x: (x==0).sum())
        total = df_copy.groupby(col)['access_granted'].count()
        risk_col = f'{col.split("_")[0]}_risk'
        df_copy[risk_col] = df_copy[col].map((denials + 1) / (total + 2))

    
    df_copy['user_device_risk'] = df_copy['user_risk'] * df_copy['device_risk']
    df_copy['user_location_risk'] = df_copy['user_risk'] * df_copy['location_risk']
    df_copy['device_location_risk'] = df_copy['device_risk'] * df_copy['location_risk']

    if fit:
        preprocessor.fit(df_copy)

    X = preprocessor.transform(df_copy)

    
    interaction_features = df_copy[['user_device_risk', 'user_location_risk', 'device_location_risk']].values
    X = np.hstack([X, interaction_features])

    
    X = np.atleast_2d(X)
    return X, preprocessor

# Using this function for threshold tunning 
def find_best_threshold(y_true, y_proba):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_thresh = 0.5
    best_f1 = 0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=0)  
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh


def train_model(data_path='access_logs.csv'):
    print("Loading data...")
    df = pd.read_csv(data_path)

    # Spliting train test and validation 
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['access_granted'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['access_granted'])

    # Preprocessing
    print("Preprocessing training data..")
    X_train, preprocessor = preprocess_with_interactions(train_df, fit=True)
    y_train = train_df['access_granted'].values

    print("Preprocessing validation data..")
    X_val, _ = preprocess_with_interactions(val_df, preprocessor=preprocessor, fit=False)
    y_val = val_df['access_granted'].values

    print(" Preprocessing test data..")
    X_test, _ = preprocess_with_interactions(test_df, preprocessor=preprocessor, fit=False)
    y_test = test_df['access_granted'].values

    # TF-IDF for vectorization 
    tfidf = TfidfVectorizer(max_features=100, ngram_range=(1,2))
    X_text_train = tfidf.fit_transform(train_df['intent_prompt'].astype(str)).toarray()
    X_text_val = tfidf.transform(val_df['intent_prompt'].astype(str)).toarray()
    X_text_test = tfidf.transform(test_df['intent_prompt'].astype(str)).toarray()

    
    X_train = np.hstack([X_train, X_text_train])
    X_val = np.hstack([X_val, X_text_val])
    X_test = np.hstack([X_test, X_text_test])

    # Using SMOTE on training data to balance it 
    print("Balancing classes with SMOTE (ratio 1.0) on training data..")
    sm = SMOTE(sampling_strategy=1.0, random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Using Random forest which gives us an accuracy of 97 percent 
    param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    print(" Training Random Forest with hyperparameter tuning...")
    grid.fit(X_train_res, y_train_res)
    clf = grid.best_estimator_
    print(f"Best Parameters: {grid.best_params_}")

    
    y_val_proba = clf.predict_proba(X_val)[:, 1]

    # Finding optimal threshold using validation set
    best_thresh = find_best_threshold(y_val, y_val_proba)
    print(f"Optimal threshold for class 0 (from validation): {best_thresh:.2f}")

    # Evaluating on the test set using the tuned threshold
    y_test_proba = clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_thresh).astype(int)

    print("Test Set Evaluation:")
    print(classification_report(y_test, y_test_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")

    #saving data into pickle files 
    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    print("Training complete. Model + Preprocessor + TF-IDF saved in 'models/'")

if __name__ == "__main__":
    train_model("access_logs.csv")
