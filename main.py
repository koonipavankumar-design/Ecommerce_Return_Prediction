
# E-COMMERCE PRODUCT RETURN PREDICTION
# RANDOM FOREST MACHINE LEARNING

# STEP 1: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# STEP 2: LOAD DATASET

df = pd.read_csv("ecommerce_returns.csv")

print("\nDATASET LOADED SUCCESSFULLY")
print(df.head())

# STEP 3: EXPLORE DATASET

print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())

# STEP 4: REMOVE DUPLICATES

df = df.drop_duplicates()

# STEP 5: REMOVE ID, DATE AND LEAKAGE COLUMNS

columns_to_remove = [
    "order_id",
    "customer_id",
    "product_id",
    "order_date",
    "delivered_date",
    "request_date",
    "return_date",
    "refund_amount",
    "refund_status",
    "return_reason",
    "return_requested",
    "return_completed",
    "refund_date",
    "return_approved",
    "returned_quantity"
]

df = df.drop(
    columns=columns_to_remove,
    errors="ignore"
)

# STEP 6: SELECT TARGET COLUMN

target_column = "returned"

if target_column not in df.columns:
    raise ValueError(
        "Target column 'returned' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df.dropna(subset=[target_column])

# Convert common Yes/No target values to 1/0
target_text = df[target_column].astype(str).str.strip().str.lower()

target_mapping = {
    "yes": 1,
    "returned": 1,
    "true": 1,
    "1": 1,
    "no": 0,
    "not returned": 0,
    "false": 0,
    "0": 0
}

if set(target_text.unique()).issubset(target_mapping.keys()):
    y = target_text.map(target_mapping).astype(int)
else:
    raise ValueError(
        "Please check the returned column. "
        "Expected Yes/No or 1/0 values."
    )

X = df.drop(columns=[target_column])

print("\nInput Features:")
print(X.head())

print("\nTarget Distribution:")
print(y.value_counts())

# STEP 7: IDENTIFY NUMERICAL AND CATEGORICAL FEATURES

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical Features:", numerical_features)
print("Categorical Features:", categorical_features)

if not numerical_features and not categorical_features:
    raise ValueError("No usable input features were found.")

# STEP 8: DATA PREPROCESSING

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# STEP 9: SPLIT DATASET

if y.nunique() != 2 or y.value_counts().min() < 2:
    raise ValueError(
        "You need at least two examples of each target class."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# STEP 10: INITIALIZE RANDOM FOREST

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# STEP 11: TRAIN MODEL

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", rf_model)
])

model.fit(X_train, y_train)

print("\nMODEL TRAINING COMPLETED!")

# STEP 12: MAKE PREDICTIONS

y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

# STEP 13: EVALUATE MODEL

print("\nAccuracy:",
      accuracy_score(y_test, y_pred) * 100, "%")

print("Precision:",
      precision_score(y_test, y_pred, zero_division=0))

print("Recall:",
      recall_score(y_test, y_pred, zero_division=0))

print("F1 Score:",
      f1_score(y_test, y_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# STEP 14: CONFUSION MATRIX GRAPH

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("E-Commerce Return Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# STEP 15: RETURN DISTRIBUTION GRAPH

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="returned",
    order=["No", "Yes"]
)

plt.title("Product Return Distribution")
plt.xlabel("Return Status")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.show()

# STEP 16: FEATURE IMPORTANCE GRAPH

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = (
    model.named_steps["classifier"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES:")
print(importance_df.head(10))

plt.figure(figsize=(9, 6))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()

# STEP 17: SAVE TRAINED MODEL

joblib.dump(
    model,
    "ecommerce_return_model.pkl"
)

print("\nMODEL SAVED SUCCESSFULLY!")
print("File: ecommerce_return_model.pkl")
