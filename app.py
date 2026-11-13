import os
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

app = FastAPI(title="ReturnGuard API", version="1.0")

# Enable CORS for seamless frontend and browser communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "ecommerce_return_model.pkl"

# Input Schema matching the 11 required model features
class ProductInput(BaseModel):
    category: str
    price: float
    discount: float
    quantity: int
    payment_method: str
    region: str
    customer_age: int
    customer_gender: str
    shipping_cost: float
    total_amount: float
    profit_margin: float

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=500, 
            detail="Trained model file (ecommerce_return_model.pkl) not found in directory."
        )
    return joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "ReturnGuard FastAPI Backend is Running Successfully!"}

@app.get("/model-info")
def model_info():
    if not os.path.exists(MODEL_PATH):
        return {"status": "Model not found", "features": [], "pipeline_steps": []}
    model = joblib.load(MODEL_PATH)
    features = [
        "price", "discount", "quantity", "customer_age", 
        "shipping_cost", "total_amount", "profit_margin",
        "category", "payment_method", "region", "customer_gender"
    ]
    pipeline_steps = [step[0] for step in model.steps] if hasattr(model, "steps") else ["Model Pipeline"]
    return {
        "model_file": MODEL_PATH,
        "features": features,
        "pipeline_steps": pipeline_steps
    }

@app.post("/predict")
def predict_return(data: ProductInput):
    model = load_model()
    input_df = pd.DataFrame([data.dict()])
    
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
        
    return {
        "prediction": int(prediction),
        "return_probability": round(float(probability), 4),
        "return_risk_percent": round(float(probability) * 100, 2)
    }

@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        model = load_model()
        
        required_features = [
            "price", "discount", "quantity", "customer_age", 
            "shipping_cost", "total_amount", "profit_margin",
            "category", "payment_method", "region", "customer_gender"
        ]
        
        for f in required_features:
            if f not in df.columns:
                raise HTTPException(status_code=400, detail=f"Missing required column in CSV: {f}")
                
        X = df[required_features]
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]
        
        df["predicted_return"] = preds
        df["return_probability"] = np.round(probs, 4)
        df["return_risk_percent"] = np.round(probs * 100, 2)
        
        output_path = "predicted_output.csv"
        df.to_csv(output_path, index=False)
        
        return FileResponse(output_path, media_type="text/csv", filename="return_predictions.csv")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if "returned" not in df.columns:
            raise HTTPException(status_code=400, detail="Uploaded CSV must contain the target column 'returned'.")
            
        features = [
            "price", "discount", "quantity", "customer_age", 
            "shipping_cost", "total_amount", "profit_margin",
            "category", "payment_method", "region", "customer_gender"
        ]
        
        X = df[features]
        y = df["returned"].apply(lambda val: 1 if str(val).strip().lower() in ["yes", "1", "true"] else 0)
        
        numeric_features = ["price", "discount", "quantity", "customer_age", "shipping_cost", "total_amount", "profit_margin"]
        categorical_features = ["category", "payment_method", "region", "customer_gender"]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        try:
            roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        except:
            roc = 0.0
            
        joblib.dump(clf, MODEL_PATH)
        
        return {
            "message": "Model successfully retrained and saved!",
            "accuracy": round(float(acc), 4),
            "roc_auc": round(float(roc), 4),
            "training_rows": len(X_train),
            "testing_rows": len(X_test)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Retraining error: {str(e)}")