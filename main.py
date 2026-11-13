from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib
import io
import os
import shutil

# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="ReturnGuard",
    description="E-Commerce Product Return Prediction API",
    version="1.0"
)

BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "index.html"
MODEL_FILE = BASE_DIR / "ecommerce_return_model.pkl"
BACKUP_FILE = BASE_DIR / "ecommerce_return_model_backup.pkl"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# MODEL FEATURES
# --------------------------------------------------

FEATURES = [
    "category",
    "price",
    "discount",
    "quantity",
    "payment_method",
    "region",
    "customer_age",
    "customer_gender",
    "shipping_cost",
    "total_amount",
    "profit_margin"
]

NUMERIC_FEATURES = [
    "price",
    "discount",
    "quantity",
    "customer_age",
    "shipping_cost",
    "total_amount",
    "profit_margin"
]

CATEGORICAL_FEATURES = [
    "category",
    "payment_method",
    "region",
    "customer_gender"
]

# --------------------------------------------------
# CACHED MODEL LOADER
# --------------------------------------------------

_cached_model = None

def get_model():
    global _cached_model
    if _cached_model is None:
        if not MODEL_FILE.exists():
            raise HTTPException(
                status_code=500,
                detail="ecommerce_return_model.pkl not found."
            )
        try:
            _cached_model = joblib.load(MODEL_FILE)
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load model: {str(error)}"
            )
    return _cached_model


# --------------------------------------------------
# INPUT SCHEMAS
# --------------------------------------------------

class SinglePredictionInput(BaseModel):
    category: str = Field(default="Electronics")
    price: float = Field(default=100.0, ge=0)
    discount: float = Field(default=0.0, ge=0, le=100)
    quantity: int = Field(default=1, ge=1)
    payment_method: str = Field(default="Credit Card")
    region: str = Field(default="North")
    customer_age: int = Field(default=30, ge=10, le=120)
    customer_gender: str = Field(default="Female")
    shipping_cost: float = Field(default=5.0, ge=0)
    total_amount: Optional[float] = None
    profit_margin: Optional[float] = None
    product_name: Optional[str] = None


# --------------------------------------------------
# HOME & HEALTH
# --------------------------------------------------

@app.get("/")
def home():
    if not HTML_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found. Check your project folder."
        )
    return FileResponse(
        str(HTML_FILE),
        media_type="text/html"
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "application": "ReturnGuard",
        "model_loaded": MODEL_FILE.exists()
    }


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

@app.get("/model-info")
def model_info():
    model = get_model()
    classes = [str(c) for c in getattr(model, "classes_", [])]
    return {
        "model_name": "E-Commerce Return Prediction Pipeline",
        "model_loaded": model is not None,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "classes": classes
    }


# --------------------------------------------------
# SINGLE RECORD PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict_single(data: SinglePredictionInput):
    model = get_model()

    total = data.total_amount
    if total is None:
        total = round(data.price * data.quantity * (1 - data.discount / 100), 2)

    margin = data.profit_margin if data.profit_margin is not None else 15.0

    row = {
        "price": float(data.price),
        "discount": float(data.discount),
        "quantity": int(data.quantity),
        "customer_age": int(data.customer_age),
        "shipping_cost": float(data.shipping_cost),
        "total_amount": float(total),
        "profit_margin": float(margin),
        "category": str(data.category),
        "payment_method": str(data.payment_method),
        "region": str(data.region),
        "customer_gender": str(data.customer_gender)
    }

    input_df = pd.DataFrame([row])

    try:
        prediction = int(model.predict(input_df)[0])
        prob = 0.5

        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            pos_idx = next(
                (i for i, v in enumerate(classes) if str(v).strip().lower() in ["yes", "1", "true", 1]),
                1 if len(classes) > 1 else 0
            )
            prob = float(model.predict_proba(input_df)[0][pos_idx])

        risk_pct = round(prob * 100, 2)
        will_return = bool(prediction == 1 or prob >= 0.5)

        return {
            "prediction": prediction,
            "will_return": will_return,
            "return_probability": round(prob, 4),
            "return_risk_percent": risk_pct,
            "product_name": data.product_name or "Product"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# --------------------------------------------------
# CSV BATCH PREDICTION
# --------------------------------------------------

@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        missing = [column for column in FEATURES if column not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns in CSV: {missing}"
            )

        # Impute missing values gracefully if present
        for col in NUMERIC_FEATURES:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        for col in CATEGORICAL_FEATURES:
            if df[col].isnull().any():
                df[col] = df[col].fillna("Unknown")

        model = get_model()
        X = df[FEATURES]

        predictions = model.predict(X)
        df["predicted_return"] = predictions

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            classes = list(model.classes_)
            positive_index = next(
                (
                    i for i, value in enumerate(classes)
                    if str(value).strip().lower() in ["yes", "1", "true", 1]
                ),
                1 if len(classes) > 1 else 0
            )
            df["return_probability"] = probabilities[:, positive_index].round(4)
            df["return_risk_percent"] = (df["return_probability"] * 100).round(2)

        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="return_predictions.csv"'
            }
        )

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


# --------------------------------------------------
# MODEL RETRAINING
# --------------------------------------------------

@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    global _cached_model

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    temporary_model = BASE_DIR / "ecommerce_return_model_temp.pkl"

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        required_columns = FEATURES + ["returned"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )

        X = df[FEATURES].copy()
        y = df["returned"].copy()

        if y.dtype == object:
            y = (
                y.astype(str)
                .str.strip()
                .str.lower()
                .map({
                    "yes": 1,
                    "no": 0,
                    "1": 1,
                    "0": 0,
                    "true": 1,
                    "false": 0
                })
            )

        if y.isna().any() or y.nunique() != 2:
            raise HTTPException(
                status_code=400,
                detail="Target 'returned' must contain two valid classes: Yes/No or 1/0."
            )

        if y.value_counts().min() < 2:
            raise HTTPException(
                status_code=400,
                detail="Each class needs at least two records."
            )

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ])

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=4,
                random_state=42,
                class_weight="balanced"
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        auc = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probabilities)

        # Save newly trained model temporarily
        joblib.dump(pipeline, temporary_model)

        # Preserve previous model before replacement
        if MODEL_FILE.exists():
            shutil.copy2(MODEL_FILE, BACKUP_FILE)

        os.replace(temporary_model, MODEL_FILE)
        _cached_model = pipeline

        return {
            "message": "Model retrained successfully",
            "accuracy": round(float(accuracy), 4),
            "roc_auc": round(float(auc), 4) if auc is not None else None,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "backup_created": BACKUP_FILE.exists()
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        if temporary_model.exists():
            temporary_model.unlink()


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
