
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib
import io
import os
import shutil

# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="ReturnGuard - E-Commerce Return Prediction",
    description="E-Commerce Product Return Prediction API",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent

HTML_FILE = BASE_DIR / "index.html"
MODEL_FILE = BASE_DIR / "ecommerce_return_model.pkl"
BACKUP_FILE = BASE_DIR / "ecommerce_return_model_backup.pkl"
TEMP_MODEL_FILE = BASE_DIR / "ecommerce_return_model_temp.pkl"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
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
# LOAD TRAINED MODEL
# --------------------------------------------------

def load_model():

    if not MODEL_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="ecommerce_return_model.pkl not found."
        )

    try:
        return joblib.load(MODEL_FILE)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load model: {str(error)}"
        )


# --------------------------------------------------
# HOME PAGE - SERVE HTML
# --------------------------------------------------

@app.get("/")
def home():

    if not HTML_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found. Place it beside main.py."
        )

    return FileResponse(
        str(HTML_FILE),
        media_type="text/html"
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "running",
        "application": "ReturnGuard"
    }


# --------------------------------------------------
# API INFORMATION
# --------------------------------------------------

@app.get("/api")
def api_home():

    return {
        "message": "ReturnGuard FastAPI Backend is Running Successfully!",
        "endpoints": [
            "/",
            "/docs",
            "/health",
            "/model-info",
            "/predict-csv",
            "/retrain"
        ]
    }


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

@app.get("/model-info")
def model_info():

    model = load_model()

    return {
        "model_name": "E-Commerce Return Prediction Model",
        "model_loaded": model is not None,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES
    }


# --------------------------------------------------
# CSV PREDICTION
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

        missing = [
            column for column in FEATURES
            if column not in df.columns
        ]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )

        model = load_model()

        X = df[FEATURES]

        predictions = model.predict(X)

        df["predicted_return"] = predictions

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X)

            classes = list(model.classes_)

            positive_index = next(
                (
                    i for i, value in enumerate(classes)
                    if str(value).strip().lower()
                    in ["yes", "1", "true"]
                ),
                None
            )

            if positive_index is not None:

                return_probabilities = probabilities[:, positive_index]

                df["return_probability"] = return_probabilities

                df["return_risk_percent"] = (
                    return_probabilities * 100
                ).round(2)

        output = io.BytesIO()

        df.to_csv(output, index=False)

        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition":
                'attachment; filename="return_predictions.csv"'
            }
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# MODEL RETRAINING
# --------------------------------------------------

@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:

        contents = await file.read()

        df = pd.read_csv(io.BytesIO(contents))

        required_columns = FEATURES + ["returned"]

        missing = [
            column for column in required_columns
            if column not in df.columns
        ]

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
                detail="returned must contain two valid classes: Yes/No or 1/0."
            )

        if y.value_counts().min() < 2:

            raise HTTPException(
                status_code=400,
                detail="Each class needs at least two records."
            )

        # Preprocessing

        numeric_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ])

        # Random Forest model

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            ))
        ])

        # Split dataset

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Train

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        auc = None

        if hasattr(pipeline, "predict_proba"):

            probabilities = pipeline.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, probabilities)

        # Save new model to temporary file first

        joblib.dump(pipeline, TEMP_MODEL_FILE)

        # Back up existing model before replacing it

        if MODEL_FILE.exists():

            shutil.copy2(MODEL_FILE, BACKUP_FILE)

        # Replace model only after successful training

        os.replace(TEMP_MODEL_FILE, MODEL_FILE)

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

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


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
