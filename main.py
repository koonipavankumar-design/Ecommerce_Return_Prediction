
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import joblib
import io
import os

app = FastAPI(title="E-Commerce Product Return Prediction API")

BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "index.html"
MODEL_FILE = BASE_DIR / "ecommerce_return_model.pkl"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def load_model():
    if not MODEL_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail="Model file ecommerce_return_model.pkl was not found."
        )
    return joblib.load(MODEL_FILE)


@app.get("/")
def home():
    if not HTML_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html was not found in the project folder."
        )
    return FileResponse(HTML_FILE)


@app.get("/health")
def health():
    return {"status": "running"}


@app.get("/model-info")
def model_info():
    model = load_model()
    return {
        "model_name": "E-Commerce Return Prediction Model",
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_loaded": model is not None
    }


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        missing = [col for col in FEATURES if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )

        model = load_model()
        predictions = model.predict(df[FEATURES])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df[FEATURES])
            classes = list(model.classes_)
            positive_index = next(
                (i for i, value in enumerate(classes)
                 if str(value).strip().lower() in ["yes", "1", "true"]),
                len(classes) - 1
            )
            return_probabilities = probabilities[:, positive_index]
        else:
            return_probabilities = [None] * len(df)

        df["predicted_return"] = predictions
        df["return_probability"] = return_probabilities

        if all(value is not None for value in return_probabilities):
            df["return_risk_percent"] = (
                df["return_probability"] * 100
            ).round(2)

        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        from fastapi.responses import StreamingResponse

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline

        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        required = FEATURES + ["returned"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )

        X = df[FEATURES].copy()
        y = df["returned"].copy()

        if y.dtype == object:
            y = y.astype(str).str.strip().str.lower().map({
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
                "true": 1,
                "false": 0
            })

        if y.isna().any() or y.nunique() != 2:
            raise HTTPException(
                status_code=400,
                detail="The returned column must contain two valid classes: Yes/No or 1/0."
            )

        if y.value_counts().min() < 2:
            raise HTTPException(
                status_code=400,
                detail="Each return class must have at least two rows."
            )

        numeric_transformer = Pipeline([
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ])

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        auc = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probabilities)

        temporary_model = BASE_DIR / "ecommerce_return_model_temp.pkl"
        backup_model = BASE_DIR / "ecommerce_return_model_backup.pkl"

        joblib.dump(pipeline, temporary_model)

        if MODEL_FILE.exists():
            import shutil
            shutil.copy2(MODEL_FILE, backup_model)

        os.replace(temporary_model, MODEL_FILE)

        return {
            "message": "Model retrained successfully",
            "accuracy": round(float(accuracy), 4),
            "roc_auc": round(float(auc), 4) if auc is not None else None,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "backup_created": backup_model.exists()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
