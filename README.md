# ReturnGuard: E-Commerce Product Return Prediction

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/koonipavankumar-design/Ecommerce_Return_Prediction)

An end-to-end Machine Learning web application and REST API designed to predict customer product returns for e-commerce retailers, helping minimize reverse logistics costs and improve order retention.

---

## Overview

Product returns account for substantial margin loss in online retail. **ReturnGuard** analyzes customer order attributes, discounts, product categories, and buyer demographics at the time of purchase to calculate return risk scores and classify orders likely to be returned.

### Key Capabilities
- **Real-Time Order Risk Scoring:** Live single-product prediction via an interactive UI and REST API.
- **Batch CSV Scoring:** Bulk predict thousands of orders with high throughput and instant CSV export.
- **Pipeline Inspector:** Educational breakdown of exploratory data analysis, data leakage prevention, feature engineering, and model serialization.
- **In-Memory Caching:** High-performance model loading with FastAPI lifespan management to ensure sub-50ms inference.
- **Production-Ready Deployment:** Native Render deployment specification via `render.yaml`.

---

## Tech Stack

- **Machine Learning:** `scikit-learn`, `numpy`, `pandas`, `joblib`
- **Backend API:** `FastAPI`, `uvicorn`, `pydantic`
- **Frontend Dashboard:** Modern Responsive HTML5, `Tailwind CSS`, `Font Awesome`
- **Deployment:** `Render`

---

## Feature Schema

The trained model utilizes 11 core features across numeric and categorical variables:

| Feature | Type | Description | Sample Value |
| :--- | :--- | :--- | :--- |
| `category` | Categorical | Product category (`Beauty`, `Electronics`, `Fashion`, `Grocery`, `Home`, `Sports`, `Toys`) | `Fashion` |
| `price` | Numeric | Base unit price in dollars | `145.00` |
| `discount` | Numeric | Applied discount percentage (0–100%) | `25.0` |
| `quantity` | Numeric | Number of units purchased | `1` |
| `payment_method`| Categorical | Method of payment (`Credit Card`, `COD`, `Debit Card`, `PayPal`, `UPI`, `Wallet`) | `COD` |
| `region` | Categorical | Customer geographic region (`Central`, `East`, `North`, `South`, `West`) | `North` |
| `customer_age` | Numeric | Age of the customer | `28` |
| `customer_gender`| Categorical| Customer gender (`Female`, `Male`, `Other`) | `Female` |
| `shipping_cost`| Numeric | Shipping fee charged | `5.00` |
| `total_amount` | Numeric | Net total order amount after discounts | `108.75` |
| `profit_margin`| Numeric | Estimated profit margin percentage | `15.0` |

---

## Local Setup & Quickstart

### Prerequisites
- Python 3.9+ (Python 3.10 or 3.11 recommended)
- `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/koonipavankumar-design/Ecommerce_Return_Prediction.git
cd Ecommerce_Return_Prediction
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser to view the interactive dashboard.

---

## API Reference

### 1. Single Record Prediction
- **Endpoint:** `POST /predict`
- **Request Body:**
  ```json
  {
    "product_name": "Nike Running Shoes",
    "category": "Fashion",
    "price": 145.0,
    "discount": 40.0,
    "quantity": 1,
    "payment_method": "COD",
    "region": "North",
    "customer_age": 24,
    "customer_gender": "Male",
    "shipping_cost": 0.0,
    "total_amount": 87.0,
    "profit_margin": 15.0
  }
  ```
- **Response:**
  ```json
  {
    "prediction": 1,
    "will_return": true,
    "return_probability": 0.742,
    "return_risk_percent": 74.2,
    "product_name": "Nike Running Shoes"
  }
  ```

### 2. Batch CSV Prediction
- **Endpoint:** `POST /predict-csv`
- **Content-Type:** `multipart/form-data`
- **Response:** Streaming CSV download with added columns `predicted_return`, `return_probability`, and `return_risk_percent`.

### 3. Model Information & Health Check
- `GET /health` — Service health status
- `GET /model-info` — Details of the loaded model pipeline, preprocessing transformers, and features

---

## Deployment (Render)

This project is pre-configured for automated cloud deployment on Render:
1. Fork or push this repository to GitHub.
2. In [Render Dashboard](https://dashboard.render.com), click **New +** > **Blueprint**.
3. Select your repository. Render will automatically read [`render.yaml`](./render.yaml) and deploy the FastAPI service.

---

## License

MIT License. Designed and maintained for e-commerce intelligence and return optimization.
