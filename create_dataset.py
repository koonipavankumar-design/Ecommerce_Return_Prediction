
import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "order_id": range(1001, 1001 + n),
    "customer_id": np.random.randint(100, 500, n),
    "product_id": np.random.randint(1000, 2000, n),
    "category": np.random.choice(
        ["Electronics", "Clothing", "Shoes", "Home", "Books"], n
    ),
    "price": np.random.randint(200, 20000, n),
    "discount": np.random.randint(0, 51, n),
    "quantity": np.random.randint(1, 6, n),
    "payment_method": np.random.choice(
        ["UPI", "Credit Card", "Debit Card", "Cash on Delivery"], n
    ),
    "order_date": pd.date_range(
        "2025-01-01", periods=n, freq="h"
    ).astype(str),
    "delivered_date": pd.date_range(
        "2025-01-02", periods=n, freq="h"
    ).astype(str),
    "region": np.random.choice(
        ["North", "South", "East", "West"], n
    ),
    "customer_age": np.random.randint(18, 65, n),
    "customer_gender": np.random.choice(
        ["Male", "Female", "Other"], n
    ),
    "shipping_cost": np.random.randint(20, 300, n),
})

df["total_amount"] = df["price"] * df["quantity"] * (
    1 - df["discount"] / 100
)

df["profit_margin"] = np.random.uniform(5, 35, n).round(2)

# Generate sample return labels for demonstration
score = (
    0.35 * (df["payment_method"] == "Cash on Delivery")
    + 0.025 * df["discount"]
    + 0.10 * (df["category"] == "Clothing")
    + 0.08 * (df["category"] == "Shoes")
    + 0.04 * df["quantity"]
    - 1.0
)

probability = 1 / (1 + np.exp(-score))

df["returned"] = np.where(
    np.random.random(n) < probability,
    "Yes",
    "No"
)

df.to_csv("ecommerce_returns.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nReturn distribution:")
print(df["returned"].value_counts())
