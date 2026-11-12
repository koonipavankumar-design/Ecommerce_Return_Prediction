
import streamlit as st
import pandas as pd
import joblib

# PAGE SETTINGS

st.set_page_config(
    page_title="E-Commerce Return Prediction",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ E-Commerce Product Return Prediction")
st.write("Enter the order details to predict the return status.")

# LOAD TRAINED MODEL

@st.cache_resource
def load_model():
    return joblib.load("ecommerce_return_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Trained model not found. Please run main.py first."
    )
    st.stop()

# GET TRAINED FEATURE INFORMATION

preprocessor = model.named_steps["preprocessor"]

input_data = {}

st.subheader("Enter Product and Order Details")

# CREATE INPUTS FOR NUMERICAL AND CATEGORICAL FEATURES

for name, transformer, columns in preprocessor.transformers_:

    if name == "num":
        medians = transformer.named_steps[
            "imputer"
        ].statistics_

        for column, median in zip(columns, medians):
            default_value = float(median)

            input_data[column] = st.number_input(
                column.replace("_", " ").title(),
                value=default_value,
                key=f"num_{column}"
            )

    elif name == "cat":
        imputer = transformer.named_steps["imputer"]
        encoder = transformer.named_steps["encoder"]

        for i, column in enumerate(columns):
            options = list(encoder.categories_[i])

            default_value = imputer.statistics_[i]

            if default_value in options:
                default_index = options.index(default_value)
            else:
                default_index = 0

            input_data[column] = st.selectbox(
                column.replace("_", " ").title(),
                options=options,
                index=default_index,
                key=f"cat_{column}"
            )

# PREDICTION BUTTON

if st.button("Predict Return Status", type="primary"):

    input_df = pd.DataFrame(
        [input_data],
        columns=model.feature_names_in_
    )

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Product is likely to be RETURNED")
    else:
        st.success("Prediction: Product is likely NOT to be returned")

    # SHOW PREDICTION PROBABILITY

    probabilities = model.predict_proba(input_df)[0]

    class_list = list(model.classes_)
    return_index = class_list.index(1)

    return_probability = probabilities[return_index] * 100

    st.metric(
        "Estimated Return Probability",
        f"{return_probability:.2f}%"
    )

    st.progress(float(probabilities[return_index]))

st.caption(
    "This is an educational machine learning demonstration. "
    "Predictions are estimates, not guarantees."
)
