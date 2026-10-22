
import os
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend ("backend" is the container name on the Docker network)
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")

# Page title
st.title("SuperKart Sales Prediction")
st.write(
    "Enter the product and store details below to predict the total sales of the product in the store."
)

# Input fields for product and store data
st.subheader("Product Details")
Product_Weight = st.number_input("Product Weight", min_value=0.0, max_value=50.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027, step=0.001, format="%.3f")
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
Product_Id_char = st.selectbox("Product ID Prefix (FD = Food, DR = Drinks, NC = Non-Consumables)", ["FD", "DR", "NC"])
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

st.subheader("Store Details")
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, max_value=100, value=16)

# Create JSON payload
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

# Single Prediction
if st.button("Predict", type='primary'):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=30)
        if response.status_code == 200:
            predicted_sales = response.json()["Sales"]
            st.success(f"Predicted Product Store Sales Total: {predicted_sales:,.2f}")
        else:
            st.error(f"Prediction failed ({response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect to the prediction API: {e}")

# Batch Prediction
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload a CSV file with the same columns as above",
    type=["csv"]
)

if uploaded_file is not None:

    if st.button("Predict for Batch", type='primary'):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": uploaded_file.getvalue()},
                timeout=60
            )
            if response.status_code == 200:
                results = response.json()
                st.success("Predictions completed successfully!")

                # Show the uploaded rows alongside their predicted sales
                input_df = pd.read_csv(uploaded_file)
                input_df["Predicted_Sales"] = [results[str(i)] for i in range(len(input_df))]
                st.dataframe(input_df, use_container_width=True)

                st.download_button(
                    "Download predictions as CSV",
                    input_df.to_csv(index=False).encode("utf-8"),
                    file_name="superkart_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"Prediction failed ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Unable to connect to the prediction API: {e}")
