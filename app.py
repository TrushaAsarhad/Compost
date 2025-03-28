import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data.csv")

# Load trained model
model = joblib.load("compost_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
category_encoder = joblib.load("category_encoder.pkl")

# Define compostability mapping
compost_mapping = {0: "Green", 1: "Brown", 2: "Partial", 3: "Non-Compostable"}

def predict_compostability(food_item, category):
    food_vector = vectorizer.transform([food_item]).toarray()
    category_encoded = category_encoder.transform([category])[0]
    input_features = np.hstack((food_vector, [[category_encoded]]))

    pred = model.predict(input_features)[0]
    compost_status = compost_mapping[pred]

    details = df[(df["Food Item"].str.lower() == food_item.lower()) & (df["Category"] == category)]
    
    if not details.empty:
        method = details["Composting Method"].values[0] if "Composting Method" in details.columns else "Unknown"
        notes = details["Notes"].values[0] if "Notes" in details.columns else "Unknown"
    else:
        method, notes = "Unknown", "Unknown"

    return f"🗑️ **{food_item}** is **{compost_status}**.\n📌 **Method:** {method}\n📖 **Notes:** {notes}"

# Streamlit UI
st.title("♻️ Compostability Predictor")
st.write("Enter a food item and its category to check compostability!")

food_item = st.text_input("Enter food item:", "")
category = st.selectbox("Select category:", df["Category"].unique())

if food_item:
    result = predict_compostability(food_item, category)
    st.markdown(result)
