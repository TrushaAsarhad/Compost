import streamlit as st
import joblib
import pandas as pd

# Load model and dataset
pipeline = joblib.load("compost_model.pkl")
df = pd.read_csv("data.csv")

# Streamlit UI
st.title("♻️ Compostability Predictor")
st.write("Enter a food item to check if it's compostable and learn how to compost it.")

# User Input
food_item = st.text_input("Enter food item:", "")

if st.button("Check Compostability"):
    if food_item:
        # Predict class (1 = Compostable, 0 = Not Compostable)
        pred = pipeline.predict([food_item])[0]

        # Find details
        details = df[df["Food Item"].str.lower() == food_item.lower()]

        if pred == 1:
            method = details["Composting Method"].values[0] if not details.empty else "Unknown"
            notes = details["Notes"].values[0] if not details.empty else "Unknown"
            st.success(f"✅ **{food_item} is compostable!**")
            st.write(f"**🟢 Composting Method:** {method}")
            st.write(f"**📝 Notes:** {notes}")
        else:
            notes = details["Notes"].values[0] if not details.empty else "Unknown"
            st.error(f"❌ **{food_item} is NOT compostable.**")
            st.write(f"**⚠️ Reason:** {notes}")
    else:
        st.warning("Please enter a food item.")

# Run this app using: `streamlit run app.py`
