# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.title("EEG → Emotion Detection + Performance Metrics")

# Load model, scaler and metrics
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
metrics = pickle.load(open("metrics.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

uploaded = st.file_uploader("Upload EEG CSV (channels only)", type=["csv"])


def visualize_eeg(df):
    st.subheader("EEG Signal Graph")
    plt.figure(figsize=(10, 5))
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.plot(df[col][:200], label=col)
    plt.legend()
    st.pyplot(plt)


if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    visualize_eeg(df)

    if st.button("Predict Emotion"):

        # FIX 1 — Drop label column if present
        if "label" in df.columns:
            df = df.drop("label", axis=1)

        # FIX 2 — Keep only numeric columns
        df_numeric = df.select_dtypes(include=[np.number])

        if df_numeric.empty:
            st.error("No numeric EEG channels found in the uploaded CSV.")
        else:
            # Scale
            X = scaler.transform(df_numeric)

            # Predict
            pred = model.predict(X)
            decoded = label_encoder.inverse_transform(pred)

            st.success(f"Predicted Emotion: **{decoded[0]}**")

            # Cognitive suggestion
            suggestions = {
                "calm": "Good mental state. Keep going!",
                "neutral": "Stable state. Maintain focus.",
                "stressed": "Take a deep breath or short break.",
                "drowsy": "You're tired. Take a 5-minute rest."
            }

            st.info(suggestions.get(decoded[0], "No suggestion available."))

            # ---------------------
            # PERFORMANCE METRICS
            # ---------------------
            st.subheader("Model Performance (Saved from Training)")

            st.write("Accuracy:", metrics["acc"])

            st.write("Classification Report:")
            st.text(metrics["report"])

            st.write("Confusion Matrix:")
            fig, ax = plt.subplots()
            sns.heatmap(metrics["cm"], annot=True, cmap="Blues", fmt="d", ax=ax)
            st.pyplot(fig)

else:
    st.info("Please upload an EEG CSV file.")
