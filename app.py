import streamlit as st
import pickle
import os

model_path = 'models/best_model.pkl'

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Optional: Read the first few characters to debug
        with open(model_path, 'r', errors='ignore') as f:
            st.text(f"File start: {f.read(50)}")
else:
    st.error("Model file not found.")
