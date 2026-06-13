import streamlit as st
import numpy as np
import pickle
from gensim.models import FastText
from xgboost import XGBClassifier
import re
import os
import pandas as pd

st.set_page_config(page_title="Multilingual Language Detector", page_icon="🌐", layout="centered")

st.title("Multilingual Language Detector")

with st.expander("See all 22 supported languages"):
    st.write("""
    * **Americas & Europe:** English, Spanish, French, Dutch, Swedish, Estonian, Romanian, Portugese, Latin, Russian
    * **Middle East & South Asia:** Arabic, Persian, Urdu, Hindi, Pushto
    * **East & Southeast Asia:** Chinese, Japanese, Korean, Thai, Indonesian
    * **Southern India:** Tamil
    """)

@st.cache_resource
@st.cache_resource
def load_pipeline():
    model_path = "fasttext_lang.model"
    
    # Check if local model file exists (It won't on Streamlit Cloud)
    if os.path.exists(model_path):
        ft_model = FastText.load(model_path)
    else:
        # Fallback: Train FastText instantly on the cloud using your CSV dataset
        st.info("Configuring server language environment... This happens only on first boot.")
        
        # Load your dataset
        df = pd.read_csv("language.csv")
        
        # Tokenize exactly like your training pipeline
        X_tokens = [str(text).lower().split() for text in df['Text']]
        
        # Train FastText (takes ~15 seconds on Streamlit's servers)
        ft_model = FastText(sentences=X_tokens, vector_size=100, window=5, min_count=1, workers=4)
    
    # Load your XGBoost model and Label Encoder natively
    xgb_model = XGBClassifier()
    xgb_model.load_model("xgboost_lang.json")
    
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
        
    return ft_model, xgb_model, label_encoder

try:
    ft_model, xgb_model, label_encoder = load_pipeline()
except Exception as e:
    st.error("Error loading models.")
    st.stop()

def vectorize_text(text, model, vector_size=100):
    text_str = str(text).lower().strip()
    
    # Check if the text belongs to an East Asian script that doesn't use space separations
    # (Chinese: 4E00-9FFF, Japanese: 3040-30FF, Korean: AC00-D7AF)
    has_cjk = any(0x4E00 <= ord(char) <= 0x9FFF or 
                  0x3040 <= ord(char) <= 0x30FF or 
                  0xAC00 <= ord(char) <= 0xD7AF for char in text_str)
    
    if has_cjk:
        # For non-space languages, break the string directly down into single characters
        tokens = [char for char in text_str if not char.isspace()]
    else:
        # For spaced languages (English, Hindi, Arabic, Romanian), standard space split is safest
        tokens = text_str.split()
        
    # Generate vectors natively from FastText
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    
    if len(valid_vectors) > 0:
        return np.mean(valid_vectors, axis=0).reshape(1, -1)
    else:
        # Fallback to the training set's vocabulary background instead of a flat zero vector
        # This keeps XGBoost from breaking on empty inputs
        return np.mean(model.wv.vectors, axis=0).reshape(1, -1)

user_input = st.text_area("Enter Text:", height=150, placeholder="Type something here...")

if st.button("Detect Language", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        vectorized_input = vectorize_text(user_input, ft_model)
        
        pred_encoded = xgb_model.predict(vectorized_input)
        
        predicted_language = label_encoder.inverse_transform(pred_encoded)[0]
        
        st.success(f"### Predicted Language: **{predicted_language}**")