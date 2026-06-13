import streamlit as st
import numpy as np
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
def load_pipeline():
    model_path = "fasttext_lang.model"
    
    if os.path.exists(model_path):
        ft_model = FastText.load(model_path)
    else:
        st.info("Configuring server language environment... This happens only on first boot.")
        df = pd.read_csv("language.csv")
        X_tokens = [str(text).lower().split() for text in df['Text']]
        ft_model = FastText(sentences=X_tokens, vector_size=100, window=5, min_count=1, workers=4)
    
    xgb_model = XGBClassifier()
    xgb_model.load_model("xgboost_lang.json")
    
    languages = [
        'Arabic', 'Chinese', 'Dutch', 'English', 'Estonian', 'French', 
        'Hindi', 'Indonesian', 'Japanese', 'Korean', 'Latin', 'Persian', 
        'Portugese', 'Pushto', 'Romanian', 'Russian', 'Spanish', 
        'Swedish', 'Tamil', 'Thai', 'Urdu'
    ]
        
    return ft_model, xgb_model, languages

try:
    ft_model, xgb_model, languages_list = load_pipeline()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def vectorize_text(text, model, vector_size=100):
    text_str = str(text).lower().strip()
    
    has_cjk = any(0x4E00 <= ord(char) <= 0x9FFF or 
                  0x3040 <= ord(char) <= 0x30FF or 
                  0xAC00 <= ord(char) <= 0xD7AF for char in text_str)
    
    if has_cjk:
        tokens = [char for char in text_str if not char.isspace()]
    else:
        tokens = text_str.split()
        
    valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
    
    if len(valid_vectors) > 0:
        return np.mean(valid_vectors, axis=0).reshape(1, -1)
    else:
        return np.mean(model.wv.vectors, axis=0).reshape(1, -1)

user_input = st.text_area("Enter Text:", height=150, placeholder="Type something here...")

if st.button("Detect Language", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        vectorized_input = vectorize_text(user_input, ft_model)
        
        pred_encoded = xgb_model.predict(vectorized_input)[0]
        
        predicted_language = languages_list[int(pred_encoded)]
        
        st.success(f"### Predicted Language: **{predicted_language}**")