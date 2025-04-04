import streamlit as st
import pandas as pd
import preprocessing as pp
import constant as ct
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load preprocessed data and model training steps
df = pp.get_data(ct.path, ct.label_code)

# TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words=ct.Stopwords)
x = df["Clean_Ticket_Description"]
y = df["Target"]
x_tfidf = tfidf_vectorizer.fit_transform(x)

# Train model
model = SVC()
model.fit(x_tfidf, y)

# UI
st.title("🎫 Support IT Ticket Classifier")
st.write("Enter your support issue below, and the system will classify it.")

user_input = st.text_area("Enter your IT issue:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a query to classify.")
    else:
        cleaned_input = pp.clean(user_input.lower())
        vectorized_input = tfidf_vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]

        # Find label name from code 
        label_name = [key for key, val in ct.label_code.items() if val == prediction][0]
        st.success(f"✅ The issue is classified as: **{label_name}**")
