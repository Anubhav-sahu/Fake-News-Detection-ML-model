import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# App title
st.title("Fake News Detection App")

# Input text box
user_input = st.text_area("Enter a news article or headline:")

# Button to make prediction
if st.button("Check"):
    if user_input.strip():
        # Transform input using vectorizer
        user_input_vectorized = vectorizer.transform([user_input])
        
        # Make prediction
        prediction = model.predict(user_input_vectorized)[0]
        confidence = model.predict_proba(user_input_vectorized)[0].max() * 100
        
        # Display results
        if prediction == 'FAKE':
            st.error(f"Prediction: Fake News 🛑 (Confidence: {confidence:.2f}%)")
        else:
            st.success(f"Prediction: Real News ✅ (Confidence: {confidence:.2f}%)")
    else:
        st.warning("Please enter some text!")


# To run the project, use the following command in the terminal:
# streamlit run app.py
