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

# Save the modified app.py file

# Initialize a new Git repository
# Run the following commands in the terminal:
# cd /C:/Users/LENOVO/Desktop/Coding/python/fake news detection/
# git init

# Add the files to the repository
# Run the following command in the terminal:
# git add .

# Commit the changes
# Run the following command in the terminal:
# git commit -m "Initial commit"

# Create a new repository on GitHub

# Add the remote repository URL
# Run the following command in the terminal:
# git remote add origin <remote_repository_url>

# Push the code to GitHub
# Run the following command in the terminal:
# git push -u origin master

# To run the project, use the following command in the terminal:
# streamlit run app.py
