import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# Load dataset
df = pd.read_csv("fake_and_real_news1.csv")
X = df['Text']
y = df['label']
# Preprocessing
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, 
random_state=42)
# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
