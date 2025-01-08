# **Fake News Detection Using Machine Learning**

## 🚀 **Overview**
This project demonstrates a **Machine Learning model** that detects fake news based on the **Fake and Real News dataset**. The model is deployed with an interactive **Streamlit UI**,
making it user-friendly and accessible for testing and analysis.

---

## 🛠️ **Features**
- 🌟 **Streamlit-powered UI** for easy interaction.  
- 📊 **Real-time predictions**: Classifies news articles as **Fake** or **Real**.  
- 🧠 **ML Techniques Used**: Text preprocessing, TF-IDF vectorization, Logistic Regression (or the algorithm used).  
- 📂 **Dataset**: Based on the widely used **Fake and Real News dataset**.

---

## 📂 **Dataset**
- **Source**: [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)  
- Contains two categories:
  - `fake.csv`: Articles labeled as fake news.  
  - `real.csv`: Articles labeled as real news.  

---

## 🛠️ **Technologies and Tools**
- **Programming Language**: Python  
- **Libraries**:  
  - Data Handling: `Pandas`, `NumPy`  
  - NLP: `NLTK`, `Scikit-learn`  
  - Deployment: `Streamlit`  
- **Model**: Logistic Regression.  

---

## ⚙️ **Setup Instructions**
Follow these steps to set up the project on your local system:  

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

### **2. Install Dependencies**
Make sure Python is installed . Install the required libraries using:  
```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit App**
Start the Streamlit UI by running:  
```bash
streamlit run app.py
```

---

## 🎯 **How It Works**
1. **Input News**: Enter or paste a news article into the Streamlit app.  
2. **Text Preprocessing**: The text undergoes tokenization, stopword removal, and TF-IDF vectorization.  
3. **Prediction**: The trained ML model predicts whether the article is **Fake** or **Real**.  

---

---

## 📌 **Future Enhancements**
- Enhance the dataset with additional sources for better accuracy.  
- Deploy the app on cloud platforms like **Heroku** or **AWS**.   

---

## 🤝 **Contributions**
Contributions are welcome! If you want to add new features or fix issues, feel free to do that.

---
