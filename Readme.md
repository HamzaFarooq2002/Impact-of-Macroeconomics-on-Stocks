# 📊 Financial ML Streamlit App

A full-featured Streamlit web app to:
- Import stock data (Kaggle or Yahoo Finance)
- Preprocess and adjust prices for inflation
- Engineer features, train linear regression
- Evaluate model accuracy (MSE, R²)
- Visualize predictions

---

## 🔧 Features
- `streamlit_app.py`: Interactive UI for data upload, inflation adjustment, training, prediction
- `inflation.py`: CPI-based inflation adjustment module
- `test_inflation.py`: Unit tests for inflation logic
- `test_model.py`: Unit tests for model training and accuracy
- `requirements.txt`: All dependencies

---

## 🚀 How to Run

```bash
# Clone the repo
https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Create environment and install
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

---

## 🧪 Run Tests
```bash
python -m unittest test_inflation.py
python -m unittest test_model.py
```

---

## 📈 Sample Screenshot
> Upload stock or fetch via ticker → preprocess → apply inflation → feature eng → train → evaluate → export

---

## 🛠 Built With
- Python, Streamlit, Scikit-learn, Plotly, YFinance

---

## 📬 Contact
Author: *Hamza Farooq*  
Roll Number: *I22-2742*  
GitHub: (https://github.com/)
