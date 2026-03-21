# Customer Churn Prediction 

A machine learning project to predict customer churn using classification algorithms. Built with Scikit-learn and deployed with Streamlit for interactive predictions.

##  Project Overview

This project demonstrates:
- **Data preprocessing** & feature scaling (StandardScaler)
- **Classification modeling** (Logistic Regression, Random Forest, SVM)
- **Feature importance** analysis
- **Model evaluation** (accuracy, precision, recall, F1-score, confusion matrix)
- **Interactive Streamlit app** for real-time predictions

##  Dataset

Uses the **Telco Customer Churn dataset** (publicly available).
- **7000+ records** with customer demographics, services, and churn status
- **Features**: tenure, monthly charges, contract type, internet service, etc.
- **Target**: Binary classification (Churn: Yes/No)

##  Project Structure

```
customer-churn-prediction/
├── data/
│   └── churn_data.csv          # Dataset (download from Kaggle)
├── models/
│   └── churn_model.pkl         # Trained model (generated after training)
├── train.py                    # Model training script
├── app.py                      # Streamlit app
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download the Telco Customer Churn dataset from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) and save it to the `data/` folder as `churn_data.csv`.

### 4. Train the Model
```bash
python train.py
```

This will:
- Load and preprocess the data
- Apply feature scaling
- Train multiple models
- Save the best model to `models/churn_model.pkl`
- Display model metrics and feature importance

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start making predictions!

## Model Performance

The trained models achieve:
- **Accuracy**: 82.19%~
- **Precision & Recall**: Balanced for business impact
- **ROC-AUC**: Excellent discrimination between churn/no-churn

*(Metrics will vary based on train-test split and model selection)*

## Streamlit App Features

- **Input fields** for customer attributes (tenure, monthly charges, contract type, etc.)
- **Real-time predictions** with confidence scores
- **Feature importance visualization**
- **Model metrics dashboard**
- **Interactive exploration** of the dataset

## Technologies Used

- **Python 3.13.4**
- **Scikit-learn** — Machine Learning
- **Pandas** — Data manipulation
- **NumPy** — Numerical computing
- **Matplotlib & Seaborn** — Visualization
- **Streamlit** — Web app framework
- **Joblib** — Model serialization

##  Learning Outcomes

This project teaches:
Data preprocessing & handling categorical variables
Feature scaling importance in ML
Classification model selection & comparison
Feature importance interpretation
Model evaluation metrics
Building interactive ML apps with Streamlit
GitHub best practices

## Contributing

Feel free to fork, improve, and submit pull requests!

## Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Kaggle Telco Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

---

**Happy Learning!** Star this repo if you found it helpful!