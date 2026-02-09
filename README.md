# Health Insurance Charges Prediction (Regression)

This project predicts health insurance charges using machine learning regression techniques.
The goal is to estimate annual insurance costs based on personal and lifestyle attributes.

---

## Project Overview

- **Problem Type:** Regression
- **Target Variable:** `charges`
- **Dataset:** Health Insurance Dataset (public)
- **Final Model:** Random Forest Regressor
- **Deployment:** Streamlit app deployed on Hugging Face Spaces

---

## Dataset

The dataset contains information about individuals and their medical insurance costs.

**Features:**
- `age` – age of the individual
- `sex` – gender
- `bmi` – body mass index
- `children` – number of dependents
- `smoker` – smoking status
- `region` – residential area

**Target:**
- `charges` – insurance costs

---

## Project Structure

health_insurance_prediction/
├── app.py
├── requirements.txt
├── models/
│ ├── insurance_charges_model.joblib
│ └── training_columns.json
├── notebooks/
│ └── health_insurance_prediction.ipynb
└── README.md


---

## Methodology

### Exploratory Data Analysis (EDA)
- Analysis of target distribution
- Feature inspection and correlation analysis
- Strong influence of smoking status identified

### Preprocessing
- Numerical features scaled using StandardScaler
- Categorical features encoded using OneHotEncoder
- Implemented with sklearn Pipelines and ColumnTransformer

### Modeling
- **Baseline:** Linear Regression
- **Final Model:** Random Forest Regressor
- Random Forest was selected due to its ability to capture non-linear relationships

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Deployment

- Interactive Streamlit web application
- Model pipeline loaded directly for prediction
- Deployed on Hugging Face Spaces

---

## How to Run Locally

```bash
git clone https://github.com/enesbayraktar61/health-insurance-charges-prediction.git
cd health-insurance-charges-prediction
pip install -r requirements.txt
streamlit run app.py

---

Conclusion

This project demonstrates a complete end-to-end machine learning regression workflow,
covering data analysis, preprocessing, model training, evaluation, and deployment using Streamlit.

---

Future Improvements

Hyperparameter tuning

Feature importance visualization

Additional model comparisons
