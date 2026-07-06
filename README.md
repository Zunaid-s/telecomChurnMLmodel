<<<<<<< HEAD
# 📡 Telecom Customer Churn Prediction App

A professional **Streamlit** web application that serves a trained **Decision Tree Pipeline** (`StandardScaler` + `DecisionTreeClassifier`) to predict whether a telecom customer is likely to churn — complete with probability scores, a polished UI, and downloadable results.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-FF4B4B)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)


## 🔍 Overview

Customer churn is one of the costliest problems in the telecom industry. This app lets an analyst, retention team, or support agent enter a customer's account and usage details and instantly get:


The underlying model was trained and tuned in `ML_Flow_and_Supervised_Learning.ipynb` on the classic [Telecom Churn dataset](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset), then exported as a scikit-learn `Pipeline` (`best_model.pkl`) via `joblib`.


## ✨ Features



## 🧠 Model Details

| Property | Value |
|---|---|
| Pipeline | `StandardScaler` → `DecisionTreeClassifier` |
| Criterion | Entropy |
| Max depth | 5 |
| Min samples per leaf | 4 |
| Features used | 67 |
| Target | `Churn` (0 = No, 1 = Yes) |

### Feature Engineering

The app reproduces the exact feature engineering performed during training:

| Feature | Formula |
|---|---|
| `Total Charges` | `day_charge + eve_charge + night_charge + intl_charge` |
| `Total_Usage` | `day_minutes + eve_minutes + night_minutes + intl_minutes` |
| `Service_Stress` | `customer_service_calls / (account_length + 1)` |
| `Revenue_Segment` | Tertile bucket (`Low` / `Medium` / `High`) of `Total Charges`, using the bin edges learned from the training set |
| `State_*` | One-hot encoded U.S. state (drop-first, `AK` is the baseline) |

Highly correlated raw columns (individual day/eve/night/intl charge fields and `Number vmail messages`) were dropped during training and are **not** part of the final 67-feature vector — they're only used transiently to compute `Total Charges`.


## 📁 Project Structure

```
.
├── app.py                              # Main Streamlit application
├── best_model.pkl                      # Trained sklearn Pipeline (joblib-serialized)
├── requirements.txt                    # Python dependencies
├── telecomChurn.csv                    # Training dataset (optional, for reference)
├── ML_Flow_and_Supervised_Learning.ipynb  # Notebook: EDA, feature engineering, model training
└── README.md                           # You are here
```


## 🚀 Getting Started

### Prerequisites


### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/telecom-churn-predictor.git
cd telecom-churn-predictor

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

> ⚠️ Make sure `best_model.pkl` is in the same directory as `app.py` — the app loads it on startup and will show a friendly error if it's missing.


## 🖥️ Usage

1. Fill in the customer's **account profile** (state, account length, plans, service calls).
2. Enter their **day, evening, night, and international usage** (minutes, calls, charges).
3. Click **🔮 Predict Churn**.
4. Review the **risk verdict** and **probability breakdown**.
5. Optionally expand **"View engineered feature vector"** to inspect the raw model input.
6. Click **⬇️ Download This Prediction (CSV)** to save the result, or download the full **session history**.


## 🛠️ Tech Stack



## 📌 Notes & Limitations



## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute it.


## 🙌 Acknowledgements

=======
# telecomChurnMLmodel
Streamlit app that predicts telecom customer churn using a trained Decision Tree pipeline, with probability scores and downloadable results.
>>>>>>> 0fb3c63 (All files added)
