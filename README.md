# email-spam-classifier
# 📧 Email Spam Classifier (Logistic Regression)

A machine learning model that classifies emails as **spam** or **not spam** using logistic regression. Built for learning and real-world application of binary classification, feature engineering, and NLP vectorization.

---

## 🚀 Project Overview

This project uses a dataset of email contents with labels indicating whether they are spam (1) or not spam (0). It includes:

- Text preprocessing
- Feature extraction using TF-IDF vectorization
- Training a Logistic Regression model
- Model evaluation and comparison

---

## 🧠 What I Learned

- How to preprocess large textual datasets
- How vectorization (TF-IDF) works in NLP
- How to train and evaluate a binary classifier
- How to interpret metrics like accuracy, MSE, and R²

---

## 🧰 Tech Stack

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

---

## 🗂️ Dataset

- **Source**: `emails.csv`
- **Shape**: 5172 rows × 3002 columns
- **Features**:
  - 3000 columns indicating presence of certain words
  - `Email No.` (ID column)
  - `Prediction` (0 = not spam, 1 = spam)

---

## 📊 Model Performance

| Metric        | Value  |
|---------------|--------|
| Accuracy      | ~97%   |
| Mean Squared Error (MSE) | 0.03 |
| R² Score      | 0.85   |

> Sample Performacnce:
> 
> <img width="453" height="767" alt="image" src="https://github.com/user-attachments/assets/101d4faf-2e3b-4287-be7f-5f4a76f2114f" />


---

## 📦 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/email-spam-classifier.git
   cd email-spam-classifier
