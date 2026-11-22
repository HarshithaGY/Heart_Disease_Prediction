# 💜 CARDIO AI – Heart Disease Prediction Using Machine Learning

An interactive machine learning web application that predicts the possibility of heart disease based on user-provided health parameters.

Built using Python, Scikit-learn and Streamlit, this project demonstrates data preprocessing, exploratory data analysis, machine learning model training and prediction.

## 🌟 Project Overview

CARDIO AI is a machine learning-based prediction system developed to explore how classification algorithms can analyse health-related data and generate predictions.

The application provides a simple and interactive interface where users can enter health parameters and receive a model-generated prediction.

**Note:** This project is developed for educational purposes only and is not intended for medical diagnosis.

## ✨ Features

- Interactive web application using Streamlit.
- Heart disease prediction using Machine Learning.
- Data cleaning and preprocessing.
- Exploratory Data Analysis (EDA).
- Comparison of multiple classification algorithms.
- Confusion matrix visualization.
- Model evaluation using accuracy, precision, recall and F1-score.
- Attractive lavender-themed user interface with animations.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| Streamlit | Web application |
| Joblib | Model serialization |

## 📊 Dataset

**Dataset:** Heart Failure Prediction Dataset

**Source:** [Kaggle – Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

- Records: 918
- Features: 11 input features
- Target column: HeartDisease

The dataset contains health-related attributes such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- ST Slope

## 🤖 Machine Learning Algorithms

Three classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

### Model Performance

| Algorithm | Accuracy |
|---|---:|
| Logistic Regression | 89.13% |
| Decision Tree | 77.72% |
| Random Forest | 87.50% |

**Best-performing model:** Logistic Regression, based on the evaluated train-test split.

The dataset was divided into 80% training data and 20% testing data.

## 📁 Project Structure

```text
CARDIO-AI/
│
├── app.py
├── train_model.py
├── data_cleaning.py
├── preprocessing.py
├── eda.py
├── model_training.py
├── model_evaluation.py
│
├── data/
│   └── heart.csv
│
├── heart_model.pkl
├── confusion_matrix.png
├── model_comparison.png
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd CARDIO-AI
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the model

```bash
python train_model.py
```

This generates the trained model file:

```text
heart_model.pkl
```

### 7. Run the application

```bash
streamlit run app.py
```

The application will open in your web browser.

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

The project includes visualizations for model comparison and confusion matrix analysis.

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Data preprocessing and cleaning.
- Exploratory Data Analysis.
- Training and evaluating classification models.
- Comparing machine learning algorithms.
- Building interactive applications using Streamlit.
- Integrating a trained ML model into a web application.

## 👩‍💻 Author

**Harshitha GY**

BE – Computer Science and Engineering

GitHub: [HarshithaGY](https://github.com/HarshithaGY)

LinkedIn: [Connect with me](https://linkedin.com/in/harshitha-gy-6414bb392)

---

⭐ If you find this project interesting, feel free to explore the repository!

**Disclaimer:** This application is an educational machine learning project. Its predictions should not be used as a substitute for professional medical advice or diagnosis.
