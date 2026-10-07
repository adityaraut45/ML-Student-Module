# Student Performance ML Module

A beginner-friendly Streamlit project based on the supplied student dataset.

## Dataset

The dataset contains:
- student_id
- age
- gender
- study_hours_per_day
- attendance_percent
- previous_score
- sleep_hours
- internet_access
- parental_education
- extracurricular
- final_score
- result

## Features

### Supervised Learning
Classification:
- Logistic Regression
- Decision Tree
- KNN

Regression:
- Linear Regression
- Decision Tree Regressor

The app can:
- train a model
- calculate performance
- predict a new student's result
- predict a new student's final score

### Unsupervised Learning
- K-Means clustering
- selectable numeric features
- cluster table
- cluster summary
- scatter plot

### Data preprocessing
- Missing numeric values -> median
- Missing categorical values -> most frequent category
- Categorical columns -> One-Hot Encoding
- Numeric columns -> Standard Scaling

## Run

Open the terminal in this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Important modeling choice

When predicting `result`, `final_score` is excluded from the input features to avoid target leakage.

When predicting `final_score`, `result` is excluded.
