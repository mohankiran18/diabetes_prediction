# Diabetes Prediction Using Support Vector Machine

Binary classification model that predicts whether a patient is diabetic based on clinical health indicators, trained on the PIMA Indian Diabetes Dataset.

---

## Problem Statement

Early detection of diabetes is critical for effective treatment and prevention of complications. Clinical diagnosis often requires specialist review and repeated testing. This project automates the initial screening process using a machine learning model trained on real-world medical data, enabling fast and data-driven risk assessment.

---

## Solution Overview

An SVM classifier with a linear kernel is trained on 768 patient records, each containing 8 health parameters. The trained model accepts new patient input, standardizes the values, and returns a prediction — diabetic or non-diabetic — with accuracy validated on a held-out test set.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Preprocessing | Scikit-learn (StandardScaler) |
| Model | Scikit-learn (SVM — LinearSVC) |
| Evaluation | Scikit-learn (accuracy_score) |
| Environment | Jupyter Notebook / Google Colab |

---

## Dataset

- **Name**: PIMA Indian Diabetes Dataset
- **Source**: UCI Machine Learning Repository / Kaggle
- **Samples**: 768 female patients
- **Features**: 8 clinical parameters
- **Target**: `Outcome` — 0 (Non-Diabetic) or 1 (Diabetic)
- **File**: `diabetes.csv`

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (µU/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Genetic diabetes risk score |
| Age | Patient age in years |

---

## Model Pipeline

```
diabetes.csv (768 × 8 features)
        │
        ▼
Exploratory Data Analysis
(shape, describe, value_counts, groupby mean)
        │
        ▼
Separate Features (X) and Labels (Y)
        │
        ▼
StandardScaler → Normalize feature values
        │
        ▼
Stratified Train/Test Split (80% / 20%)
        │
        ▼
SVM Classifier (Linear Kernel)
        │
        ├──── Training Accuracy
        └──── Test Accuracy
                │
                ▼
     New Patient Input (8 values)
        → Standardize → Predict → Output
```

---

## Project Structure

```
diabetes_prediction/
│
├── diabetes_prediction/
│   ├── diabetes.csv                  # PIMA dataset
│   ├── main.py                       # Training and prediction script
│   └── readme.md
│
├── Diabetes_prediction.ipynb         # Full notebook with EDA and model
└── README.md
```

---

## Installation

**Prerequisites**: Python 3.8 or higher

1. Clone the repository:

```bash
git clone https://github.com/mohankiran18/diabetes_prediction.git
cd diabetes_prediction/diabetes_prediction
```

2. Install dependencies:

```bash
pip install numpy pandas scikit-learn
```

---

## Usage

**Run via script:**

```bash
python main.py
```

**Run via notebook:**

Open `Diabetes_prediction.ipynb` in Jupyter or Google Colab and execute cells sequentially.

To classify a new patient, provide 8 values in this order:

```python
input_data = (Pregnancies, Glucose, BloodPressure, SkinThickness,
              Insulin, BMI, DiabetesPedigreeFunction, Age)

# Example
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
```

**Output:**

```
The person is Non-Diabetic.
```

or

```
The person is Diabetic.
```

---

## Results

| Metric | Value |
|---|---|
| Training Accuracy | _Add from your output_ |
| Test Accuracy | _Add from your output_ |
| Train/Test Split | 80% / 20% |
| Sampling Strategy | Stratified |

> Run the script and paste your printed accuracy scores here. Typical SVM performance on this dataset is 77–80% test accuracy with a linear kernel.

---

## Key Design Decisions

**Why SVM?** SVM with a linear kernel handles binary classification on small, well-structured medical datasets effectively without overfitting. It finds an optimal decision boundary even when classes partially overlap.

**Why StandardScaler?** The 8 features span very different numerical ranges (e.g., Age vs. Insulin). Standardization ensures no single feature dominates the SVM's margin calculation.

**Why stratified split?** The dataset has a class imbalance (~65% non-diabetic, ~35% diabetic). Stratified sampling preserves this ratio in both training and test sets, preventing evaluation bias.

---

## Future Improvements

- Benchmark SVM against Random Forest, XGBoost, and Logistic Regression
- Handle zero-value imputation (Glucose, Insulin, BMI cannot be 0 clinically)
- Add cross-validation for more reliable performance estimates
- Build a Streamlit interface for clinical input and prediction
- Deploy as a REST API for integration with healthcare screening tools

---

## Author

**Mohan Kiran**  
B.Tech — Artificial Intelligence and Machine Learning  
GitHub: [github.com/mohankiran18](https://github.com/mohankiran18)  
Portfolio: [mohan-kiran.netlify.app](https://mohan-kiran.netlify.app/)

---

## License

This project is licensed under the [MIT License](LICENSE).
