# -------------------------------
# Diabetes Prediction using SVM
# -------------------------------

# Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# -------------------------------
# Data Collection and Analysis
# -------------------------------

# Load the dataset
diabetes_dataset = pd.read_csv("diabetes.csv")

# Display first 5 rows
print(diabetes_dataset.head())

# Dataset info
print("Shape:", diabetes_dataset.shape)
print(diabetes_dataset.describe())

# Check distribution of outcome
print(diabetes_dataset['Outcome'].value_counts())

# Group by outcome
print(diabetes_dataset.groupby("Outcome").mean())

# -------------------------------
# Separating Data and Labels
# -------------------------------
X = diabetes_dataset.drop(columns="Outcome", axis=1)
Y = diabetes_dataset["Outcome"]

print(X)
print(Y)

# -------------------------------
# Data Standardization
# -------------------------------
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset["Outcome"]

print(X)
print(Y)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print(X.shape, X_train.shape, X_test.shape)

# -------------------------------
# Training the Model
# -------------------------------
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

# -------------------------------
# Model Evaluation
# -------------------------------
# Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy score of the training data:", training_data_accuracy)

# Accuracy on testing data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy score of the test data:", test_data_accuracy)

# -------------------------------
# Making a Predictive System
# -------------------------------
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)

# Convert input data to NumPy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape (predicting for one instance)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

# Make prediction
prediction = classifier.predict(std_data)
print("Prediction:", prediction)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
