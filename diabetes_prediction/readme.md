🩺 Diabetes Prediction Using Machine Learning (Support Vector Machine)
🌟 Project Overview

In this project, I built a machine learning model to predict whether a person is diabetic or not based on various medical parameters such as glucose level, blood pressure, BMI, age, etc.
I used the Support Vector Machine (SVM) algorithm — one of the most powerful supervised learning models for classification tasks.

The main goal was to train the system using real-world medical data and then create a predictive system that can identify diabetic patients accurately.

🧩 Step 1: Importing the Required Dependencies

To start with, I imported all the essential Python libraries that support data processing, visualization, and model training.

NumPy → For numerical computations

Pandas → For handling and analyzing datasets

StandardScaler → To standardize features

train_test_split → To split data into training and testing sets

svm → To build the Support Vector Machine model

accuracy_score → To evaluate model performance

These libraries form the foundation of the entire workflow.

📊 Step 2: Data Collection and Initial Analysis

For this project, I used the PIMA Indian Diabetes Dataset, which is publicly available on Kaggle and the UCI Machine Learning Repository.
This dataset contains key health indicators for women, such as glucose level, BMI, insulin levels, and more, along with the outcome (0 → Non-Diabetic, 1 → Diabetic).

Loading and Exploring the Dataset

After loading the dataset using pd.read_csv(), I explored it to understand its structure:

Used head() to preview the first 5 rows

Used shape to check the number of rows and columns

Used describe() for statistical insights

Checked the outcome distribution using value_counts()

To get a better understanding of feature averages per class, I grouped the data using:

diabetes_dataset.groupby("Outcome").mean()


This helped me see which features differ most between diabetic and non-diabetic patients.

🧹 Step 3: Data Preprocessing

Before training the model, I separated the features and labels:

X → All input features

Y → The output label (Outcome column)

Since the dataset contains values of different scales, I applied data standardization using StandardScaler().
Standardization helps the SVM perform better by scaling all features into a similar range.

I used:

scaler.fit_transform(X)


This method both fits and transforms the data in a single step — ensuring the model works efficiently with consistent values.

🧠 Step 4: Splitting the Dataset

Next, I divided the dataset into training and testing sets using an 80-20 split ratio:

80% for training the model

20% for testing its performance

To maintain the same class distribution in both sets, I used stratified sampling.
This ensures that the proportion of diabetic and non-diabetic samples remains balanced across both subsets.

⚙️ Step 5: Training the Support Vector Machine Model

For model training, I used the SVM classifier with a linear kernel since it performs well for binary classification problems.
After fitting the model on the training data, the SVM learned to classify new samples by finding the optimal hyperplane that separates the two classes effectively.

📈 Step 6: Model Evaluation

Once the model was trained, I evaluated its performance using accuracy score.

I calculated:

Accuracy on training data

Accuracy on testing data

This step helped me ensure that the model was not overfitting (performing well on training data but poorly on new data).
The results were consistent, indicating that the model generalized well to unseen data.

🔮 Step 7: Building the Predictive System

Finally, I created a predictive system to test the model on custom input values.

I provided a sample tuple of medical parameters:

(4, 110, 92, 0, 0, 37.6, 0.191, 30)


The process involved:

Converting input into a NumPy array

Reshaping it (since the model predicts for one instance at a time)

Standardizing the input

Predicting the result using the trained SVM model

If the prediction output was 0, it indicated Non-Diabetic,
and if it was 1, it indicated Diabetic.

The system successfully predicted outcomes based on medical parameters, showing that machine learning can play an important role in healthcare analytics.

✅ Conclusion

Through this project, I learned how to:

Collect, clean, and preprocess real-world data

Train and evaluate a supervised machine learning model

Implement SVM for binary classification

Build a simple predictive system capable of real-time inference

This project strengthened my understanding of data preprocessing, model training, and evaluation techniques in machine learning using Python.