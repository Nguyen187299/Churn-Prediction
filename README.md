

<img width="252" alt="Screenshot 2024-12-11 at 11 42 28 AM" src="https://github.com/user-attachments/assets/192985a6-eac2-4217-aac4-52d2ebccf65e" />



# Bank Churn Classifier

This repository holds an attempt to predict whether a person's bank account will churn or not using machine learning algorithms with data from the Binary Classification with a Bank Churn Dataset <br> https://www.kaggle.com/competitions/playground-series-s4e1/data?select=test.csv

## Overview

  * **Definition of the tasks / challenge**  The task, as defined by the Kaggle challenge is to predict whether a customer continues with their account or closes it (churn)
  * **Your approach** Ex: The approach in this repository formulates the problem as classification task, using random forest, SVC, and logistic regression as the models with the bank account features as input. We compared the performance of 3 different algorithms.
  * **Summary of the performance achieved**  Our best model was able to predict the correct churn outcome 85% of the time

## Summary of Workdone


### Data

* Data:
  * Type: Tabular
    * Input: CSV file containing customer information (features like CreditScore, Age, Tenure, Balance, etc.).
    * Output: "Exited" column indicating whether the customer has churned (1) or not (0).
  * Size: Number of rows: 165034 <br>
          Number of features: 11
  * Instances (Train, Test, Validation Split): 70% for training,
                                               15% for testing,
                                               15% for validation,

#### Preprocessing / Clean up
* Remove duplicates
* Removing unnecessary columns (ex: id, Customerid)
* Scaling
   * MinMaxScaler
* One-Hot Encoding
   * Categorical columns (ex: Geography, Gender)

#### Data Visualization

<img width="757" alt="Screenshot 2024-12-11 at 11 37 35 AM" src="https://github.com/user-attachments/assets/07cda7ee-dcd9-4da2-b182-3c3f5e5199cf" />

<img width="161" alt="Screenshot 2024-12-11 at 11 42 00 AM" src="https://github.com/user-attachments/assets/de31d896-efbc-4137-8df2-92336e8de50b" />


### Problem Formulation

* Define:
  * Input: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
  * Output: Exited Binary Classification
  * Models: <br>

Random Forest:

Reason: Random Forest was chosen because it works well with multiple features and can prevent overfitting
Hyperparameters:
n_estimators: 100 
random_state: 42 

Logistic Regression:

Reason: Logistic Regression was chosen as a baseline model because it's easy to understand and works well with binary classification
Hyperparameters:
max_iter: 1000 

Support Vector Machine (SVM):

Reason: SVM was chosen because it works well with high dimensional data
Hyperparameters:
kernel: 'rbf' 
random_state: 42 

### Training

  * Software: Python
    * Libraries: pandas, scikit-learn, matplotlib
  * Hardware: Standard CPU
  * How long did training take
    * Few minutes due to the size of the dataset
  * Training curves (loss vs epoch for test/train).
    * Check metrics such as accuracy, ROC, and AUC
  * How did you decide to stop training.
    * Training was stopped when satisfactory accuracies were achieved relative to the domain
  * Any difficulties? How did you resolve them?
    * Initially had a training accuracy of 100% but it was resolved after some data preprocessing and tuning

### Performance Comparison

* Key Performance Metrics
  * Accuracy: How often the model makes the correct prediction
  * Precision: How many customers predicted to churn actually churned
  * Recall: How many actual churned customers were correctly identified
  * ROC/AUC: Shows how well the model distinguishes the churned and not churned custoemrs

### Conclusions

* All the models had satisfactory performances with the Random Forest performing the best overall

### Future Work

* Future Improvements:
  * Further model tuning
  * Use ensemble methods
  * Cross-Validation
* Further Studies:
  * Comparison with other companies
  * Behavioral Data

## How to reproduce results

* Data Loading
  * Load the data and look at each of the features such as statistical summaries and visualizations
* Data Cleaning and Preprocessing
  * Remove any missing or invalid data
  * Scale numerical data if needed
  * Encode categorical features
  * Remove duplicate rows
* Model Training
  * Split the data in test, train, and validation
  * Apply the algorithms
  * Tune the model to improve performance

### Overview of files in repository


### Software Setup
* Packages:
  * pandas, scikit-learn, matplotlib, seaborn
* You can install packages using !pip install command

### Data

* The data can be found at https://www.kaggle.com/competitions/playground-series-s4e1/overview

### Training

* Prepare your data by cleaning it, getting rid of any missing values or outliers. Use feature engineering if necessary. Split your dataset into train, test, and validation splits. Initialize and train the model. Evaluate the model and hypertune if necessary.

#### Performance Evaluation

* You can present the performance using a classification report, ROC/AUC curve, or cross-validation


## Citations

https://www.kaggle.com/competitions/playground-series-s4e1/overview








