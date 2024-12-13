

<img width="256" alt="Screenshot 2024-12-13 at 11 31 08 AM" src="https://github.com/user-attachments/assets/816be204-bd51-4ba7-9ea6-c3d4e35b5813" />


# Bank Churn Classifier

This repository holds an attempt to predict whether a person's bank account will churn or not using machine learning algorithms with data from the Binary Classification with a Bank Churn Dataset <br> https://www.kaggle.com/competitions/playground-series-s4e1/data?select=test.csv

## Overview

  * **Definition of the tasks / challenge**  The task, as defined by the Kaggle challenge is to predict whether a customer continues with their account or closes it (churn)
  * **Your approach** Ex: The approach in this repository formulates the problem as classification task, using random forest, SVM, and logistic regression as the models with the bank account features as input. We compared the performance of 3 different algorithms.
  * **Summary of the performance achieved**  Our best model was the Random Forest which was able to predict the correct churn outcome 85% of the time

## Summary of Workdone


### Data

* Data:
  * Type: Tabular
    * Input: CSV file containing customer information (features like CreditScore, Age, Tenure, Balance, etc.).
    * Output: "Exited" column indicating whether the customer has churned (1) or not (0).
  * Size:
    * Number of rows: 165034
    * Number of features: 11
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

<img width="665" alt="Screenshot 2024-12-11 at 12 01 42 PM" src="https://github.com/user-attachments/assets/119afac8-e396-424d-8950-63ba8ce1d80d" />



### Problem Formulation

* Define:
  * Input: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
  * Output: Exited Binary Classification
  * Models: <br>

     * Random Forest:
     
        Reason: Random Forest was chosen because it works well with multiple features and can prevent overfitting <br>
        Hyperparameters: 
        * n_estimators: 100 <br>
        * random_state: 42
     
     * Logistic Regression:
     
        Reason: Logistic Regression was chosen as a baseline model because it's easy to understand and works well with binary classification <br>
        Hyperparameters: 
        * max_iter: 1000 
     
     * Support Vector Machine (SVM):
     
        Reason: SVM was chosen because it works well with high dimensional data <br>
        Hyperparameters: 
        * kernel: 'rbf' <br>
        * random_state: 42 

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
  * ROC/AUC: Shows how well the model distinguishes the churned and not churned customers
  * Random Forest Score: 85%
  * Logistic Regression Score: 82%
  * SVM Score: 85%

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








