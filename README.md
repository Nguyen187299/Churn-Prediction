Duy Derrick Nguyen

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
   * Geography and Gender columns

#### Data Visualization



### Problem Formulation

* Define:
  * Input: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.
  * Output: Exited Binary Classification
  * Models

    Random Forest Classifier:

Reason: Random Forest was chosen because it's a robust, ensemble learning method that tends to work well for classification tasks with a large number of features. It's particularly good at handling nonlinear relationships between features and their interactions.
Hyperparameters:
n_estimators: 100 (number of trees in the forest).
random_state: 42 (to ensure reproducibility).
Logistic Regression:

Reason: Logistic Regression was chosen as a baseline model because it's simple, interpretable, and often performs well on binary classification tasks with well-behaved data.
Hyperparameters:
max_iter: 1000 (to ensure convergence).
Support Vector Machine (SVM):

Reason: SVM was chosen for its ability to perform well in high-dimensional spaces and when the decision boundary between classes is not linear.
Hyperparameters:
kernel: 'rbf' (Radial basis function kernel).
random_state: 42 (to ensure reproducibility).

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







