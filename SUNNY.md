# MACHINE LEARNING AND PREDICTIVE ANALYTICS IN HEALTH CARE
### Developing Models to Predict Outcomes in Healthcare


## Table of Contents  
1. [Introduction](#introduction)  
2. [Methodology](#methodology)  
3. [Data Collection](#data-collection)  
4. [Model Selection](#model-selection)  
5. [Evaluation Metrics](#evaluation-metrics)  
6. [Results and Discussion](#results-and-discussion)  
7. [Conclusion](#conclusion)  
8. [References](#references) 


---

## INTRODUCTION
### PROBLEM DEFINATION

Machine learning enables healthcare prediction because healthcare institutions have better access to patient data that lets them implement earlier interventions employing individualized therapeutic approaches.
The goal of this work utilizes a patient records database to create diagnostic prediction models that will assist clinicians in both better diagnosis and healthcare decision processes.

### CONTEXT AND BACKGROUND

Healthcare predictive analysis performs pattern identification through statistical and machine learning operations on patient information. Logistic regression functions as one of the traditional methods utilized for binary classification since a long time. The predictive power of Support Vector Machines (SVM) and XGBoost increases because they can identify sophisticated patterns present in the data. The assessment of these models in actual healthcare applications becomes possible through analysis of the CSV dataset which contains demographic details and symptom information along with diagnosed diseases.

### OBJECTIVES AND GOALS
The main objectives of our capstone project focus on analytics which we detail in the three points listed below.

- Create multiple machine learning models alongside their evaluation to identify diseases.
- A comparison of the disease classification outcomes between logistic regression, SVM and XGBoost must be evaluated.
- The solution provides data assessment techniques to discover diseases early and enhance medical outcomes.
- The team conducts exploratory data analysis which includes prediction trend analysis by using visualizations.
- The system generates multiple plots containing confusion matrices together with ROC curves and feature importance charts.
- Using identified patient characteristics which affect disease predictions will help healthcare practitioners make better decisions.
- Evaluating the model's resilience requires using hyperparameter adjustment along with cross-validation.

### SUMMARY OF APPROACH

Data analysis through exploratory data analysis (EDA) begins by determining missing values alongside examining data distributions and the main influencing features in disease outcomes. The machine learning model receives high-quality inputs through preprocessing which includes data cleaning together with categorical values encoding and feature scaling procedures.
Three predictive frameworks including Logistic Regression together with Support Vector Machine (SVM) and XGBoost function to perform disease classifications through patient data analysis. The training process includes data splitting and optimized performance through the execution of hyperparameter tuning followed by cross-validation methods. The evaluation of model effectiveness depends on metrics which include accuracy together with precision recall and F1-score and AUC-ROC curves.
The interpretation of results improves through the generation of three specific visual elements: confusion matrices and feature importance plots along with ROC curves. This study demonstrates the significance of disease prediction indicators which help enhance early medical diagnosis platforms for healthcare administrations.

#### LOGISTIC REGRESSION
Logistic Regression functions as a statistical method dedicated to binary classification by determining disease occurrence probabilities.

#### SUPPORT VECTOR MACHINE
 The classification technology Support Vector Machines (SVM) detects the best dividing hyperplane which separates diverse disease classes.
 
#### XGBOOST
XGBoost functions as an algorithm which uses gradient boosting to construct precise predictions from several weak prediction models.

## METHODOLOGY

### DATA AND SOURCES
The analyzed dataset originated from Kaggle under its name "Disease Symptoms and Patient Profile Dataset." The dataset includes both demographic data which includes age and gender alongside the recorded symptoms and confirmed diseases. The collection of data originated from healthcare records and assessments of patient symptoms after anonymous data processing. 
Dataset preprocessing covers three essential steps that involve filling empty data points while converting categories into numbers and normalizing all numerical data for data consistency and prediction accuracy improvement.
Patient attributes consisting of age along with gender and symptoms and diagnosed diseases are included in the "Disease Symptoms and Patient Profile Dataset." The processing stage addresses multiple steps by dealing with null values alongside encoding group data points and creating normalized scales for numerical measurements.

### MATHEMATICAL MODELS 

#### LOGISTIC REGRESSION 
The linear logistic regression model serves as a method for two-category classification. Using the sigmoid function logistic regression identifies the probability that a disease will appear.


$$
P(y=1|X) = \frac{1}{1+e^{- (\beta_0 + \beta_1 X_1 + \dots + \beta_n X_n)} }
$$

#### SUPPORT VECTOR MACHINE
The SVM technique generates a boundary hyperplane which splits data classes inside multi-dimensional spaces. The decision boundary emerges from maximizing the separation distance between data points:

$$
f(X) = w^T X + b
$$

w is the weight vector and b is the bias term.

#### XGBOOST
XGBOOST is the ensemble learning technique that builds multiple decision trees.

```math
L(\theta) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k} \Omega(f_k)
```
l is the loss function and omega is the regularization term.


### EXPERIMENTAL DESIGN AND ANALYTICAL PROCEDURES

The analytical steps inlcude the following;

1. DATA PREPROCESSING:
The preprocessing stage includes addressing data gaps while converting categorical groups into numerical forms (symptoms) and age normalization for numerical attributes.

2. FEATURE SELECTION:
Analysis of disease prediction features requires techniques that use correlation analysis and feature importance ranking methods.

3. MODEL TRAINING:
The baseline model of Logistic Regression received training as a benchmark for describing model performance.Support Vector Machine uses its algorithm to discover the hyperplane which provides maximum separation between different classes.
XGBoost operates through boosting methodology to combine various weak learners that improve classification outcomes.

4. EVALUATION:
The successful measurement of model accuracy requires determining the ratio of accurate predictions and precision assesses the true positive count against all detected positives. Model performance regarding its ability to detect positive cases is measured by recall and its classification capabilities are evaluated through AUC-ROC measurements.

5. INTERPRETATION:
The evaluation of models through performance metrics along with analysis of their feature importance allows practitioners to identify which patient symptoms and characteristics best indicate particular diseases.

### SOFTWARE TOOLS 
