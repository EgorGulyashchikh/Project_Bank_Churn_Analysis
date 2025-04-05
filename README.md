# Project Bank Churn Analysis

##  Overview
This project analyzes customer churn in a banking context using machine learning. The goal is to predict which customers are likely to leave the bank (churn) based on historical data, helping the bank take proactive retention measures.

**Key Questions:**
- What factors contribute most to customer churn?
- Can we predict churn with high accuracy?
- What actionable insights can the bank derive?

## Project Organization
```
├── images/                             : Contains images
├── static/                             : Plot to show shap values in Flask App 
│   ├── shap_force.png
├── templates/                          : Contains html template for Flask app
│   └── index.html
├── LICENSE.md                          : MIT License
├── README.md                           : Report
├── app.py                              : Flask App
├── churn - EDA.ipynb                   : Exploratory data analysis
├── churn - Model building.ipynb        : Building and comparing models
├── churn.csv                           : Dataset
├── churn_model.pkl                     : XGB_Classifier model
├── train_data.pkl                      : Data for using the model
```

## Dataset

For this project I used a dataset from the Kaggle website - https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers/data  
It contains the following data:
- RowNumber — the record number.
- CustomerId — contains random values.
- Surname — the surname of a customer.
- CreditScore — credit score of a customer.
- Geography — a customer’s location.
- Gender — a customer’s gender.
- Age — a customer’s age.
- Tenure — the number of years that the customer has been a client of the bank.
- Balance — the amount of money in the balance.
- NumOfProducts — the number of products that a customer has purchased through the bank.
- HasCrCard — whether or not a customer has a credit card.
- IsActiveMember — whether or not a customer is an active member.
- EstimatedSalary — a customer’s salary.
- Exited — whether or not the customer left the bank.


## Exploratory data analysis

- **Our dataset contains 10,000 records**

- **Among the 13 features, only 10 are important for the model. 4 of them(CreditScore, Age, Balance, EsyimatedSalary) are numerical. Other 6 are categorical**:
1) Geography - **France**, **Spain** or **Germany**
2) Gender - **Male** or **Female**
3) Tenure - number **between 0 and 10**
4) NumOfProducts - number **between 1 and 4**
5) IsActiveMember and HasCrCard - **1** or **0**

- **Data is highly imbalanced, only 20.37% of records belong to the target class**

- ### Almost a third of the clients from Germany are Exited:
<img src=https://github.com/EgorGulyashchikh/Project_Bank_Churn_Analysis/blob/main/images/Geography.png>

- ### Clients who have 3 or 4 products that were purchased through the bank are most likely to leave
- ### All clients who have 4 products are Exited
<img src=https://github.com/EgorGulyashchikh/Project_Bank_Churn_Analysis/blob/main/images/NumOfProducts.png>


- ### Mean age of Exited clients is higher
<img src=https://github.com/EgorGulyashchikh/Project_Bank_Churn_Analysis/blob/main/images/Age.png>


- ### Median balance of Exited clients is higher among both Males and Females
<img src=https://github.com/EgorGulyashchikh/Project_Bank_Churn_Analysis/blob/main/images/Balance.png>


## Model building

### For categorical feature I used One-Hot Encoding with *drop='first'* to avoid linear dependencies of features

### I chose **F1_score** as scoring metric because it keeps the balance between precision and recall. Both precision and recall are important for business in our case

### I built and compared three different models:
1) Logistic Regression
2) Random Forest
3) Gradient Boosting(XGBoost)

### There were 3 stages for each model:
1) Building a base model
2) Configuring hyperparameters
3) Calibration of the Classification Threshold

### XGB_Classifier turned out to be the best model:
<img src=https://github.com/EgorGulyashchikh/Project_Bank_Churn_Analysis/blob/main/images/F1_score.png>


## Model interpretation

- ### F1_score of our model is **0.65**. It's a good result for the data with a large target class imbalance.

- ### Classification report:
<img src=https://github.com/EgorGulyashchikh/Project_Bank_Churn_Analysis/blob/main/images/Confusion_matrix.png>

- ### Feature importance:
<img src=https://github.com/EgorGulyashchikh/Project_Bank_Churn_Analysis/blob/main/images/Feature_importance.png>

The most important features for our model are **Age**, **Balance**, **NumOfProducts_2**, **Gender_Male**, **Geography_Germany** and **IsActiveMember_1**

- ### SHAP Summary plot:
<img src=https://github.com/EgorGulyashchikh/Project_Bank_Churn_Analysis/blob/main/images/SHAP_summary_plot.png>

### The main conclusions:
1) The higher the age, the higher the probability of churn
2) Clients with 2 products and Active members stay loyal more often
3) A high balance often indicates a churn
4) Males are more likely to remain loyal
5) Customers from Germany are likely to leave


## App for prediction
I created Flask app that can predict probability of churn  
It takes 10 parameters that user will enter and return prediction based on our XGB_Classifier model  
Besides the probability, it also show a SHAP force plot. This plot explains what features have the most impact on the prediction. Red features talk about potential churn, blue ones, on the contrary, indicate that the client will remain  
<img src=https://github.com/EgorGulyashchikh/Project_Bank_Churn_Analysis/blob/main/images/app.png>
