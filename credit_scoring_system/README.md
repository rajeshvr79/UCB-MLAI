# Credit Scoring System

<center>
    <center>
        <img src = images/credit_scoring_proj.png width = 100%/>
    </center>
</center>

**Objective:** Create a machine learning model that predicts the creditworthiness of loan applicants.

**Skills:** Classification, feature engineering, data preprocessing.

## CRISP-DM (Cross-Industry Standard Process for Data Mining) Framework


CRISP-DM consists of six phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.

<center>
    <center>
        <img src = images/crisp.png width = 70%/>
    </center>
</center>

## Business Understanding

### 1) What is a Credit Scoring System?

A **Credit Scoring System** is a financial tool used to evaluate the creditworthiness of individuals or businesses applying for loans or credit. It is a quantitative assessment of a person’s ability and likelihood to repay debt based on several financial and behavioral factors. The credit score, typically a three-digit number, summarizes a person’s credit history and predicts the risk associated with lending to them.

The system takes into account multiple factors, including:
- **Payment History**: Whether an individual pays their bills on time.
- **Credit Utilization**: The percentage of available credit currently being used.
- **Length of Credit History**: How long the individual has maintained credit accounts.
- **Types of Credit Used**: Credit cards, mortgages, auto loans, etc.
- **Recent Credit Inquiries**: Whether the individual has applied for new credit recently.

Credit scoring models use these factors to generate a score, which lenders use to make decisions about whether to approve or deny credit applications, as well as to determine loan terms (interest rates, loan limits, etc.).

### 2) Why Credit Scoring System?

The use of a credit scoring system is essential for several reasons, both for lenders and borrowers:

### a) Objective and Automated Decision-Making
A credit scoring system automates the process of evaluating potential borrowers, providing an objective and unbiased assessment. Rather than relying on manual evaluation, lenders can use a standardized credit score to make fast, accurate, and consistent decisions across all applicants.

### b) Risk Management for Lenders
Credit scoring allows lenders to assess the risk associated with lending to an individual. By predicting the likelihood of default, lenders can manage their risk exposure, minimize losses from bad debt, and allocate loans based on risk profiles. High-risk borrowers may be offered loans at higher interest rates, while low-risk borrowers might receive favorable terms.

### c) Efficiency and Scalability
Traditional methods of assessing creditworthiness (interviews, manual reviews) are time-consuming and prone to human error or bias. A credit scoring system is scalable and allows lenders to process thousands of applications simultaneously. This is especially important for institutions handling large volumes of credit applications, such as banks, credit unions, and online lenders.

### d) Borrower Awareness and Credit Building
Borrowers can also benefit from credit scoring systems, as they provide clear metrics to understand their credit health. By knowing their credit score, individuals can take steps to improve it over time, leading to better financial opportunities, such as lower interest rates and higher loan approvals. This empowers people to manage their credit more effectively and build financial responsibility.

### e) Personalized Loan Terms
A credit scoring system enables lenders to personalize loan terms based on the borrower's risk profile. Instead of providing the same loan terms for all applicants, lenders can use credit scores to adjust interest rates, repayment periods, and credit limits, tailoring offers to individual needs.

## Summary

In essence, the **Credit Scoring System** is a vital component of modern finance. It helps lenders mitigate risk, promotes financial inclusion, and enhances the efficiency of the lending process. For borrowers, it provides transparency and an opportunity to build or improve their creditworthiness over time.


## Data Set Information:

This dataset was downloaded from UCI repository.

**Source:** https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

## Data Understanding

### 1) Overview of the Dataset
The dataset used for building the Credit Scoring System is typically sourced from financial institutions or publicly available datasets, such as the UCI repository. It contains both numerical and categorical features, representing various aspects of a customer's financial history and personal details. Some common features include:

- **Customer Demographics**: Age, employment status, marital status, etc.
- **Credit Information**: Existing credit history, checking account status, credit amount, and duration of credit.
- **Behavioral Features**: History of timely payments, outstanding debt, etc.
  
### 2) Key Features
- **Existing Checking Account**: Represents the status of the customer’s checking account, a categorical feature that indicates their financial health.
- **Duration**: Numeric feature indicating the duration (in months) of the credit.
- **Credit History**: Categorical feature that describes the past credit behavior of the customer.
- **Purpose**: Categorical feature indicating the purpose for which the credit is requested (e.g., car, education, business).
- **Credit Amount**: Numeric feature showing the amount of credit requested by the customer.
- **Savings Account**: Categorical feature representing the customer's savings status.
- **Employment Since**: Describes the customer’s length of employment, which gives insight into their financial stability.
- **Installment Rate**: The percentage of disposable income allocated to credit installment payments.
- **Age**: Numeric feature representing the age of the customer.

### 3) Target Variable
- The **Target** variable is binary and represents whether a customer is classified as a **good risk** (creditworthy) or **bad risk** (non-creditworthy). The target is usually encoded as 1 for good credit and 0 for bad credit.

### 4) Data Quality Issues
Before moving on to the modeling process, it is important to address the following potential data quality issues:
- **Missing Values**: Checking for missing values and handling them (through imputation or removal).
- **Imbalanced Data**: In some datasets, there may be more "good risk" customers than "bad risk" ones. Techniques like oversampling or undersampling can be employed to balance the data.
- **Outliers**: Extreme values in numeric features like credit amount or age can distort the model’s predictions. Identifying and handling outliers is important.
- **Categorical Encoding**: Categorical variables need to be encoded (using techniques like one-hot encoding or label encoding) for most machine learning models to process the data.

### 5) Feature Correlations and Relationships
It’s crucial to examine correlations between numerical features, as highly correlated features might lead to multicollinearity issues in certain models (e.g., linear regression). We can also analyze how different features interact with the target variable to better understand their impact on creditworthiness.

### 6) Exploratory Data Analysis (EDA)
Exploratory Data Analysis is conducted to gain insights into the dataset before modeling:
- **Histograms** for numerical variables like age, credit amount, and duration help visualize their distributions.
- **Bar plots** for categorical variables like checking account status and credit history provide insight into the frequency of each category.
- **Correlation heatmaps** for numeric features help identify relationships and multicollinearity.
- **Box plots** and **count plots** allow us to explore how different variables interact with the target variable and identify potential patterns.
  
### 7) Conclusion of Data Understanding
A thorough understanding of the data structure, key features, potential quality issues, and exploratory patterns is essential for building an effective credit scoring model. This step lays the foundation for feature engineering, model selection, and performance optimization in the subsequent phases of the project.


## Data Preparation and Visualization
<img src="images/1_Distribution_of_Age.png">

<img src="images/1_Distribution_of_CreditAmount.png">

<img src="images/1_Distribution_of_Duration.png">

<img src="images/2_box_plot_of_Age.png">

<img src="images/2_box_plot_of_CreditAmount.png">

<img src="images/2_box_plot_of_Duration.png">

<img src="images/3_count_plot_of_CreditHistory.png">

<img src="images/3_count_plot_of_EmploymentSince.png">

<img src="images/3_count_plot_of_ExistingCheckingAccount.png">

<img src="images/3_count_plot_of_ForeignWorker.png">

<img src="images/3_count_plot_of_Housing.png">

<img src="images/3_count_plot_of_Job.png">

<img src="images/3_count_plot_of_OtherDebtors.png">

<img src="images/3_count_plot_of_OtherInstallmentPlans.png">

<img src="images/3_count_plot_of_PersonalStatusandSex.png">

<img src="images/3_count_plot_of_Property.png">

<img src="images/3_count_plot_of_Purpose.png">

<img src="images/3_count_plot_of_SavingsAccount.png">

<img src="images/3_count_plot_of_Telephone.png">

<img src="images/4_heatmap_numeric_feature.png">

<img src="images/5_boxplot_credit_target.png">

<img src="images/6_pairplot_numeric_target.png">

<img src="images/7_pairplot_categorical_target.png">


## Train Models and Visualization

### 1. Classification report

Classification Report for Logistic Regression:
              precision    recall  f1-score   support

           0       0.79      0.85      0.82       202
           1       0.85      0.79      0.82       218

    accuracy                            0.82       420
    macro avg       0.82      0.82      0.82       420
    weighted avg    0.82      0.82      0.82       420

Classification Report for Decision Tree:
              precision    recall  f1-score   support

           0       0.69      0.82      0.75       202
           1       0.80      0.67      0.72       218

    accuracy                           0.74       420
    macro avg       0.74      0.74     0.74       420
    weighted avg    0.75      0.74     0.74       420

Classification Report for Random Forest:
              precision    recall  f1-score   support

           0       0.79      0.85      0.82       202
           1       0.85      0.79      0.82       218

    accuracy                           0.82       420
    macro avg       0.82      0.82     0.82       420
    weighted avg    0.82      0.82     0.82       420

Classification Report for SVM:
              precision    recall  f1-score   support

           0       0.81      0.86      0.83       202
           1       0.86      0.81      0.83       218

    accuracy                           0.83       420
    macro avg       0.83      0.83     0.83       420
    weighted avg    0.84      0.83     0.83       420

Classification Report for Gradient Boosting:
              precision    recall  f1-score   support

           0       0.79      0.83      0.81       202
           1       0.84      0.80      0.82       218

    accuracy                           0.81       420
    macro avg       0.81      0.81     0.81       420
    weighted avg    0.82      0.81     0.81       420

Classification Report for k-NN:
              precision    recall  f1-score   support

           0       0.78      0.86      0.82       202
           1       0.86      0.78      0.81       218

    accuracy                           0.82       420
    macro avg       0.82      0.82     0.82       420
    weighted avg    0.82      0.82     0.82       420


### 2. Confusion Matrix Visualization

<img src="images/8_confusion_matrix_Decision Tree.png">

<img src="images/8_confusion_matrix_Gradient Boosting.png">

<img src="images/8_confusion_matrix_Logistic Regression.png">

<img src="images/8_confusion_matrix_Random Forest.png">

<img src="images/8_confusion_matrix_SVM.png">

<img src="images/8_confusion_matrix_k-NN.png">


### 3. ROC Curve and AUC Curve

<img src="images/9_roc_curve.png">


### 4. Accuracy Comparison Bar Plot

<img src="images/10_bar_plot_model_comparison.png">


## Hyperparameter Tuning with Grid Search for Random Forest

### 1. Hyperparameter Tuning for Random Forest

<img src="images/hyperparameter_tuning_rf.png">


### 2. Feature Importance Analysis with Random Forest

<img src="images/feature_importance_rf.png">

### 3. Cross-Validation to Ensure Model Robustness

<img src="images/cross_validation.png">

### 3. Cross-Validation to Ensure Model Robustness

<img src="images/cross_validation.png">

## Summary of Visualizations

### 1. Hyperparameter Tuning Heatmap: Shows the accuracy for each combination of n_estimators and max_depth for the Random Forest. This helps identify the optimal set of hyperparameters.

### 2. Feature Importance Bar Plot: Displays the relative importance of each feature in the final Random Forest model, providing insights into which features contribute most to the model's predictions.

### 3. Cross-Validation Histogram: Illustrates the distribution of accuracy scores across different folds in cross-validation, helping to assess the model's robustness.
