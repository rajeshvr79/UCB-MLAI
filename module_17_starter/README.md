# Bank Marketing Campaigns research using Machine Learning

This project involves building and evaluating machine learning models using the Bank Marketing dataset. The goal is to predict whether a client will subscribe to a term deposit based on a variety of features.

## Dataset

The dataset used is the **Bank Marketing** dataset, which contains information on various attributes such as age, job, marital status, education, default status, balance, housing loan, personal loan, and more.

- **Source**: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## Project Structure

The project is organized into the following sections:

1. **Data Loading and Exploration**:
    - Load the dataset and explore its structure and contents.
      Data columns (total 21 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   age             41188 non-null  int64  
 1   job             41188 non-null  object 
 2   marital         41188 non-null  object 
 3   education       41188 non-null  object 
 4   default         41188 non-null  object 
 5   housing         41188 non-null  object 
 6   loan            41188 non-null  object 
 7   contact         41188 non-null  object 
 8   month           41188 non-null  object 
 9   day_of_week     41188 non-null  object 
 10  duration        41188 non-null  int64  
 11  campaign        41188 non-null  int64  
 12  pdays           41188 non-null  int64  
 13  previous        41188 non-null  int64  
 14  poutcome        41188 non-null  object 
 15  emp.var.rate    41188 non-null  float64
 16  cons.price.idx  41188 non-null  float64
 17  cons.conf.idx   41188 non-null  float64
 18  euribor3m       41188 non-null  float64
 19  nr.employed     41188 non-null  float64
 20  y               41188 non-null  object 
dtypes: float64(5), int64(5), object(11)
    - Identify and handle any missing values.
      No missing values
    - Encode categorical variables.
      Categorical columns were encoded using Preprocessing Labelencoder.

2. **Feature Engineering**:
    - Transform and select features for model training.
    - Split the data into training and testing sets.

3. **Baseline Model**:
    - Implement a baseline model using `DummyClassifier` with the strategy `"most_frequent"`.
    - Evaluate the baseline model using various metrics like accuracy, precision, recall, and F1 score.

4. **Modeling**:
    - Train and evaluate multiple classifiers:
        - k-Nearest Neighbors (kNN)
        - Decision Trees
        - Logistic Regression
        - Support Vector Machines (SVM)
    - Compare models based on their performance.

5. **Model Visualization**:
    - Visualize decision boundaries for models using 2D projections.
    - Plot confusion matrices for each model.
    - Visualize ROC curves to compare the models' performance.
    - Plot feature importance for models that support it.

6. **Findings**:
    - Summarize the key findings from the analysis and modeling.
    - Provide actionable insights and recommendations for stakeholders.

7. **Next Steps**:
    - Suggestions for further improvements, including testing additional models, hyperparameter tuning, and more detailed feature engineering.

## Installation and Usage

To run this notebook, you'll need Python 3.x and the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
