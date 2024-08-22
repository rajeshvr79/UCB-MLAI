
# Bank Marketing Campaigns research

This project involves building and evaluating machine learning models using the Bank Marketing dataset. The goal is to predict whether a client will subscribe to a term deposit based on a variety of features.

## Dataset

The dataset used is the **Bank Marketing** dataset, which contains information on various attributes such as age, job, marital status, education, default status, balance, housing loan, personal loan, and more.

- **Source**: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## Project Structure

The project is organized into the following sections:

1. **Data Loading and Exploration**:
    - Load the dataset and explore its structure and contents.
      
  Input variables:
   **Bank client data:**
   1 - age (numeric)
   2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services") 
   3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
   4 - education (categorical: "unknown","secondary","primary","tertiary")
   5 - default: has credit in default? (binary: "yes","no")
   6 - balance: average yearly balance, in euros (numeric) 
   7 - housing: has housing loan? (binary: "yes","no")
   8 - loan: has personal loan? (binary: "yes","no")
   **Last contact of the current campaign:**
   9 - contact: contact communication type (categorical: "unknown","telephone","cellular") 
  10 - day: last contact day of the month (numeric)
  11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  12 - duration: last contact duration, in seconds (numeric)
  **Other attributes:**
  13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
  15 - previous: number of contacts performed before this campaign and for this client (numeric)
  16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

  **Output variable (desired target):**
  17 - y - has the client subscribed a term deposit? (binary: "yes","no")

  ![Data Exploration](visuals/1_hist_plot.png)
  
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

