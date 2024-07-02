# What drives the price of a car?

<img width="793" alt="Screen Shot 2024-07-02 at 9 50 27 AM" src="https://github.com/rajeshvr79/UCB-MLAI/assets/145634280/29ad2294-8e76-4d4c-9dab-6417b6a9fe22">


### OVERVIEW

This project is to identify key drivers for used car prices using the Machine Learning Algorithm.  


### Business Understanding

The used car market constitutes a significant portion of the automotive landscape. The used car market has consistently grown over time. In 2021, worldwide there was a high increase of demand for the used cars due to new cars availability in the market. Projections indicate a steady expansion, with an anticipated compound annual growth rate. The US used vehicles market is one of the list of the strongest ones. In 2021, the US used car dealer market sized reached to its max sales that was never happened in the history. Such a market growth tendency is attributed to improved vehicle durability, escalating prices of new cars, and a wider selection of used vehicles available to consumers. With the history of used car price data, identifying the factors that decides the price of the car will be helpful from Business perspective to meet the demand by particular sector.


### CRISP-DM Framework

<img width="378" alt="Screen Shot 2024-07-02 at 10 26 39 AM" src="https://github.com/rajeshvr79/UCB-MLAI/assets/145634280/371039b4-69a2-4bd1-b9e9-a6f7f42adad7">

CRISP-DM, which stands for Cross-Industry Standard Process for Data Mining, is an industry-proven way to guide your data mining efforts.

The life cycle model consists of six phases with arrows indicating the most important and frequent dependencies between phases. The sequence of the phases is not strict. In fact, most projects move back and forth between phases as necessary.

The CRISP-DM model is flexible and can be customized easily. For example, if your organization aims to detect money laundering, it is likely that you will sift through large amounts of data without a specific modeling goal. Instead of modeling, your work will focus on data exploration and visualization to uncover suspicious patterns in financial data. CRISP-DM allows you to create a data mining model that fits your particular needs.

In such a situation, the modeling, evaluation, and deployment phases might be less relevant than the data understanding and preparation phases. However, it is still important to consider some of the questions raised during these later phases for long-term planning and future data mining goals.


### Data Understanding

After considering the business understanding, we want to get familiar with our data. Writing down the steps that are required to get to know the dataset and identify any quality issues within. This dataset car price consists of 17 columns with numerical/categorical columns and toal 426880 records. VIN is the unique identifier of the car which is good to have in the dataset. The rest of features like year, make, odometer about the car will decide the driving factors for the Modeling.

Below is the structure of the dataset:

<img width="391" alt="Screen Shot 2024-07-02 at 10 47 21 AM" src="https://github.com/rajeshvr79/UCB-MLAI/assets/145634280/77634603-ff12-40d7-95c4-d2f9a8c2b992">






### Data Preparation

After our initial exploration and fine tuning of the business understanding, it is time to construct our final dataset prior to modeling. Here, we want to make sure to handle any integrity issues and cleaning, the engineering of new features, any transformations that we believe should happen (scaling, logarithms, normalization, etc.), and general preparation for modeling with sklearn.

- The dataset had duplicates of VIN. So we need to keep the latest record per VIN by dropping the duplicates. 
- There are also missing values of main columns like VIN, year, odometer, manufacturer, model. The missing values were replaced by using IterativeImputer with BayesianRidge estimator.
- InterQuartile Range (IQR) method is used to remove the outliers from the data
- After the above steps, the dataset has records of 241,827 used car information.
- Identified the categorical columns - cat_cols=['state','region','manufacturer','model','condition','cylinders','fuel','title_status','transmission','drive','size','type','paint_color'] and used sklearn library LabelEncoder to replace the categorical columns into numerical values.
- The dataset is not normally distributed. All the features have different ranges. Without normalization, the ML model will try to disregard coefficients of features that have low values because their impact will be so small compared to the big value. Hence to normalized, sklearn library i.e. MinMaxScaler is used.
- In the regression model, for any fixed value of X, Y is distributed in this problem data-target value (Price ) not normally distributed, it is right skewed. To solve this problem, the log transformation on the target variable is applied when it has skewed distribution and we need to apply an inverse function on the predicted values to get the actual predicted target value.
- 90% of the data was split for the train data and 10% of the data was taken as test data.


### Modeling

Different machine learning algorithms are used to predict price of the car.

1. # Linear Regression
2. # Ridge Regression
3. # Lasso Regression
