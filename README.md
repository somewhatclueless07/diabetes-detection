
# Diabetes Detection

A diabetes detection machine learning project involves using data and algorithms to train a model to accurately predict the likelihood of an individual having diabetes based on various features such as Glucose, age, and blood Pressure.

## Description

Diabetes detection machine learning project is a system that uses algorithms, statistical models and historical data to predict the likelihood of an individual having diabetes. The goal is to use features like Glucose, age, and Blood Pressure to identify individuals who have diabetes or are at risk of developing it. Machine learning models are trained using this data and can then be used to make predictions on new individuals.


## About Dataset

### Context
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.

### Content
Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)
- Outcome: Class variable (0 or 1)

### Sources
(a) Original owners: National Institute of Diabetes and Digestive and
Kidney Diseases
(b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
Research Center, RMI Group Leader
Applied Physics Laboratory
The Johns Hopkins University
Johns Hopkins Road
Laurel, MD 20707
(301) 953-6231
(c) Date received: 9 May 1990


## Feature Selection

The process of feature selection in machine learning is used to identify and select the most relevant features from a dataset to improve the performance and efficiency of a machine learning model. It is an important step in the model building process as it can help to reduce overfitting, increase model interpretability, and improve the accuracy of predictions.

### Feature Selection methods use in this projects

#### correlation matrix
A correlation matrix can be used to identify highly correlated features, which can then be removed or consolidated. Highly correlated features can cause a problem in machine learning models as they can introduce multicollinearity, which can lead to unstable and unreliable model estimates.


## Outlier Treatment

Outlier treatment is the process of identifying and handling extreme values or observations that are significantly different from other observations in a dataset. Outliers can have a significant impact on the results of data analysis and modeling, and can skew the mean and standard deviation of a dataset.

### Methods used to identify outliers

- **Visualization:** Outliers can be identified by creating visualizations such as box plots, scatter plots, and histograms to identify observations that fall outside of the typical range.

- **Interquartile range (IQR):** Outliers can be identified by calculating the interquartile range (IQR), which is the difference between the first and third quartile. Values that fall outside of 1.5 * IQR above the third quartile or 1.5 * IQR below the first quartile are considered outliers.

### Methods used to treat outliers

- **Imputing the missing values:** This method can be used to replace outliers with the mean, median, or mode of the dataset.


## Features transformation and scaling

**Feature transformation** is the process of applying mathematical functions to the features in a dataset in order to change their distribution or to extract additional information from the data. Feature transformations are often used to improve the performance of machine learning models by making the features more suitable for the model.

- **Label encoding** is a method of converting categorical variables, represented as text values, into numerical values. It assigns a unique numerical value to each category or level of a categorical feature. This is often used as a preprocessing step before training a machine learning model.

- **Target encoding** is a technique used in machine learning to encode categorical variables. It replaces each category with the average value of the target variable for that category. This can improve the performance of a model by allowing it to better handle categorical variables. It is also known as mean encoding or probability encoding.

**Feature scaling** is the process of normalizing the range of values for each feature in a dataset. This is often done to ensure that all features are on a similar scale and to prevent some features from having a greater impact on the outcome of a machine learning model than others.

- **StandardScaler** is a pre-processing method in machine learning used to standardize a dataset by subtracting the mean and scaling to unit variance. It is commonly used for feature scaling before applying a supervised learning algorithm to a dataset. 


## Model Selection

### Selection of model

- **Logistic regression** is a statistical method used for predicting binary outcomes (i.e. outcomes with two possible results, such as success or failure). It is a type of generalized linear model (GLM) that is used to model a binary dependent variable based on one or more independent variables. 

- **Naive Bayes classifier** is a probabilistic machine learning algorithm that is based on the Bayes' theorem, which states that the probability of a hypothesis (in this case, a class label) given some observed evidence (in this case, a feature vector) is equal to the probability of the evidence given the hypothesis, multiplied by the prior probability of the hypothesis, divided by the overall probability of the evidence.

- **K-Nearest Neighbors (KNN)** is a type of instance-based, or lazy, learning algorithm. It is a classification algorithm that is used to assign a class label to an unlabeled observation based on the class labels of the k-nearest observations to it in feature space.

- **A Decision Tree Classifier** is a type of supervised learning algorithm (having a pre-defined target variable) that is mostly used in classification problems. It works by recursively partitioning the dataset into subsets based on the values of the input features.

- **Support Vector Classifier (SVC)** is a type of supervised learning algorithm that can be used for classification and regression tasks. The main idea behind SVC is to find the best hyperplane (a decision boundary) that separates the different classes in the feature space. The best hyperplane is the one that maximizes the margin, which is the distance between the hyperplane and the closest data points from each class, also known as support vectors.

### Evaluation of algorithm

- **Confusion Matrix**

A confusion matrix is a table that is used to define the performance of a classification algorithm. Each row of the matrix represents the instances in a predicted class, while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabeling one as another). The diagonal elements represent the number of points for which the predicted label is equal to the true label, while off-diagonal elements are those that are mislabeled by the classifier. It is a useful tool for understanding the performance of a classification algorithm, including the types of errors that the classifier is making.

**Confusion Matrix accuracy of Algorithms:**

- Logistic regression: 0.8053435114503816
- Naive Bayes classifier: 0.7709923664122137
- K-Nearest Neighbors (KNN): 0.7366412213740458
- A Decision Tree Classifier: 0.7404580152671756
- Support Vector Classifier (SVC): 0.7977099236641222

We chose the Logistic regression because the highest Confusion matrix accuracy.

### Hyperparameter Tuning

Hyperparameter tuning is the process of selecting the best set of hyperparameters for a machine learning model. Hyperparameters are parameters that are not learned from the data, but are set before training the model.

**Grid search:** is a technique used to tune the hyperparameters of a machine learning model. It is a systematic way of going through multiple combinations of parameter settings, cross-validating as it goes, and returning the best set of parameters that yield the highest performance for a given model. The technique involves specifying a set of values for each hyperparameter, creating a "grid" of all possible combinations of those values, and then training and evaluating a model for each combination of values. 

**Hyperparameters for Logistic regression :**

- solvers: ['newton-cg', 'lbfgs', 'liblinear']
- penalty: ['l2']
- c_values: [100, 10, 1.0, 0.1, 0.01]

Confusion matrix accuracy for Logistic regression After Hyper Tuning(Complete dataset): 0.7760416666666666

So, here we are getting better results with Logistic regression. Hence we are selecting a Logistic regression as the main model for the web app.
  

## Acknowledgements

- This project was inspired by the original work by Nilesh Parab, licensed under the MIT License.